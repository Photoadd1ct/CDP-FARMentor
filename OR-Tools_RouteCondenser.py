# Importing packages
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from shapely.geometry import LineString, Point
from typing import Tuple

#Importing data
def ImportFormattedGeoJson(Level: str = "GM" , Folder : str = "./GIS/", ReturnOutline: bool = True , ReturnCenter: bool = True) -> dict: 
    """Returns a dictionary with the requested data at the requested level.

    Parameters
    ----------
    Level : str, optional
        Level of detail of the data. Possible options sorted by number of regions: NL (Netherlands), PV (Province), LG (Agricultural area group), CR (COROP regions), LB (Agricultural area), GM (Municipalities). by default GM (Municipality regions)
    Folder : str, optional
        Location of folder containing GIS data, by default same level as this file "./"
    ReturnOutline : bool, optional
        Whether or not to include the outline file, by default True
    ReturnCenter : bool, optional
        Whether or not to include the Center file, by default True

    Returns
    -------
    dict
        Dictionary containing the outline data as "Outline" and center as "Center".
    """
    Data = {
    "Outline" : gpd.read_file(f"./GIS/Outline/{Level}.geojson"),
    "Center" : gpd.read_file(f"./GIS/Center/{Level}.geojson")
    }

    # Removing outline geometry to increase speed
    Data["Outline"].drop(columns=["geometry"])

    # Setting ID as index
    Data["Outline"] = Data["Outline"].set_index("ID")
    Data["Center"] = Data["Center"].set_index("ID")

    return Data

def FilterFormattedGeoJson(Data: dict, MinManure: float = None, MaxDistance: float = None, IDToRemove: list = []) -> dict:
    """Filters the data based on different properties and splits locations that are too big for a trucks capacity.

    Parameters
    ----------
    Data : dict
        Data imported using ImportFormattedGeoJson
    MinManure : float, optional
        Filter out all rows with less than MinManure production per year in tonnes, by default None
    IDToRemove : list, optional
        List of ID's that should be filtered out, by default None

    Returns
    -------
    dict
        Returns the filtered data
    """

    Outline = Data["Outline"]
    Center = Data["Center"]
    IDToRemove = pd.Index(IDToRemove)

    if MaxDistance != None: 
        Locations = []
        for Municipality in list(Outline.index):
            Locations.append([Data["Center"]["geometry"][Municipality].x, Data["Center"]["geometry"][Municipality].y])
        Locations = np.array(Locations)
        DistanceFromDepot = np.linalg.norm(Locations - Data["Constants"]["DepotLocation"], axis=1)

        MaskToRemove = (DistanceFromDepot > MaxDistance)
        IDToRemove = IDToRemove.union(Outline.index[MaskToRemove])

    Outline = Outline.drop(index = IDToRemove)

    if MinManure != None:
        Outline = Outline[Outline["ManureYear"] >= MinManure]

    Center = Center.filter(list(Outline.index), axis = 0)


    # Splitting Municipalities into truck-sized sections
    Data["Municipalities"] = []
    Data["ManureDay"] = []
    for Municipality in list(Outline.index):
        ManureDayVar = round(Outline["ManureYear"][Municipality] * (1000/365)) # Daily manure in kg
        
        while ManureDayVar > Data["Constants"]["TruckCapacity"]:
            Data["Municipalities"].append(Municipality)
            Data["ManureDay"].append(Data["Constants"]["TruckCapacity"])
            ManureDayVar -= Data["Constants"]["TruckCapacity"]

        # adding the remainder
        Data["Municipalities"].append(Municipality)
        Data["ManureDay"].append(ManureDayVar)

    
    return Data

def GenerateCostMatrix(Data : dict, DepotLocation: Tuple[float, float]) -> dict:

    def GenerateTimeMatrix(IncludeFillTime : bool):
        """Calculates the total driving time and is used to generate the cost matrix."""       

        A = 10_000
        B = 100_000
        C = 75_000

        TruckSpeedHighway = Data["Constants"]["TruckSpeedHighway"]
        TruckSpeedRoad = Data["Constants"]["TruckSpeedRoad"]
        FillTime = Data["Constants"]["FillTime"]

        DistanceMatrix = np.array(Data["DistanceMatrix"], dtype=np.float32)
        HighwayFactor = np.zeros_like(DistanceMatrix)

        # Highway factor masks
        mask_1 = (DistanceMatrix == 0)
        mask_2 = (DistanceMatrix > 0) & (DistanceMatrix < A)
        mask_3 = (DistanceMatrix >= A) & (DistanceMatrix < C)
        mask_4 = (DistanceMatrix >= C) & (DistanceMatrix < B)
        mask_5 = (DistanceMatrix >= B)

        # Apply triangular distribution logic
        HighwayFactor[mask_2] = 0.10
        HighwayFactor[mask_3] = 0.10 + 0.85 * ((DistanceMatrix[mask_3] - A) ** 2) / ((B - A) * (C - A))
        HighwayFactor[mask_4] = 0.95 - 0.85 * ((B - DistanceMatrix[mask_4]) ** 2) / ((B - A) * (B - C))
        HighwayFactor[mask_5] = 0.95

        # Compute time (in minutes)
        TimeMatrix = (
            (DistanceMatrix * HighwayFactor) / TruckSpeedHighway +
            (DistanceMatrix * (1 - HighwayFactor)) / TruckSpeedRoad
        )

        # Only add FillTime where distance > 0
        if IncludeFillTime:
            TimeMatrix[~mask_1] += FillTime

        return TimeMatrix

    # Splitting X and Y data from center points and adding the depot at the beginning
    Locations = []

    Locations.append(Data["Constants"]["DepotLocation"]) # Adding depot at the beginning

    for Municipality in Data["Municipalities"]:
        Locations.append([Data["Center"]["geometry"][Municipality].x, Data["Center"]["geometry"][Municipality].y]) # Adding location of each municipality, with many duplicates

    Data["DistanceMatrix"] = cdist(Locations, Locations, metric='euclidean').round().tolist() # Distance in metres
    Data["WorkTimeMatrix"] = GenerateTimeMatrix(True)
    Data["DriveTimeMatrix"] = GenerateTimeMatrix(False)

    DistanceCostMatrix = np.multiply(Data["DistanceMatrix"], (Data["Constants"]["DieselPrice"] * Data["Constants"]["TruckCons"] / (1E5)))
    TimeCostMatrix = Data["WorkTimeMatrix"] * (Data["Constants"]["DriverPrice"] / 60)
    CostMatrix = DistanceCostMatrix + TimeCostMatrix

    Revenues = np.array(np.multiply(Data["ManureDay"], Data["Constants"]["ManurePrice"]/1000), dtype=np.float32)
    CostMatrix[:,1:] -= Revenues
        
    # Shifting the CostMatrix to make all costs positive.
    MinCost = np.min(CostMatrix)
    if MinCost < 0:
        ShiftValue = abs(MinCost)
    else:
        ShiftValue = 0

    CostMatrix += ShiftValue

    Data["CostMatrix"] = CostMatrix.tolist()
    return Data

def TSPSolver(Data, NumVehicles: int, OptimisationTimeLim: float, MaxAllowedTolerance: float):

    ##### Initialising router #####
    Manager = pywrapcp.RoutingIndexManager(len(Data["CostMatrix"]), NumVehicles, 0)
    Router = pywrapcp.RoutingModel(Manager)

    ##### Making all the nodes optional (disjunctions) #####
    def CalcPenalty(NodeID):
        """Calculates a penalty that is applied if a node is dropped from a trucks path"""
        MunID = Data["Municipalities"][NodeID - 1] # -1 because depot is in 1st position and doesn't exist in Outline
        ManureYear = Data["Outline"]["ManureYear"][MunID] # in tonnes/year
        ManureAtLocation = (ManureYear/365) * Data["Constants"]["DayBetweenPickup"] # in tonnes/day
        PenaltyFactor = 1
        Penalty = round(PenaltyFactor * ManureAtLocation * (Data["Constants"]["ManurePrice"]/1000))
        
        return Penalty

    for Node in range(len(Data["DistanceMatrix"])):
        if Node == 0:
            continue  # Depot is always required

        NodePenalty = CalcPenalty(Node) # Calculates the penalty for each node
        index = Manager.NodeToIndex(Node) # Converting the NodeID to the solver used index 
        Router.AddDisjunction([index], NodePenalty) # Making the node not required and applying the nodepenalty

    ##### Adding a cost to use a truck #####
    for VehicleID in range(NumVehicles):
        Router.SetFixedCostOfVehicle(Data["Constants"]["TruckPrice"], VehicleID)


    ##### Cost callback #####
    def CostCallback(FromIndex, ToIndex):
        FromNode = Manager.IndexToNode(FromIndex)
        ToNode = Manager.IndexToNode(ToIndex)
        return int(Data["CostMatrix"][FromNode][ToNode])

    CostCallbackIndex = Router.RegisterTransitCallback(CostCallback)
    Router.SetArcCostEvaluatorOfAllVehicles(CostCallbackIndex)

    ##### Truck capacity callback #####
    def TruckCapCallback(FromIndex, ToIndex):
        """Calculates how full the truck is and is used to set the limit for truck capacity."""
        ToNode = Manager.IndexToNode(ToIndex)
        if ToNode == 0:
            return int(0)
        else:
            ManureAtLocation = Data["ManureDay"][ToNode-1]
            return int(ManureAtLocation)

    TruckCapCallbackIndex = Router.RegisterTransitCallback(TruckCapCallback)

    Router.AddDimension(
        TruckCapCallbackIndex,
        0,  # no slack
        Data["Constants"]["TruckCapacity"], # Upper limit
        True,  # start cumul to zero
        "Truck capacity",
    )

    TruckCapDimension = Router.GetDimensionOrDie("Truck capacity")

    ##### Driving time callback #####
    def DriveTimeCallback(FromIndex, ToIndex):
        """Calculates the total driving time and is used to set the limit for driving time."""
        FromNode = Manager.IndexToNode(FromIndex)
        ToNode = Manager.IndexToNode(ToIndex)
        TimeRequired = Data["DriveTimeMatrix"][FromNode][ToNode]

        return int(TimeRequired)

    DriveTimeCallbackIndex = Router.RegisterTransitCallback(DriveTimeCallback)

    Router.AddDimension(
        DriveTimeCallbackIndex,
        0,  # no slack
        Data["Constants"]["MaxDriveHours"]*60, # Upper limit
        True,  # start cumul to zero
        "Driving time",
    )

    DriveTimeDimension = Router.GetDimensionOrDie("Driving time")

    ##### Working time callback #####
    def WorkTimeCallback(FromIndex, ToIndex):
        """Calculates the total working time and is used to set the limit for working time."""
        FromNode = Manager.IndexToNode(FromIndex)
        ToNode = Manager.IndexToNode(ToIndex)        
        WorkTime = Data["WorkTimeMatrix"][FromNode][ToNode]
        return int(WorkTime)

    WorkTimeCallbackIndex = Router.RegisterTransitCallback(WorkTimeCallback)
    
    Router.AddDimension(
        WorkTimeCallbackIndex,
        0,  # no slack
        Data["Constants"]["MaxWorkHours"]*60, # Upper limit
        True,  # start cumul to zero
        "Working time",
    )

    WorkTimeDimension = Router.GetDimensionOrDie("Working time")

    ##### Setting the Total manure collected constraint #####
    TotalManureCollected = []

    for VehicleID in range(NumVehicles):
        RouteManureCollected = TruckCapDimension.CumulVar(Router.End(VehicleID))  # Total manure collected on a route
        TotalManureCollected.append(RouteManureCollected)

    MinManureCollected = round((1 - MaxAllowedTolerance)* Data["Constants"]["TotalManureRequired"])
    MaxManureCollected = round((1 + MaxAllowedTolerance) * Data["Constants"]["TotalManureRequired"])

    # Setting the constraint
    Router.solver().Add(Router.solver().Sum(TotalManureCollected) >= MinManureCollected)
    Router.solver().Add(Router.solver().Sum(TotalManureCollected) <= MaxManureCollected)

    ##### Defining the searching methods and limits #####
    SearchParams = pywrapcp.DefaultRoutingSearchParameters()
    SearchParams.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    SearchParams.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    SearchParams.time_limit.seconds = OptimisationTimeLim

    SearchParams.log_search = True
    Solution = Router.SolveWithParameters(SearchParams)
    print(Router.status())
    return Manager, Router, Solution

def SolutionPrinter(Manager, Router, Solution, Verbose = True):
    """Prints solution for all vehicles, including time."""
    TotalDistance = 0
    # TotalCost = 0
    TotalDriveTime = 0
    TotalWorkTime = 0
    TotalManureCollected = 0
    TrucksRequired = 0
    TotalRevenue = 0
    TotalPrice = 0

    DriveTimeDimension = Router.GetDimensionOrDie("Driving time")
    WorkTimeDimension = Router.GetDimensionOrDie("Working time")
    TruckCapDimension = Router.GetDimensionOrDie("Truck capacity")

    for VehicleID in range(Router.vehicles()):
        Index = Router.Start(VehicleID)
        RouteOutput = f"Route for vehicle {VehicleID}:\n"
        # RouteCost = 0
        RouteRevenue = 0
        RoutePrice = 0
        RouteDistance = 0

        while not Router.IsEnd(Index):
            FromNode = Manager.IndexToNode(Index)

            RouteOutput += f" {FromNode} ->"

            Index = Solution.Value(Router.NextVar(Index))
            ToNode = Manager.IndexToNode(Index)
            # RouteCost += Router.GetArcCostForVehicle(IndexOld, Index, VehicleID)
            RouteRevenue += Data["ManureDay"][ToNode - 1] * (Data["Constants"]["ManurePrice"] / 1000)
            RoutePrice += (Data["DistanceMatrix"][FromNode][ToNode] * (Data["Constants"]["DieselPrice"] * Data["Constants"]["TruckCons"] / (1E5))) + (Data["WorkTimeMatrix"][FromNode][ToNode] * (Data["Constants"]["DriverPrice"] / 60))
            RouteDistance += Data["DistanceMatrix"][FromNode][ToNode]

        # Handle final stop
        Node = Manager.IndexToNode(Index)
        RouteDriveTime = Solution.Value(DriveTimeDimension.CumulVar(Index))
        RouteWorkTime = Solution.Value(WorkTimeDimension.CumulVar(Index))
        RouteManureCollected = Solution.Value(TruckCapDimension.CumulVar(Index))

        if RouteDistance > 0:
            if Verbose == True:
                RouteOutput += f" {Node}\n"
                # RouteOutput += f"Cost of the route: {RouteCost:.1f} EUR\n"
                RouteOutput += f"Route distance: {RouteDistance/1000:.1f} km\n"
                RouteOutput += f"Route driving time (working time): {RouteDriveTime/60:.1f} ({RouteWorkTime/60:.1f}) hours\n"
                RouteOutput += f"Route price: {RoutePrice:.2f} EUR\n"
                RouteOutput += f"Route manure: {RouteManureCollected/1000:.1f} tonne\n"
                RouteOutput += f"Route revenue: {RouteRevenue:.2f} EUR\n"
                print(RouteOutput)

            # TotalCost += RouteCost
            TotalDriveTime += RouteDriveTime
            TotalWorkTime += RouteWorkTime
            TotalManureCollected += RouteManureCollected
            TotalDistance += RouteDistance
            TotalPrice += RoutePrice
            TotalRevenue += RouteRevenue
            TrucksRequired += 1

    # print(f"Total cost of all routes: {TotalCost:.2f} EUR")
    print(f"Total trucks required: {TrucksRequired}")
    print(f"Total distance: {TotalDistance/1000:.1f} km/day")
    print(f"Total driving time (working time): {TotalDriveTime/60:.1f} ({TotalWorkTime/60:.1f}) hours")
    print(f"Total manure : {TotalManureCollected/1000:.1f} tonne/day")
    print(f"Total price : {TotalPrice:.2f} EUR/day")
    print(f"Total revenue : {TotalRevenue:.2f} EUR/day")
    print(f"Total profit : {TotalRevenue - TotalPrice:.2f} EUR/day")
    print(f"Profit margin : {100 * (TotalRevenue - TotalPrice)/(TotalRevenue):.2f}%")

def SoltoGeoDF(Manager, Router, Solution):
    # Get dimensions
    DriveTimeDimension = Router.GetDimensionOrDie("Driving time")
    WorkTimeDimension = Router.GetDimensionOrDie("Working time")
    TruckCapDimension = Router.GetDimensionOrDie("Truck capacity")

    VehicleData = []

    for VehicleID in range(Router.vehicles()):
        Index = Router.Start(VehicleID)
        RouteMunicipalities = []
        RouteCoords = []
        RouteRevenue = 0
        RoutePrice = 0
        RouteDistance = 0

        # Adding route
        while not Router.IsEnd(Index):
            FromNode = Manager.IndexToNode(Index)
            if FromNode == 0:
                RouteMunicipalities.append("Depot")
                RouteCoords.append(Point(Data["Constants"]["DepotLocation"]))
            else:
                MunID = Data["Municipalities"][FromNode - 1] # -1 because depot is in 1st position and doesn't exist in Outline
                RouteMunicipalities.append(MunID)
                RouteCoords.append(Data["Center"].loc[MunID, "geometry"])
                RouteRevenue += Data["ManureDay"][FromNode - 1] * (Data["Constants"]["ManurePrice"] / 1000)
            
            Index = Solution.Value(Router.NextVar(Index))
            ToNode = Manager.IndexToNode(Index)
            RoutePrice += (Data["DistanceMatrix"][FromNode][ToNode] * (Data["Constants"]["DieselPrice"] * Data["Constants"]["TruckCons"] / (1E5))) + (Data["WorkTimeMatrix"][FromNode][ToNode] * (Data["Constants"]["DriverPrice"] / 60))
            RouteDistance += Data["DistanceMatrix"][FromNode][ToNode]

        
        # Add final node (depot or last stop)
        RouteMunicipalities.append("Depot")
        RouteCoords.append(Point(Data["Constants"]["DepotLocation"]))
        RouteDriveTime = Solution.Value(DriveTimeDimension.CumulVar(Index))
        RouteWorkTime = Solution.Value(WorkTimeDimension.CumulVar(Index))
        RouteTruckCap = Solution.Value(TruckCapDimension.CumulVar(Index))


        # Build LineString path
        linestring = LineString([pt for pt in RouteCoords])

        VehicleData.append({
            "VehicleID": VehicleID,
            "Municipalities": RouteMunicipalities,
            "RouteDistance": RouteDistance / 1000, # convert to km
            "RouteDriveTime": RouteDriveTime / 60, # convert to hours
            "RouteWorkTime": RouteWorkTime / 60, # convert to hours
            "RouteManureCollected": RouteTruckCap, # kg
            "RouteRevenue" : RouteRevenue, # EUR
            "RoutePrice" : RoutePrice, # EUR
            "geometry": linestring,
        })

    # Convert to GeoDataFrame
    df = pd.DataFrame(VehicleData).set_index("VehicleID")
    df = df[df["RouteDistance"] != 0].reset_index()
    MinVehicleID = df['VehicleID'].min()
    df['VehicleID'] = df['VehicleID'] - MinVehicleID
    df = df.set_index('VehicleID')
    gdf = gpd.GeoDataFrame(df, geometry = df["geometry"], crs="EPSG:28992")

    return gdf

####################################################################
Data = ImportFormattedGeoJson()
Data["Constants"] = {
    "DriverPrice" : 15.36, #EUR/hour
    "DieselPrice" : 1.489, # EUR/L
    "TruckCons" : 34, #L/100 km
    "TruckCapacity" : 40000 - 7365, # kg
    "TruckPrice" : round((50000 + 100000)/(365*10)), # Daily price for a truck in EUR, in this case assuming 10 years
    "DayBetweenPickup" : 1, #Number of days between pickup of manure, acts as multiplyer for manure at location, and divides the number of trucks required. 
    "TruckSpeedHighway" : int(100*1000/60), #m/min
    "TruckSpeedRoad" : int(50*1000/60), # m/min
    "FillTime" : 20, # Time it takes to fill up with manure per location in minutes
    "MaxWorkHours" : 8,
    "MaxDriveHours" : 6,
    "TotalManureRequired" : 8537331, # kg / day
    "DepotLocation" : (261983, 592176), 
    "ManurePrice" : 32 # EUR/tonne
}
Data = FilterFormattedGeoJson(Data, MaxDistance=75*1000)
print("Generating cost matrix...")
Data = GenerateCostMatrix(Data, Data["Constants"]["DepotLocation"])
print("Finding solution...")
Manager, Router, Solution = TSPSolver(Data, 300, round(9.5*60*60), 0.01)

if Solution:
    print("Solution found!")
    SolutionPrinter(Manager, Router, Solution)

    # print("Exporting data...")
    GeoTSP = SoltoGeoDF(Manager, Router, Solution)
    GeoTSP.to_file(f"./GIS/TSP/GM.geojson", driver="GeoJSON")
else:
    print("No solution found")