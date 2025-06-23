import geopandas as gpd
from shapely.geometry import LineString
from ortools.sat.python import cp_model

#Importing data
def ImportSolution() -> dict: 
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
    "Outline" : gpd.read_file(f"./GIS/Outline/GM.geojson"),
    "Center" : gpd.read_file(f"./GIS/Center/GM.geojson"),
    "Solution" : gpd.read_file(f"./GIS/TSP/GM_1.geojson")
    }

    # Removing outline geometry to increase speed
    Data["Outline"].drop(columns=["geometry"])

    # Setting ID as index
    Data["Outline"] = Data["Outline"].set_index("ID")
    Data["Center"] = Data["Center"].set_index("ID")

    return Data

def CombineRoutes(Data, scale=60):
    """
    Groups routes into bins such that:
      - RouteDriveTime per bin <= MaxDriveHours
      - RouteWorkTime per bin <= MaxWorkHours
      - Total RouteRevenue is minimized

    Parameters:
    - data: dict with:
        - "Solution": GeoDataFrame with route data
        - "Constants": dict with MaxDriveHours and MaxWorkHours
    - scale: factor to scale float values to integers (default: 1000)

    Returns:
    - assignment: dict of bin -> list of route indices
    """

    gdf = Data["Solution"]
    max_drive = round(Data["Constants"]["MaxDriveHours"] * scale)
    max_work = round(Data["Constants"]["MaxWorkHours"] * scale)

    n = max(len(gdf), 1)
    routes = list(gdf.index)

    drive_times = [round(val * scale) for val in gdf["RouteDriveTime"]]
    work_times = [round(val * scale) for val in gdf["RouteWorkTime"]]
    revenues = [round(val * scale) for val in gdf["RouteRevenue"]]  # scale to int for objective

    model = cp_model.CpModel()

    # Variables: x[i, j] = 1 if route i assigned to bin j
    x = {}
    for i in range(n):
        for j in range(n):
            x[i, j] = model.NewBoolVar(f'x_{i}_{j}')

    # Each route goes to one bin
    for i in range(n):
        model.Add(sum(x[i, j] for j in range(n)) == 1)

    # Drive and work time constraints per bin
    for j in range(n):
        model.Add(sum(drive_times[i] * x[i, j] for i in range(n)) <= max_drive)
        model.Add(sum(work_times[i] * x[i, j] for i in range(n)) <= max_work)

    # Minimize total RouteRevenue

    bin_used = {}
    fixed_bin_cost = 10**6
    for j in range(n):
        bin_used[j] = model.NewBoolVar(f'bin_used_{j}')
        for i in range(n):
            model.Add(x[i, j] <= bin_used[j])
    total_revenue = sum(revenues[i] * x[i, j] for i in range(n) for j in range(n))
    total_bin_cost = sum(bin_used[j] * fixed_bin_cost for j in range(n))
    model.Minimize(total_revenue + total_bin_cost)

    # Solve
    solver = cp_model.CpSolver()
    # solver.parameters.log_search_progress = True
    solver.parameters.max_time_in_seconds = 60
    status = solver.Solve(model)

    assignment = {}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for j in range(n):
            group = []
            for i in range(n):
                if solver.BooleanValue(x[i, j]):
                    group.append(i)
            if group:
                assignment[j] = group
    else:
        print("No feasible solution found.")

    return assignment

def merge_assigned_routes(gdf, assignment, output_path="GIS/TSP/GM_1_optimised.geojson"):
    merged_features = []

    for new_id, route_indices in enumerate(assignment.values()):
        combined_coords = []
        combined_municipalities = ""
        total_properties = {}

        for i, route_idx in enumerate(route_indices):
            row = gdf.loc[route_idx]
            geom = row.geometry
            coords = list(geom.coords)

            # Geometry: remove start depot from all but first
            if i == 0:
                combined_coords.extend(coords)
                combined_municipalities += row["Municipalities"] +  ","
            else:
                combined_coords.extend(coords[1:])
                combined_municipalities += row["Municipalities"][6:] + ","
            
            # Sum up all numerical columns
            for col in gdf.columns:
                if col not in ("geometry", "Municipalities"):
                    val = row[col]
                    if isinstance(val, (int, float)):
                        total_properties[col] = total_properties.get(col, 0) + val
        
        combined_municipalities = combined_municipalities[:-1]

        total_properties["RoutePrice"] += Data["Constants"]["TruckPrice"]

        # Final geometry and feature creation
        merged_geom = LineString(combined_coords)
        feature = {
            "geometry": merged_geom,
            "VehicleID": new_id,
            "Municipalities": combined_municipalities,
        }

        # Add summed properties
        feature.update(total_properties)

        merged_features.append(feature)

    # Create GeoDataFrame
    merged_gdf = gpd.GeoDataFrame(merged_features, crs=gdf.crs)

    # Save to GeoJSON
    merged_gdf.to_file(output_path, driver="GeoJSON")

    print(f"Merged GeoJSON saved to: {output_path}")
    return merged_gdf

Data = ImportSolution()
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
assignments = CombineRoutes(Data)
for bin_id, routes in assignments.items():
    print(f"Bin {bin_id}: Routes {routes}")
Optimised = merge_assigned_routes(Data["Solution"], assignments)

TotalPrice = sum(Optimised["RoutePrice"])
TotalRevenue = sum(Optimised["RouteRevenue"])

print(f"Total Revenue : {TotalRevenue}")
print(f"Total Price : {TotalPrice}")
print(f"Total Profit : {TotalRevenue - TotalPrice}")
print(f"Total Revenue : {100 * (TotalRevenue - TotalPrice)/(TotalRevenue)}")