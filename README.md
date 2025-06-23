# CDP-FARMentor
Transition Heroes x TU Delft FARMentor project which was done as a part of the CDP course.

## Document explanation
- OR-Tools_TripSolver.py - Imports treated GeoJSON files from GIS/Center and GIS/Outline and attempts to find a solution. See Logistics chapter from report for explanation of solver logic. Outputs a log (log_TripSolver.txt) and GeoJSON file under the same name in GIS/TSP. The files has trips from Depot -> collection route -> Depot
- OR_Tools_RouteCondenser.py - Imports solution GeoJSON from GIS/TSP and attempts to string together trips from OR-Tools_Tripsolver.py as long as total time constraints are still met. Then outputs the new GeoJSON under "GIS/TSP/___optimised.geojson" and calculates the total revenue, price, profit, and profit margin.
- log_TripSolver.txt - log from OR-Tools_TripSolver. Contains progress of solver, as well as the trips, distance, revenue, and price excluding the price of the truck, since that gets calculated in OR_Tools_RoouteCondenser.py.
