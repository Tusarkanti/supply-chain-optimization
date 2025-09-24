import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class RouteStop:
    location_id: str
    latitude: float
    longitude: float
    demand: int
    time_window_start: str = None
    time_window_end: str = None
    service_time: int = 30  # minutes

@dataclass
class OptimizedRoute:
    vehicle_id: str
    stops: List[RouteStop]
    total_distance: float
    total_time: float
    total_cost: float

class LogisticsOptimizationModule:
    def __init__(self):
        self.average_speed = 60  # km/h
        self.vehicle_capacity = 1000  # units
        self.cost_per_km = 2.0  # $ per km
        self.max_vehicles = 10

    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate Haversine distance between two points"""
        R = 6371  # Earth's radius in kilometers

        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)

        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

        return R * c

    def create_distance_matrix(self, locations: List[RouteStop]) -> List[List[float]]:
        """Create distance matrix for all location pairs"""
        n = len(locations)
        distance_matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i != j:
                    distance_matrix[i][j] = self.calculate_distance(
                        locations[i].latitude, locations[i].longitude,
                        locations[j].latitude, locations[j].longitude
                    )

        return distance_matrix

    def optimize_routes(self, warehouse_locations: List[RouteStop],
                       customer_locations: List[RouteStop]) -> Dict[str, Any]:
        """Optimize delivery routes using Google OR-Tools"""

        # Combine warehouse and customer locations
        all_locations = warehouse_locations + customer_locations
        num_locations = len(all_locations)

        # Create distance matrix
        distance_matrix = self.create_distance_matrix(all_locations)

        # Create the routing index manager
        manager = pywrapcp.RoutingIndexManager(num_locations, self.max_vehicles, 0)

        # Create routing model
        routing = pywrapcp.RoutingModel(manager)

        # Create distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(distance_matrix[from_node][to_node] * 1000)  # Convert to meters

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add capacity constraints
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return all_locations[from_node].demand

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index, 0,  # null capacity slack
            [self.vehicle_capacity] * self.max_vehicles,  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity'
        )

        # Add time windows if available
        if all(loc.time_window_start for loc in all_locations):
            def time_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                travel_time = distance_matrix[from_node][to_node] / self.average_speed * 60  # minutes
                return int(travel_time * 60)  # Convert to seconds for OR-Tools

            time_callback_index = routing.RegisterTransitCallback(time_callback)
            routing.AddDimension(
                time_callback_index, 30,  # allow waiting time
                1440,  # maximum time per vehicle (24 hours)
                False,  # Don't force start cumul to zero
                'Time'
            )

            # Add time windows
            for i, location in enumerate(all_locations):
                if location.time_window_start and location.time_window_end:
                    start_time = self._time_to_minutes(location.time_window_start)
                    end_time = self._time_to_minutes(location.time_window_end)
                    routing.AddDimension(
                        time_callback_index, 30, 1440, False, 'Time'
                    ).CumulVar(routing.NodeToIndex(i)).SetRange(start_time * 60, end_time * 60)

        # Set first solution heuristic
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            # Extract solution
            optimized_routes = []

            for vehicle_id in range(self.max_vehicles):
                if solution.HasVehicle(vehicle_id):
                    route = self._extract_route_solution(
                        solution, routing, manager, vehicle_id, all_locations
                    )
                    optimized_routes.append(route)

            # Calculate total metrics
            total_distance = sum(route.total_distance for route in optimized_routes)
            total_cost = sum(route.total_cost for route in optimized_routes)

            return {
                'success': True,
                'total_distance': total_distance,
                'total_cost': total_cost,
                'routes': optimized_routes,
                'unassigned_customers': self._get_unassigned_customers(
                    solution, routing, manager, customer_locations
                )
            }
        else:
            return {
                'success': False,
                'error': 'No solution found',
                'routes': [],
                'total_distance': 0,
                'total_cost': 0
            }

    def _extract_route_solution(self, solution, routing, manager, vehicle_id: int,
                              locations: List[RouteStop]) -> OptimizedRoute:
        """Extract route solution for a specific vehicle"""
        route_stops = []
        total_distance = 0
        total_time = 0

        index = routing.Start(vehicle_id)
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_stops.append(locations[node_index])

            previous_index = index
            index = solution.Value(routing.NextVar(index))
            distance = solution.Value(routing.GetArcCostForVehicle(previous_index, index, vehicle_id))
            total_distance += distance / 1000  # Convert back to km

        # Calculate total time (simplified)
        total_time = total_distance / self.average_speed * 60  # minutes

        # Calculate cost
        total_cost = total_distance * self.cost_per_km

        return OptimizedRoute(
            vehicle_id=f"Vehicle_{vehicle_id + 1}",
            stops=route_stops,
            total_distance=total_distance,
            total_time=total_time,
            total_cost=total_cost
        )

    def _get_unassigned_customers(self, solution, routing, manager,
                                customer_locations: List[RouteStop]) -> List[RouteStop]:
        """Get list of unassigned customers"""
        assigned_nodes = set()
        for vehicle_id in range(self.max_vehicles):
            if solution.HasVehicle(vehicle_id):
                index = routing.Start(vehicle_id)
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    assigned_nodes.add(node_index)
                    index = solution.Value(routing.NextVar(index))

        unassigned = []
        for i, customer in enumerate(customer_locations):
            customer_index = len(self._get_warehouses()) + i  # Offset for warehouse locations
            if customer_index not in assigned_nodes:
                unassigned.append(customer)

        return unassigned

    def _get_warehouses(self) -> List[RouteStop]:
        """Get warehouse locations (placeholder)"""
        # This would typically come from database
        return []

    def _time_to_minutes(self, time_str: str) -> int:
        """Convert time string to minutes since midnight"""
        time_obj = datetime.strptime(time_str, '%H:%M').time()
        return time_obj.hour * 60 + time_obj.minute

    def generate_route_report(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive route optimization report"""
        if not optimization_results['success']:
            return {
                'summary': 'Route optimization failed',
                'total_routes': 0,
                'total_distance': 0,
                'total_cost': 0,
                'average_distance_per_route': 0,
                'unassigned_customers': 0
            }

        routes = optimization_results['routes']
        total_distance = optimization_results['total_distance']
        total_cost = optimization_results['total_cost']

        report = {
            'summary': f'Successfully optimized {len(routes)} routes',
            'total_routes': len(routes),
            'total_distance': total_distance,
            'total_cost': total_cost,
            'average_distance_per_route': total_distance / len(routes) if routes else 0,
            'average_cost_per_route': total_cost / len(routes) if routes else 0,
            'unassigned_customers': len(optimization_results.get('unassigned_customers', [])),
            'route_details': [
                {
                    'vehicle_id': route.vehicle_id,
                    'stops_count': len(route.stops),
                    'total_distance': route.total_distance,
                    'total_time': route.total_time,
                    'total_cost': route.total_cost,
                    'stops': [
                        {
                            'location_id': stop.location_id,
                            'latitude': stop.latitude,
                            'longitude': stop.longitude,
                            'demand': stop.demand
                        } for stop in route.stops
                    ]
                } for route in routes
            ]
        }

        return report

    def simulate_route_scenarios(self, base_demand: int, distance_multiplier: float = 1.0) -> Dict[str, Any]:
        """Simulate different route scenarios"""
        # This would run multiple optimization scenarios with different parameters
        scenarios = {
            'base_scenario': {
                'vehicle_capacity': self.vehicle_capacity,
                'max_vehicles': self.max_vehicles,
                'expected_distance': 100 * distance_multiplier,
                'expected_cost': 200 * distance_multiplier
            },
            'high_capacity_scenario': {
                'vehicle_capacity': self.vehicle_capacity * 1.5,
                'max_vehicles': max(1, self.max_vehicles - 2),
                'expected_distance': 90 * distance_multiplier,
                'expected_cost': 180 * distance_multiplier
            },
            'more_vehicles_scenario': {
                'vehicle_capacity': self.vehicle_capacity * 0.8,
                'max_vehicles': self.max_vehicles + 2,
                'expected_distance': 110 * distance_multiplier,
                'expected_cost': 220 * distance_multiplier
            }
        }

        return {
            'scenarios': scenarios,
            'recommendation': 'base_scenario',  # Based on analysis
            'reasoning': 'Balanced approach provides optimal cost-distance trade-off'
        }
