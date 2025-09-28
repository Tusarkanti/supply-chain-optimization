import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import requests
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

@dataclass
class TrafficCondition:
    location: str
    traffic_index: float  # 0-1 scale (0 = no traffic, 1 = heavy traffic)
    estimated_delay_minutes: float
    last_updated: datetime
    confidence_score: float

@dataclass
class CarbonFootprint:
    distance_km: float
    fuel_consumption_liters: float
    co2_emissions_kg: float
    vehicle_type: str
    load_factor: float  # 0-1 scale
    route_efficiency: float  # 0-1 scale

@dataclass
class RegulatoryCompliance:
    route_id: str
    compliance_score: float
    violations: List[str]
    required_permits: List[str]
    restrictions: Dict[str, Any]
    environmental_impact: Dict[str, float]

@dataclass
class OptimizedRoute:
    route_id: str
    vehicle_id: str
    stops: List[Dict[str, Any]]
    total_distance: float
    total_time: float
    total_cost: float
    carbon_footprint: CarbonFootprint
    traffic_conditions: List[TrafficCondition]
    compliance_info: RegulatoryCompliance
    risk_factors: Dict[str, float]

class EnhancedLogisticsOptimizationModule:
    def __init__(self):
        self.average_speed_normal = 60  # km/h
        self.average_speed_traffic = 30  # km/h
        self.vehicle_capacity = 1000  # units
        self.cost_per_km = 2.0  # $ per km
        self.fuel_efficiency = 0.12  # liters per km
        self.co2_per_liter = 2.68  # kg CO2 per liter of diesel
        self.max_vehicles = 10
        self.model_dir = 'models/enhanced_logistics'
        os.makedirs(self.model_dir, exist_ok=True)

    def fetch_real_time_traffic_data(self, locations: List[Dict[str, Any]]) -> List[TrafficCondition]:
        """Fetch real-time traffic data for route locations"""
        traffic_conditions = []

        for location in locations:
            # In a real implementation, this would connect to:
            # - Google Maps Traffic API
            # - HERE Traffic API
            # - TomTom Traffic API
            # - Waze Traffic API

            # Mock traffic data for demonstration
            traffic_index = np.random.beta(2, 5)  # 0-1 scale
            base_delay = traffic_index * 30  # Base delay in minutes

            # Add time-of-day effects
            hour = datetime.now().hour
            if 7 <= hour <= 9 or 16 <= hour <= 18:  # Rush hour
                base_delay *= 2.0
            elif 22 <= hour <= 6:  # Night time
                base_delay *= 0.3

            traffic_conditions.append(TrafficCondition(
                location=f"{location['latitude']:.4f},{location['longitude']:.4f}",
                traffic_index=traffic_index,
                estimated_delay_minutes=base_delay,
                last_updated=datetime.now(),
                confidence_score=np.random.beta(4, 2)  # 0-1 scale
            ))

        return traffic_conditions

    def calculate_carbon_footprint(self, distance_km: float, vehicle_type: str,
                                 load_factor: float, traffic_factor: float = 1.0) -> CarbonFootprint:
        """Calculate comprehensive carbon footprint for a route segment"""
        # Adjust fuel efficiency based on vehicle type and conditions
        base_fuel_efficiency = self.fuel_efficiency

        # Vehicle type adjustments
        vehicle_multipliers = {
            'truck_small': 0.8,
            'truck_medium': 1.0,
            'truck_large': 1.3,
            'electric': 0.0,  # Electric vehicles have zero tailpipe emissions
            'hybrid': 0.5
        }

        fuel_multiplier = vehicle_multipliers.get(vehicle_type, 1.0)
        adjusted_fuel_efficiency = base_fuel_efficiency * fuel_multiplier

        # Traffic impact on fuel efficiency
        traffic_fuel_penalty = 1 + (traffic_factor * 0.3)  # Up to 30% more fuel in traffic
        adjusted_fuel_efficiency *= traffic_fuel_penalty

        # Load factor impact
        load_efficiency = 1 + (load_factor * 0.2)  # Heavier loads are more efficient per unit
        adjusted_fuel_efficiency /= load_efficiency

        fuel_consumption = distance_km * adjusted_fuel_efficiency

        # Calculate emissions
        if vehicle_type == 'electric':
            # Electric vehicles have upstream emissions from electricity generation
            co2_emissions = fuel_consumption * 0.5  # kg CO2 equivalent
        else:
            co2_emissions = fuel_consumption * self.co2_per_liter

        # Calculate route efficiency (0-1 scale)
        optimal_fuel = distance_km * base_fuel_efficiency
        route_efficiency = min(1.0, optimal_fuel / (fuel_consumption + 0.001))

        return CarbonFootprint(
            distance_km=distance_km,
            fuel_consumption_liters=fuel_consumption,
            co2_emissions_kg=co2_emissions,
            vehicle_type=vehicle_type,
            load_factor=load_factor,
            route_efficiency=route_efficiency
        )

    def check_regulatory_compliance(self, route_stops: List[Dict[str, Any]],
                                  vehicle_type: str, country: str = 'US') -> RegulatoryCompliance:
        """Check regulatory compliance for route and vehicle"""
        violations = []
        required_permits = []
        restrictions = {}

        # Mock regulatory checks
        total_distance = sum(
            np.sqrt((route_stops[i]['latitude'] - route_stops[i+1]['latitude'])**2 +
                   (route_stops[i]['longitude'] - route_stops[i+1]['longitude'])**2) * 111  # Rough km conversion
            for i in range(len(route_stops)-1)
        )

        # Weight and dimension checks
        total_weight = sum(stop.get('demand', 0) for stop in route_stops)
        if total_weight > self.vehicle_capacity:
            violations.append(f"Weight limit exceeded: {total_weight} > {self.vehicle_capacity}")

        # Route-specific restrictions
        if total_distance > 500:  # Long haul restrictions
            required_permits.append("Long-haul transportation permit")
            restrictions['max_driving_hours'] = 11
            restrictions['required_rest_stops'] = max(1, int(total_distance / 400))

        # Environmental restrictions
        if country in ['US', 'EU']:
            required_permits.append("Environmental compliance certificate")
            restrictions['emission_standards'] = 'Euro 6' if country == 'EU' else 'EPA 2010'

        # Calculate compliance score
        base_compliance = 1.0
        compliance_penalties = {
            'weight_violation': 0.3,
            'permit_missing': 0.2,
            'environmental_violation': 0.4
        }

        for violation in violations:
            if 'weight' in violation.lower():
                base_compliance -= compliance_penalties['weight_violation']
            elif 'environmental' in violation.lower():
                base_compliance -= compliance_penalties['environmental_violation']

        for permit in required_permits:
            base_compliance -= compliance_penalties['permit_missing']

        compliance_score = max(0.0, base_compliance)

        # Environmental impact assessment
        environmental_impact = {
            'air_pollution_risk': 1 - compliance_score,
            'noise_pollution': 0.3 if total_distance > 200 else 0.1,
            'traffic_congestion': 0.2 if len(route_stops) > 10 else 0.1
        }

        return RegulatoryCompliance(
            route_id=f"route_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            compliance_score=compliance_score,
            violations=violations,
            required_permits=required_permits,
            restrictions=restrictions,
            environmental_impact=environmental_impact
        )

    def calculate_enhanced_distance_matrix(self, locations: List[Dict[str, Any]],
                                        traffic_conditions: List[TrafficCondition]) -> List[List[float]]:
        """Calculate enhanced distance matrix with traffic considerations"""
        n = len(locations)
        distance_matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i != j:
                    # Base Haversine distance
                    lat1, lon1 = locations[i]['latitude'], locations[i]['longitude']
                    lat2, lon2 = locations[j]['latitude'], locations[j]['longitude']

                    R = 6371  # Earth's radius in kilometers
                    dlat = np.radians(lat2 - lat1)
                    dlon = np.radians(lon2 - lon1)
                    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
                    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                    base_distance = R * c

                    # Apply traffic adjustments
                    if i < len(traffic_conditions):
                        traffic_factor = traffic_conditions[i].traffic_index
                        # Traffic can increase effective distance by up to 50%
                        distance_matrix[i][j] = base_distance * (1 + traffic_factor * 0.5)
                    else:
                        distance_matrix[i][j] = base_distance

        return distance_matrix

    def optimize_routes_with_constraints(self, warehouse_locations: List[Dict[str, Any]],
                                       customer_locations: List[Dict[str, Any]],
                                       vehicle_types: List[str] = None) -> Dict[str, Any]:
        """Optimize delivery routes with advanced constraints and real-time data"""

        if vehicle_types is None:
            vehicle_types = ['truck_medium'] * self.max_vehicles

        # Combine all locations
        all_locations = warehouse_locations + customer_locations
        num_locations = len(all_locations)

        # Fetch real-time traffic data
        traffic_conditions = self.fetch_real_time_traffic_data(all_locations)

        # Create enhanced distance matrix
        distance_matrix = self.calculate_enhanced_distance_matrix(all_locations, traffic_conditions)

        # Create the routing index manager
        manager = pywrapcp.RoutingIndexManager(num_locations, self.max_vehicles, 0)

        # Create routing model
        routing = pywrapcp.RoutingModel(manager)

        # Create distance callback with traffic considerations
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(distance_matrix[from_node][to_node] * 1000)  # Convert to meters

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add capacity constraints
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return all_locations[from_node].get('demand', 0)

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index, 0,  # null capacity slack
            [self.vehicle_capacity] * self.max_vehicles,  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity'
        )

        # Add time windows if available
        time_windows_available = all(loc.get('time_window_start') for loc in all_locations)
        if time_windows_available:
            def time_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                base_time = distance_matrix[from_node][to_node] / self.average_speed_normal * 60  # minutes

                # Add traffic delays
                if from_node < len(traffic_conditions):
                    traffic_delay = traffic_conditions[from_node].estimated_delay_minutes
                    base_time += traffic_delay

                return int(base_time * 60)  # Convert to seconds for OR-Tools

            time_callback_index = routing.RegisterTransitCallback(time_callback)
            routing.AddDimension(
                time_callback_index, 30,  # allow waiting time
                1440,  # maximum time per vehicle (24 hours)
                False,  # Don't force start cumul to zero
                'Time'
            )

            # Add time windows
            for i, location in enumerate(all_locations):
                if 'time_window_start' in location and 'time_window_end' in location:
                    start_time = self._time_to_minutes(location['time_window_start'])
                    end_time = self._time_to_minutes(location['time_window_end'])
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
            optimized_routes = []

            for vehicle_id in range(self.max_vehicles):
                if solution.HasVehicle(vehicle_id):
                    route = self._extract_enhanced_route_solution(
                        solution, routing, manager, vehicle_id, all_locations,
                        vehicle_types[vehicle_id], traffic_conditions
                    )
                    optimized_routes.append(route)

            # Calculate total metrics
            total_distance = sum(route.total_distance for route in optimized_routes)
            total_cost = sum(route.total_cost for route in optimized_routes)
            total_carbon = sum(route.carbon_footprint.co2_emissions_kg for route in optimized_routes)

            return {
                'success': True,
                'total_distance': total_distance,
                'total_cost': total_cost,
                'total_carbon_footprint': total_carbon,
                'routes': optimized_routes,
                'traffic_conditions': traffic_conditions,
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
                'total_cost': 0,
                'total_carbon_footprint': 0
            }

    def _extract_enhanced_route_solution(self, solution, routing, manager, vehicle_id: int,
                                       locations: List[Dict[str, Any]], vehicle_type: str,
                                       traffic_conditions: List[TrafficCondition]) -> OptimizedRoute:
        """Extract enhanced route solution with all metrics"""
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

        # Calculate total time with traffic
        total_time = total_distance / self.average_speed_normal * 60  # minutes
        traffic_delays = sum(tc.estimated_delay_minutes for tc in traffic_conditions[:len(route_stops)])
        total_time += traffic_delays

        # Calculate cost
        total_cost = total_distance * self.cost_per_km

        # Calculate carbon footprint
        load_factor = sum(stop.get('demand', 0) for stop in route_stops) / (self.vehicle_capacity * len(route_stops))
        load_factor = min(1.0, load_factor)

        avg_traffic_factor = np.mean([tc.traffic_index for tc in traffic_conditions[:len(route_stops)]])
        carbon_footprint = self.calculate_carbon_footprint(total_distance, vehicle_type, load_factor, avg_traffic_factor)

        # Check regulatory compliance
        compliance_info = self.check_regulatory_compliance(route_stops, vehicle_type)

        # Calculate risk factors
        risk_factors = {
            'traffic_risk': avg_traffic_factor,
            'compliance_risk': 1 - compliance_info.compliance_score,
            'distance_risk': min(1.0, total_distance / 500),  # Risk increases with distance
            'time_risk': min(1.0, total_time / 480)  # Risk increases with time (8 hours)
        }
        risk_factors['overall_risk'] = np.mean(list(risk_factors.values()))

        return OptimizedRoute(
            route_id=f"route_{vehicle_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            vehicle_id=f"{vehicle_type}_{vehicle_id + 1}",
            stops=route_stops,
            total_distance=total_distance,
            total_time=total_time,
            total_cost=total_cost,
            carbon_footprint=carbon_footprint,
            traffic_conditions=traffic_conditions[:len(route_stops)],
            compliance_info=compliance_info,
            risk_factors=risk_factors
        )

    def _get_unassigned_customers(self, solution, routing, manager,
                                customer_locations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

    def _get_warehouses(self) -> List[Dict[str, Any]]:
        """Get warehouse locations (placeholder)"""
        return []

    def _time_to_minutes(self, time_str: str) -> int:
        """Convert time string to minutes since midnight"""
        time_obj = datetime.strptime(time_str, '%H:%M').time()
        return time_obj.hour * 60 + time_obj.minute

    def generate_route_comparison_report(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive route comparison report"""
        if not optimization_results['success']:
            return {
                'summary': 'Route optimization failed',
                'total_routes': 0,
                'total_distance': 0,
                'total_cost': 0,
                'total_carbon_footprint': 0,
                'average_route_efficiency': 0
            }

        routes = optimization_results['routes']
        total_distance = optimization_results['total_distance']
        total_cost = optimization_results['total_cost']
        total_carbon = optimization_results['total_carbon_footprint']

        # Calculate efficiency metrics
        route_efficiencies = [route.carbon_footprint.route_efficiency for route in routes]
        compliance_scores = [route.compliance_info.compliance_score for route in routes]
        risk_scores = [route.risk_factors['overall_risk'] for route in routes]

        report = {
            'summary': f'Successfully optimized {len(routes)} routes with enhanced constraints',
            'total_routes': len(routes),
            'total_distance': total_distance,
            'total_cost': total_cost,
            'total_carbon_footprint': total_carbon,
            'average_route_efficiency': np.mean(route_efficiencies),
            'average_compliance_score': np.mean(compliance_scores),
            'average_risk_score': np.mean(risk_scores),
            'cost_per_km': total_cost / total_distance if total_distance > 0 else 0,
            'carbon_per_km': total_carbon / total_distance if total_distance > 0 else 0,
            'route_details': [
                {
                    'route_id': route.route_id,
                    'vehicle_id': route.vehicle_id,
                    'stops_count': len(route.stops),
                    'total_distance': route.total_distance,
                    'total_time': route.total_time,
                    'total_cost': route.total_cost,
                    'carbon_footprint': {
                        'co2_emissions': route.carbon_footprint.co2_emissions_kg,
                        'fuel_consumption': route.carbon_footprint.fuel_consumption_liters,
                        'route_efficiency': route.carbon_footprint.route_efficiency
                    },
                    'compliance_score': route.compliance_info.compliance_score,
                    'overall_risk': route.risk_factors['overall_risk'],
                    'stops': [
                        {
                            'location_id': stop.get('location_id', f'loc_{i}'),
                            'latitude': stop['latitude'],
                            'longitude': stop['longitude'],
                            'demand': stop.get('demand', 0)
                        } for i, stop in enumerate(route.stops)
                    ]
                } for route in routes
            ]
        }

        return report

    def simulate_traffic_scenarios(self, base_route: OptimizedRoute,
                                 scenarios: int = 100) -> Dict[str, Any]:
        """Simulate route performance under different traffic scenarios"""
        scenario_results = []

        for scenario in range(scenarios):
            # Generate random traffic conditions
            traffic_multipliers = np.random.beta(2, 5, len(base_route.stops))

            total_delay = sum(traffic_multipliers * 30)  # Up to 30 minutes delay per stop
            adjusted_time = base_route.total_time + total_delay
            adjusted_cost = base_route.total_cost * (1 + total_delay / base_route.total_time * 0.1)

            scenario_results.append({
                'scenario_id': scenario + 1,
                'total_delay_minutes': total_delay,
                'adjusted_total_time': adjusted_time,
                'adjusted_cost': adjusted_cost,
                'traffic_severity': np.mean(traffic_multipliers)
            })

        # Calculate statistics
        delays = [r['total_delay_minutes'] for r in scenario_results]
        costs = [r['adjusted_cost'] for r in scenario_results]

        return {
            'base_route_time': base_route.total_time,
            'base_route_cost': base_route.total_cost,
            'mean_delay': np.mean(delays),
            'mean_cost_increase': np.mean(costs) - base_route.total_cost,
            'delay_std': np.std(delays),
            'cost_std': np.std(costs),
            'worst_case_delay': np.max(delays),
            'worst_case_cost': np.max(costs),
            'best_case_delay': np.min(delays),
            'best_case_cost': np.min(costs),
            'delay_percentiles': {
                'p50': np.percentile(delays, 50),
                'p75': np.percentile(delays, 75),
                'p90': np.percentile(delays, 90),
                'p95': np.percentile(delays, 95)
            }
        }
