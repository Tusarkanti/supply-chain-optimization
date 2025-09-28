import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
from flask_mail import Mail, Message
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus

logger = logging.getLogger(__name__)

@dataclass
class SupplierRisk:
    supplier_id: str
    risk_score: float
    lead_time_variability: float
    quality_issues: float
    financial_stability: float
    geographic_risk: float

@dataclass
class SustainabilityMetrics:
    carbon_footprint: float
    waste_reduction: float
    energy_efficiency: float
    water_usage: float
    recycling_rate: float

@dataclass
class InventoryRecommendation:
    product_id: str
    warehouse_id: str
    current_stock: int
    recommended_order_quantity: int
    reorder_point: int
    safety_stock: int
    lead_time_days: int
    confidence_score: float
    reasoning: str
    supplier_recommendations: List[str]
    sustainability_impact: Dict[str, float]

class EnhancedInventoryManagementModule:
    def __init__(self):
        self.service_level_target = 0.95  # 95% service level
        self.lead_time_multiplier = 1.5  # Safety stock multiplier
        self.model_dir = 'models/enhanced_inventory'
        os.makedirs(self.model_dir, exist_ok=True)

    def analyze_inventory_levels(self, inventory_data: pd.DataFrame, sales_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current inventory levels and generate recommendations using enhanced optimization"""
        if inventory_data is None or sales_data is None:
            return {'recommendations': [], 'status': 'No data provided'}

        # Prepare demand data from sales_data (aggregate if needed)
        demand_data = sales_data.groupby(['product_id', 'warehouse_id'])['quantity'].sum().reset_index()
        demand_data.rename(columns={'quantity': 'demand_forecast'}, inplace=True)

        # Mock warehouses and products for optimization if not in data
        warehouses = [{'id': row['warehouse_id']} for _, row in inventory_data.iterrows() if 'warehouse_id' in row]
        products = [{'id': row['product_id']} for _, row in inventory_data.iterrows() if 'product_id' in row]
        demand_forecasts = {(row['warehouse_id'], row['product_id']): row.get('demand_forecast', 0) for _, row in demand_data.iterrows()}
        supplier_data = {row['product_id']: {'capacity': 1000} for _, row in inventory_data.iterrows() if 'product_id' in row}

        # Run optimization
        optimization_results = self.optimize_multi_echelon_inventory_pulp(warehouses, products, demand_forecasts, supplier_data)

        # Generate recommendations from results
        recommendations = []
        if optimization_results['status'] == 'optimal':
            for (wh_id, prod_id), inv_level in optimization_results['inventory_levels'].items():
                recommendations.append({
                    'product_id': prod_id,
                    'warehouse_id': wh_id,
                    'recommended_level': int(inv_level),
                    'current_stock': inventory_data[(inventory_data['warehouse_id'] == wh_id) & (inventory_data['product_id'] == prod_id)]['current_stock'].iloc[0] if not inventory_data[(inventory_data['warehouse_id'] == wh_id) & (inventory_data['product_id'] == prod_id)].empty else 0,
                    'reasoning': f"Optimized level to meet demand while minimizing costs."
                })

        return {
            'recommendations': recommendations,
            'total_analyzed': len(recommendations),
            'optimization_status': optimization_results['status'],
            'service_level': optimization_results.get('service_level', 0)
        }

    def calculate_demand_uncertainty(self, demand_history: List[float]) -> Dict[str, float]:
        """Calculate various measures of demand uncertainty"""
        if len(demand_history) < 2:
            return {'mean': 0, 'std': 0, 'cv': 0, 'mad': 0}

        demand_array = np.array(demand_history)

        # Coefficient of variation
        cv = np.std(demand_array) / np.mean(demand_array) if np.mean(demand_array) > 0 else 0

        # Mean absolute deviation
        mad = np.mean(np.abs(demand_array - np.mean(demand_array)))

        # Skewness and kurtosis
        skewness = stats.skew(demand_array)
        kurtosis = stats.kurtosis(demand_array)

        return {
            'mean': np.mean(demand_array),
            'std': np.std(demand_array),
            'cv': cv,
            'mad': mad,
            'skewness': skewness,
            'kurtosis': kurtosis
        }

    def calculate_advanced_safety_stock(self, demand_uncertainty: Dict[str, float],
                                     lead_time_days: int, service_level: float = 0.95) -> int:
        """Calculate safety stock using advanced statistical methods"""
        # Get Z-score for service level
        z_score = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}[service_level]

        # Base safety stock calculation
        base_safety_stock = z_score * demand_uncertainty['std'] * np.sqrt(lead_time_days)

        # Adjust for demand variability patterns
        if demand_uncertainty['skewness'] > 1.0:  # Highly skewed demand
            base_safety_stock *= 1.2
        elif demand_uncertainty['kurtosis'] > 3.0:  # Heavy tails
            base_safety_stock *= 1.1

        # Adjust for coefficient of variation
        if demand_uncertainty['cv'] > 0.5:  # High variability
            base_safety_stock *= (1 + demand_uncertainty['cv'])

        return max(0, int(base_safety_stock))

    def assess_supplier_risk(self, supplier_data: Dict[str, Any]) -> SupplierRisk:
        """Assess supplier risk using real KPI calculations from DB"""
        from database_schema import Supplier, db

        supplier_id = supplier_data.get('supplier_id')
        if not supplier_id:
            # Fallback to mock data if no supplier_id
            risk_factors = {
                'lead_time_variability': np.random.beta(2, 5),
                'quality_issues': np.random.beta(3, 7),
                'financial_stability': np.random.beta(4, 3),
                'geographic_risk': np.random.beta(2, 8)
            }
            overall_risk = np.mean(list(risk_factors.values()))
            return SupplierRisk(
                supplier_id='unknown',
                risk_score=overall_risk,
                lead_time_variability=risk_factors['lead_time_variability'],
                quality_issues=risk_factors['quality_issues'],
                financial_stability=risk_factors['financial_stability'],
                geographic_risk=risk_factors['geographic_risk']
            )

        # Get supplier from database
        supplier = Supplier.query.get(supplier_id)
        if not supplier:
            # Supplier not found, use mock data
            risk_factors = {
                'lead_time_variability': np.random.beta(2, 5),
                'quality_issues': np.random.beta(3, 7),
                'financial_stability': np.random.beta(4, 3),
                'geographic_risk': np.random.beta(2, 8)
            }
            overall_risk = np.mean(list(risk_factors.values()))
            return SupplierRisk(
                supplier_id=str(supplier_id),
                risk_score=overall_risk,
                lead_time_variability=risk_factors['lead_time_variability'],
                quality_issues=risk_factors['quality_issues'],
                financial_stability=risk_factors['financial_stability'],
                geographic_risk=risk_factors['geographic_risk']
            )

        # Calculate risk factors from real KPIs
        # Lead time variability (lower is better, so invert the score)
        lead_time_risk = 1 - min(1.0, supplier.lead_time_variance / 30)  # Normalize to 0-1, assuming 30 days max variance

        # Quality issues (lower rating = higher risk)
        quality_risk = 1 - (supplier.quality_rating / 5.0) if supplier.quality_rating else 0.5

        # Order accuracy (lower accuracy = higher risk)
        accuracy_risk = 1 - supplier.order_accuracy_rate if supplier.order_accuracy_rate else 0.5

        # Geographic risk based on location (simplified)
        geographic_risk = 0.3  # Default medium risk
        if supplier.country:
            high_risk_countries = ['North Korea', 'Iran', 'Venezuela']  # Example high-risk countries
            if supplier.country in high_risk_countries:
                geographic_risk = 0.9
            elif supplier.country in ['USA', 'Canada', 'Germany', 'Japan']:
                geographic_risk = 0.1

        # Overall risk score (weighted average)
        risk_factors = {
            'lead_time_variability': lead_time_risk,
            'quality_issues': quality_risk,
            'financial_stability': accuracy_risk,  # Using accuracy as proxy for financial stability
            'geographic_risk': geographic_risk
        }

        overall_risk = np.average(list(risk_factors.values()), weights=[0.3, 0.3, 0.2, 0.2])

        return SupplierRisk(
            supplier_id=str(supplier_id),
            risk_score=overall_risk,
            lead_time_variability=lead_time_risk,
            quality_issues=quality_risk,
            financial_stability=accuracy_risk,
            geographic_risk=geographic_risk
        )

    def calculate_sustainability_metrics(self, product_data: Dict[str, Any]) -> SustainabilityMetrics:
        """Calculate sustainability metrics for inventory decisions"""
        # Mock sustainability calculations
        return SustainabilityMetrics(
            carbon_footprint=np.random.normal(2.5, 0.5),  # kg CO2 per unit
            waste_reduction=np.random.beta(4, 3),  # 0-1 scale
            energy_efficiency=np.random.beta(3, 4),  # 0-1 scale
            water_usage=np.random.normal(15.0, 3.0),  # liters per unit
            recycling_rate=np.random.beta(3, 2)  # 0-1 scale
        )

    def optimize_multi_echelon_inventory(self, inventory_data: pd.DataFrame,
                                       demand_data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize inventory across multiple echelons (warehouses, distribution centers)"""
        # This would implement sophisticated multi-echelon optimization
        # For now, provide enhanced single-echelon optimization

        recommendations = []

        # Merge inventory and demand data
        merged_data = inventory_data.merge(demand_data, on=['product_id', 'warehouse_id'], how='left')

        # Group by product and warehouse
        for (product_id, warehouse_id), group in merged_data.groupby(['product_id', 'warehouse_id']):
            current_stock = group['current_stock'].iloc[0]
            reserved_stock = group['reserved_stock'].iloc[0]
            available_stock = current_stock - reserved_stock

            # Calculate demand uncertainty
            sales_group = group.dropna(subset=['quantity'])
            if len(sales_group) > 0:
                demand_history = sales_group['quantity'].tolist()
                demand_uncertainty = self.calculate_demand_uncertainty(demand_history)
                avg_daily_demand = demand_uncertainty['mean']
                lead_time_days = group['lead_time_days'].iloc[0] if 'lead_time_days' in group.columns else 7
            else:
                demand_uncertainty = {'mean': 0, 'std': 0, 'cv': 0, 'mad': 0}
                avg_daily_demand = 0
                lead_time_days = 7

            # Calculate advanced safety stock
            safety_stock = self.calculate_advanced_safety_stock(
                demand_uncertainty, lead_time_days, self.service_level_target
            )

            # Calculate reorder point
            reorder_point = (avg_daily_demand * lead_time_days) + safety_stock

            # Determine if reorder is needed
            if available_stock <= reorder_point:
                # Calculate order quantity using advanced methods
                order_quantity = self._calculate_optimal_order_quantity(
                    avg_daily_demand, lead_time_days, safety_stock, demand_uncertainty
                )

                # Calculate confidence score based on data quality
                confidence_score = self._calculate_confidence_score(
                    len(demand_history), demand_uncertainty['cv'], lead_time_days
                )

                # Assess supplier risks
                supplier_risks = []
                for supplier_id in group['supplier_id'].unique():
                    supplier_data = {'supplier_id': supplier_id}
                    supplier_risk = self.assess_supplier_risk(supplier_data)
                    supplier_risks.append({
                        'supplier_id': supplier_id,
                        'risk_score': supplier_risk.risk_score,
                        'recommendation': self._get_supplier_recommendation(supplier_risk)
                    })

                # Calculate sustainability impact
                sustainability = self.calculate_sustainability_metrics({'product_id': product_id})

                # Generate reasoning
                reasoning = self._generate_enhanced_recommendation_reasoning(
                    available_stock, reorder_point, safety_stock, avg_daily_demand,
                    lead_time_days, demand_uncertainty, supplier_risks
                )

                recommendation = InventoryRecommendation(
                    product_id=product_id,
                    warehouse_id=warehouse_id,
                    current_stock=current_stock,
                    recommended_order_quantity=int(order_quantity),
                    reorder_point=reorder_point,
                    safety_stock=safety_stock,
                    lead_time_days=lead_time_days,
                    confidence_score=confidence_score,
                    reasoning=reasoning,
                    supplier_recommendations=[s['recommendation'] for s in supplier_risks],
                    sustainability_impact=vars(sustainability)
                )

                recommendations.append(recommendation)

        return {
            'total_products_analyzed': len(merged_data['product_id'].unique()),
            'total_warehouses_analyzed': len(merged_data['warehouse_id'].unique()),
            'recommendations_count': len(recommendations),
            'recommendations': recommendations
        }

    def _calculate_optimal_order_quantity(self, avg_daily_demand: float, lead_time_days: int,
                                       safety_stock: int, demand_uncertainty: Dict[str, float]) -> float:
        """Calculate optimal order quantity using advanced methods"""
        # Base EOQ calculation
        annual_demand = avg_daily_demand * 365
        ordering_cost = 50  # Assume $50 per order
        holding_cost = 10   # Assume $10 per unit per year

        if annual_demand == 0 or ordering_cost == 0 or holding_cost == 0:
            return safety_stock * 2

        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)

        # Adjust for demand uncertainty
        if demand_uncertainty['cv'] > 0.3:
            eoq *= (1 + demand_uncertainty['cv'] * 0.5)

        # Adjust for lead time variability
        eoq *= (1 + lead_time_days / 30)  # Longer lead times require larger orders

        # Ensure minimum order quantity
        min_order_quantity = max(safety_stock * 2, avg_daily_demand * lead_time_days)

        return max(min_order_quantity, eoq)

    def _calculate_confidence_score(self, data_points: int, cv: float, lead_time_days: int) -> float:
        """Calculate confidence score based on data quality and uncertainty"""
        # Base confidence from data availability
        data_confidence = min(1.0, data_points / 100)  # More data = higher confidence

        # Adjust for demand variability
        variability_penalty = max(0.5, 1 - cv)  # High variability reduces confidence

        # Adjust for lead time
        lead_time_confidence = max(0.7, 1 - (lead_time_days / 60))  # Longer lead times reduce confidence

        return data_confidence * variability_penalty * lead_time_confidence

    def _get_supplier_recommendation(self, supplier_risk: SupplierRisk) -> str:
        """Generate supplier-specific recommendations"""
        if supplier_risk.risk_score > 0.7:
            return f"High risk supplier ({supplier_risk.supplier_id}) - consider dual sourcing"
        elif supplier_risk.risk_score > 0.5:
            return f"Medium risk supplier ({supplier_risk.supplier_id}) - monitor closely"
        else:
            return f"Low risk supplier ({supplier_risk.supplier_id}) - preferred supplier"

    def _generate_enhanced_recommendation_reasoning(self, available_stock: int, reorder_point: int,
                                                 safety_stock: int, avg_daily_demand: float,
                                                 lead_time_days: int, demand_uncertainty: Dict[str, float],
                                                 supplier_risks: List[Dict]) -> str:
        """Generate comprehensive reasoning for inventory recommendation"""
        if available_stock <= 0:
            return "Critical: Out of stock. Immediate reorder required to prevent stockouts."

        if available_stock <= reorder_point:
            days_until_stockout = available_stock / max(avg_daily_demand, 1)
            if days_until_stockout <= lead_time_days:
                urgency = "Critical"
            elif days_until_stockout <= lead_time_days * 1.5:
                urgency = "High"
            else:
                urgency = "Medium"

            reasoning = (f"{urgency} priority: Available stock ({available_stock}) is below reorder point ({reorder_point}). "
                        f"Safety stock: {safety_stock}, Average daily demand: {avg_daily_demand:.1f}, "
                        f"Lead time: {lead_time_days} days. Expected stockout in {days_until_stockout:.1f} days.")

            # Add demand uncertainty information
            if demand_uncertainty['cv'] > 0.3:
                reasoning += f" High demand variability (CV: {demand_uncertainty['cv']:.2f}) detected."

            # Add supplier risk information
            high_risk_suppliers = [s for s in supplier_risks if s['risk_score'] > 0.7]
            if high_risk_suppliers:
                reasoning += f" {len(high_risk_suppliers)} high-risk suppliers identified."

            return reasoning

        return f"Stock level ({available_stock}) is above reorder point ({reorder_point}). No immediate action required."

    def generate_inventory_report(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive inventory report with advanced analytics"""
        recommendations = analysis_results['recommendations']

        if not recommendations:
            return {
                'summary': 'No inventory recommendations generated',
                'total_recommendations': 0,
                'critical_issues': 0,
                'high_priority': 0,
                'medium_priority': 0
            }

        # Categorize recommendations
        critical = [r for r in recommendations if 'Critical' in r.reasoning]
        high_priority = [r for r in recommendations if 'High' in r.reasoning and 'Critical' not in r.reasoning]
        medium_priority = [r for r in recommendations if 'Medium' in r.reasoning]

        # Calculate total recommended order value
        total_order_value = sum(r.recommended_order_quantity * 10 for r in recommendations)  # Assume $10 per unit

        # Calculate sustainability impact
        total_carbon_footprint = sum(r.sustainability_impact.get('carbon_footprint', 0) * r.recommended_order_quantity for r in recommendations)

        report = {
            'summary': f'Generated {len(recommendations)} inventory recommendations',
            'total_recommendations': len(recommendations),
            'critical_issues': len(critical),
            'high_priority': len(high_priority),
            'medium_priority': len(medium_priority),
            'total_recommended_order_value': total_order_value,
            'total_carbon_footprint': total_carbon_footprint,
            'average_confidence_score': np.mean([r.confidence_score for r in recommendations]),
            'priority_breakdown': {
                'critical': len(critical),
                'high': len(high_priority),
                'medium': len(medium_priority)
            },
            'sustainability_summary': {
                'total_carbon_footprint': total_carbon_footprint,
                'average_carbon_per_unit': total_carbon_footprint / sum(r.recommended_order_quantity for r in recommendations) if recommendations else 0
            },
            'detailed_recommendations': [
                {
                    'product_id': r.product_id,
                    'warehouse_id': r.warehouse_id,
                    'current_stock': r.current_stock,
                    'recommended_order_quantity': r.recommended_order_quantity,
                    'reorder_point': r.reorder_point,
                    'safety_stock': r.safety_stock,
                    'confidence_score': r.confidence_score,
                    'reasoning': r.reasoning,
                    'supplier_recommendations': r.supplier_recommendations,
                    'sustainability_impact': r.sustainability_impact,
                    'priority': 'Critical' if 'Critical' in r.reasoning else ('High' if 'High' in r.reasoning else 'Medium')
                }
                for r in recommendations
            ]
        }

        return report

    def simulate_inventory_scenarios(self, current_stock: int, daily_demand: List[float],
                                   lead_time: int, reorder_point: int, order_quantity: int,
                                   scenarios: int = 1000) -> Dict[str, Any]:
        """Simulate inventory behavior under different scenarios using Monte Carlo"""
        stock_levels = []
        stockouts = []
        orders_placed = []

        for scenario in range(scenarios):
            stock_level = current_stock
            scenario_stockouts = 0
            scenario_orders = 0

            for demand in daily_demand:
                # Update stock level
                stock_level -= demand

                # Check for reorder
                if stock_level <= reorder_point and (len(stock_levels) == 0 or stock_levels[-1] > reorder_point):
                    stock_level += order_quantity
                    scenario_orders += 1

                # Count stockouts
                if stock_level < 0:
                    scenario_stockouts += 1

                stock_levels.append(max(0, stock_level))

            stockouts.append(scenario_stockouts)
            orders_placed.append(scenario_orders)

        simulation_results = {
            'mean_final_stock_level': np.mean(stock_levels),
            'mean_stockouts': np.mean(stockouts),
            'mean_orders_placed': np.mean(orders_placed),
            'stockout_probability': np.mean(np.array(stockouts) > 0),
            'service_level_achieved': 1 - np.mean(stockouts) / len(daily_demand),
            'confidence_intervals': {
                'stockouts': {
                    'lower': np.percentile(stockouts, 2.5),
                    'upper': np.percentile(stockouts, 97.5)
                },
                'orders': {
                    'lower': np.percentile(orders_placed, 2.5),
                    'upper': np.percentile(orders_placed, 97.5)
                }
            }
        }

        return simulation_results

    def optimize_multi_echelon_inventory_pulp(self, warehouses: List[Dict], products: List[Dict],
                                            demand_forecasts: Dict, supplier_data: Dict) -> Dict[str, Any]:
        """Implement multi-echelon inventory optimization using linear programming (PuLP)"""
        try:
            # Create optimization problem
            prob = LpProblem("Multi_Echelon_Inventory_Optimization", LpMinimize)

            # Decision variables
            # X[i][j][k] = quantity of product k shipped from warehouse i to warehouse j
            # I[i][k] = inventory level of product k at warehouse i
            # O[k] = total order quantity for product k from suppliers

            warehouse_ids = [w['id'] for w in warehouses]
            product_ids = [p['id'] for p in products]

            # Decision variables for inter-warehouse shipments
            X = {}
            for i in warehouse_ids:
                for j in warehouse_ids:
                    if i != j:
                        for k in product_ids:
                            X[(i, j, k)] = LpVariable(f"X_{i}_{j}_{k}", lowBound=0, cat='Continuous')

            # Decision variables for inventory levels
            I = {}
            for i in warehouse_ids:
                for k in product_ids:
                    I[(i, k)] = LpVariable(f"I_{i}_{k}", lowBound=0, cat='Continuous')

            # Decision variables for supplier orders
            O = {}
            for k in product_ids:
                O[k] = LpVariable(f"O_{k}", lowBound=0, cat='Continuous')

            # Objective function: minimize total cost
            # Costs include: holding costs, transportation costs, ordering costs, stockout costs
            holding_cost = 0
            transport_cost = 0
            ordering_cost = 0
            stockout_cost = 0

            # Holding cost
            for i in warehouse_ids:
                for k in product_ids:
                    holding_cost += 0.1 * I[(i, k)]  # $0.10 per unit per period

            # Transportation cost (simplified)
            for i in warehouse_ids:
                for j in warehouse_ids:
                    if i != j:
                        for k in product_ids:
                            distance = self._calculate_distance(warehouses, i, j)
                            transport_cost += 0.05 * distance * X[(i, j, k)]  # $0.05 per km per unit

            # Ordering cost
            for k in product_ids:
                ordering_cost += 50 * (O[k] > 0)  # Fixed ordering cost

            # Stockout cost (penalty for unmet demand)
            for i in warehouse_ids:
                for k in product_ids:
                    forecast = demand_forecasts.get((i, k), 0)
                    stockout_cost += 10 * max(0, forecast - I[(i, k)])  # $10 penalty per unit short

            prob += holding_cost + transport_cost + ordering_cost + stockout_cost

            # Constraints

            # 1. Inventory balance constraints
            for i in warehouse_ids:
                for k in product_ids:
                    inflow = O[k] if i == warehouse_ids[0] else 0  # Assume first warehouse receives from suppliers
                    outflow = lpSum([X[(i, j, k)] for j in warehouse_ids if j != i])
                    inflow += lpSum([X[(j, i, k)] for j in warehouse_ids if j != i])

                    prob += I[(i, k)] == inflow - outflow

            # 2. Demand satisfaction constraints (soft constraints via penalty in objective)
            for i in warehouse_ids:
                for k in product_ids:
                    forecast = demand_forecasts.get((i, k), 0)
                    prob += I[(i, k)] >= 0.8 * forecast  # At least 80% of forecast must be satisfied

            # 3. Capacity constraints
            for i in warehouse_ids:
                warehouse = next(w for w in warehouses if w['id'] == i)
                capacity = warehouse.get('capacity', 10000)
                prob += lpSum([I[(i, k)] for k in product_ids]) <= capacity

            # 4. Supplier capacity constraints
            for k in product_ids:
                supplier_capacity = supplier_data.get(k, {}).get('capacity', 1000)
                prob += O[k] <= supplier_capacity

            # Solve the problem
            status = prob.solve()

            if status == LpStatusOptimal:
                # Extract solution
                inventory_levels = {(i, k): I[(i, k)].value() for i in warehouse_ids for k in product_ids}
                total_inventory = sum(inventory_levels.values())
                avg_inventory = total_inventory / len(inventory_levels) if inventory_levels else 0

                # Calculate service level (percentage of demand satisfied)
                total_forecast = sum(sum(demand_forecasts.get(prod_id, {}).values()) for prod_id in product_ids)
                satisfied_demand = sum(min(forecast, inv_level) for (wh, prod), inv_level in inventory_levels.items() 
                                     for forecast in [demand_forecasts.get(prod, {}).get(wh, 0)])
                service_level = (satisfied_demand / total_forecast) if total_forecast > 0 else 0.95

                # Calculate inventory turnover (simplified: total demand / avg inventory)
                total_demand = sum(sum(forecasts.values()) for forecasts in demand_forecasts.values())
                inventory_turnover = total_demand / avg_inventory if avg_inventory > 0 else 1.0

                # Generate recommendations
                recommendations = [
                    f"Maintain inventory level for {prod} at {inv:.0f} units in {wh} to meet demand.",
                    f"Total cost minimized to ${prob.objective.value():.2f}. Review supplier orders: {sum(O[k].value() for k in product_ids):.0f} units.",
                    f"Achieve {service_level*100:.1f}% service level across {len(warehouse_ids)} warehouses."
                ]

                solution = {
                    'status': 'optimal',
                    'total_cost': prob.objective.value(),
                    'service_level': service_level,
                    'inventory_turnover': inventory_turnover,
                    'recommendations': recommendations,
                    'inventory_levels': inventory_levels,
                    'shipments': {(i, j, k): X[(i, j, k)].value() for i in warehouse_ids for j in warehouse_ids if i != j for k in product_ids},
                    'supplier_orders': {k: O[k].value() for k in product_ids}
                }
            else:
                solution = {
                    'status': 'infeasible',
                    'message': 'No optimal solution found',
                    'total_cost': 0,
                    'service_level': 0,
                    'inventory_turnover': 0,
                    'recommendations': ['Optimization failed. Check input data and constraints.']
                }

            return solution

        except Exception as e:
            logger.error(f"Multi-echelon optimization failed: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    def _calculate_distance(self, warehouses: List[Dict], warehouse_id1: str, warehouse_id2: str) -> float:
        """Calculate distance between two warehouses"""
        w1 = next(w for w in warehouses if w['id'] == warehouse_id1)
        w2 = next(w for w in warehouses if w['id'] == warehouse_id2)

        # Simple Euclidean distance calculation
        lat1, lon1 = w1.get('latitude', 0), w1.get('longitude', 0)
        lat2, lon2 = w2.get('latitude', 0), w2.get('longitude', 0)

        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of earth in kilometers

        return c * r

    def send_reorder_alert(self, recommendation: InventoryRecommendation, recipient_emails: List[str]) -> bool:
        """Send reorder alert via email using Flask-Mail"""
        try:
            from flask import current_app

            mail = Mail(current_app)

            subject = f"Inventory Reorder Alert - Product {recommendation.product_id}"

            body = f"""
            Inventory Reorder Alert

            Product ID: {recommendation.product_id}
            Warehouse ID: {recommendation.warehouse_id}
            Current Stock: {recommendation.current_stock}
            Recommended Order Quantity: {recommendation.recommended_order_quantity}
            Reorder Point: {recommendation.reorder_point}
            Safety Stock: {recommendation.safety_stock}

            Reasoning: {recommendation.reasoning}

            Supplier Recommendations:
            {chr(10).join(f"- {rec}" for rec in recommendation.supplier_recommendations)}

            Confidence Score: {recommendation.confidence_score:.2f}

            Please review and place the order as soon as possible.

            Generated at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
            """

            msg = Message(
                subject=subject,
                recipients=recipient_emails,
                body=body
            )

            mail.send(msg)
            logger.info(f"Reorder alert sent for product {recommendation.product_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to send reorder alert: {str(e)}")
            return False
