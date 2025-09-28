import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

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

class InventoryManagementModule:
    def __init__(self):
        self.service_level_target = 0.95  # 95% service level
        self.lead_time_multiplier = 1.5  # Safety stock multiplier

    def calculate_safety_stock(self, demand_std: float, lead_time_days: int, service_level: float = 0.95) -> int:
        """Calculate safety stock using statistical method"""
        # Z-score for service level (95% = 1.645)
        z_score = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}[service_level]

        # Safety stock = Z-score * demand_std * sqrt(lead_time)
        safety_stock = z_score * demand_std * np.sqrt(lead_time_days)

        return max(0, int(safety_stock))

    def calculate_reorder_point(self, average_daily_demand: float, lead_time_days: int, safety_stock: int) -> int:
        """Calculate reorder point"""
        reorder_point = (average_daily_demand * lead_time_days) + safety_stock
        return max(0, int(reorder_point))

    def calculate_economic_order_quantity(self, annual_demand: float, ordering_cost: float, holding_cost: float) -> int:
        """Calculate EOQ using the classic formula"""
        if annual_demand == 0 or ordering_cost == 0 or holding_cost == 0:
            return 0

        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        return max(1, int(eoq))

    def analyze_inventory_levels(self, inventory_data: pd.DataFrame, sales_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current inventory levels and generate recommendations"""
        recommendations = []

        # Merge inventory and sales data
        merged_data = inventory_data.merge(sales_data, on=['product_id', 'warehouse_id'], how='left')

        # Group by product and warehouse
        for (product_id, warehouse_id), group in merged_data.groupby(['product_id', 'warehouse_id']):
            current_stock = group['current_stock'].iloc[0]
            reserved_stock = group['reserved_stock'].iloc[0]
            available_stock = current_stock - reserved_stock

            # Calculate demand statistics
            sales_group = group.dropna(subset=['quantity'])
            if len(sales_group) > 0:
                avg_daily_demand = sales_group['quantity'].mean()
                demand_std = sales_group['quantity'].std()
                lead_time_days = group['lead_time_days'].iloc[0] if 'lead_time_days' in group.columns else 7
            else:
                avg_daily_demand = 0
                demand_std = 0
                lead_time_days = 7

            # Calculate safety stock and reorder point
            safety_stock = self.calculate_safety_stock(demand_std, lead_time_days, self.service_level_target)
            reorder_point = self.calculate_reorder_point(avg_daily_demand, lead_time_days, safety_stock)

            # Determine if reorder is needed
            if available_stock <= reorder_point:
                # Calculate order quantity
                annual_demand = avg_daily_demand * 365
                ordering_cost = 50  # Assume $50 per order
                holding_cost = 10   # Assume $10 per unit per year

                eoq = self.calculate_economic_order_quantity(annual_demand, ordering_cost, holding_cost)

                # Adjust EOQ based on current situation
                if eoq == 0:
                    recommended_quantity = reorder_point + safety_stock - available_stock
                else:
                    recommended_quantity = max(eoq, reorder_point + safety_stock - available_stock)

                # Calculate confidence score
                confidence_score = min(0.95, max(0.5, 1 - (demand_std / (avg_daily_demand + 1))))

                # Generate reasoning
                reasoning = self._generate_recommendation_reasoning(
                    available_stock, reorder_point, safety_stock, avg_daily_demand, lead_time_days
                )

                recommendation = InventoryRecommendation(
                    product_id=product_id,
                    warehouse_id=warehouse_id,
                    current_stock=current_stock,
                    recommended_order_quantity=int(recommended_quantity),
                    reorder_point=reorder_point,
                    safety_stock=safety_stock,
                    lead_time_days=lead_time_days,
                    confidence_score=confidence_score,
                    reasoning=reasoning
                )

                recommendations.append(recommendation)

        return {
            'total_products_analyzed': len(merged_data['product_id'].unique()),
            'total_warehouses_analyzed': len(merged_data['warehouse_id'].unique()),
            'recommendations_count': len(recommendations),
            'recommendations': recommendations
        }

    def _generate_recommendation_reasoning(self, available_stock: int, reorder_point: int,
                                         safety_stock: int, avg_daily_demand: float,
                                         lead_time_days: int) -> str:
        """Generate human-readable reasoning for inventory recommendation"""
        if available_stock <= 0:
            return f"Critical: Out of stock. Immediate reorder required to prevent stockouts."

        if available_stock <= reorder_point:
            days_until_stockout = available_stock / max(avg_daily_demand, 1)
            if days_until_stockout <= lead_time_days:
                urgency = "Critical"
            elif days_until_stockout <= lead_time_days * 1.5:
                urgency = "High"
            else:
                urgency = "Medium"

            return (f"{urgency} priority: Available stock ({available_stock}) is below reorder point ({reorder_point}). "
                   f"Expected stockout in {days_until_stockout:.1f} days based on average daily demand of {avg_daily_demand:.1f} units.")

        return f"Stock level ({available_stock}) is above reorder point ({reorder_point}). No immediate action required."

    def optimize_inventory_policy(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize inventory policy parameters for a product"""
        # This would use more sophisticated optimization techniques
        # For now, provide basic optimization

        optimization_results = {
            'recommended_service_level': 0.95,
            'recommended_safety_stock_multiplier': 1.5,
            'cost_analysis': {
                'holding_cost_impact': 'Medium',
                'stockout_cost_impact': 'High',
                'recommended_policy': 'Balanced approach with 95% service level'
            }
        }

        return optimization_results

    def generate_inventory_report(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive inventory report"""
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

        report = {
            'summary': f'Generated {len(recommendations)} inventory recommendations',
            'total_recommendations': len(recommendations),
            'critical_issues': len(critical),
            'high_priority': len(high_priority),
            'medium_priority': len(medium_priority),
            'total_recommended_order_value': total_order_value,
            'average_confidence_score': np.mean([r.confidence_score for r in recommendations]),
            'priority_breakdown': {
                'critical': len(critical),
                'high': len(high_priority),
                'medium': len(medium_priority)
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
                    'priority': 'Critical' if 'Critical' in r.reasoning else ('High' if 'High' in r.reasoning else 'Medium')
                }
                for r in recommendations
            ]
        }

        return report

    def simulate_inventory_scenario(self, current_stock: int, daily_demand: List[float],
                                  lead_time: int, reorder_point: int, order_quantity: int) -> Dict[str, Any]:
        """Simulate inventory behavior under different scenarios"""
        # Simple simulation for demonstration
        stock_levels = [current_stock]
        stockouts = 0
        orders_placed = 0

        for i, demand in enumerate(daily_demand):
            # Update stock level
            new_stock = stock_levels[-1] - demand

            # Check for reorder
            if new_stock <= reorder_point and (i == 0 or stock_levels[-2] > reorder_point):
                new_stock += order_quantity
                orders_placed += 1

            stock_levels.append(max(0, new_stock))

            # Count stockouts
            if new_stock < 0:
                stockouts += 1

        simulation_results = {
            'final_stock_level': stock_levels[-1],
            'total_stockouts': stockouts,
            'orders_placed': orders_placed,
            'average_stock_level': np.mean(stock_levels),
            'min_stock_level': min(stock_levels),
            'max_stock_level': max(stock_levels)
        }

        return simulation_results

    def optimize_multi_echelon_inventory_pulp(self, warehouses: List[Dict], products: List[Dict], 
                                            demand_forecasts: Dict, supplier_data: Dict) -> Dict[str, Any]:
        """Optimize multi-echelon inventory using PuLP"""
        try:
            from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, value
        except ImportError:
            return {
                'success': False,
                'error': 'PuLP library not available. Install with: pip install pulp'
            }

        # Create optimization problem
        prob = LpProblem("Multi_Echelon_Inventory_Optimization", LpMinimize)

        # Decision variables
        # Inventory levels at each warehouse for each product
        inventory_vars = {}
        for warehouse in warehouses:
            for product in products:
                var_name = f"inv_{warehouse['id']}_{product['id']}"
                inventory_vars[(warehouse['id'], product['id'])] = LpVariable(var_name, 0, None)

        # Transport variables between warehouses
        transport_vars = {}
        for i, wh1 in enumerate(warehouses):
            for j, wh2 in enumerate(warehouses):
                if i != j:
                    for product in products:
                        var_name = f"trans_{wh1['id']}_{wh2['id']}_{product['id']}"
                        transport_vars[(wh1['id'], wh2['id'], product['id'])] = LpVariable(var_name, 0, None)

        # Objective function: minimize total inventory holding costs
        holding_costs = []
        for warehouse in warehouses:
            for product in products:
                holding_cost = product.get('holding_cost', 1.0)  # Default holding cost
                holding_costs.append(holding_cost * inventory_vars[(warehouse['id'], product['id'])])

        prob += lpSum(holding_costs)

        # Constraints
        for warehouse in warehouses:
            for product in products:
                wh_id = warehouse['id']
                prod_id = product['id']
                
                # Demand satisfaction constraint
                demand = demand_forecasts.get(prod_id, {}).get(wh_id, 0)
                
                # Inventory balance
                inflows = [transport_vars.get((other_wh['id'], wh_id, prod_id), 0) for other_wh in warehouses if other_wh['id'] != wh_id]
                outflows = [transport_vars.get((wh_id, other_wh['id'], prod_id), 0) for other_wh in warehouses if other_wh['id'] != wh_id]
                
                prob += inventory_vars[(wh_id, prod_id)] + lpSum(inflows) - lpSum(outflows) >= demand

        # Solve the problem
        status = prob.solve()

        if LpStatus[status] == 'Optimal':
            # Extract results
            optimal_inventory = {}
            for warehouse in warehouses:
                optimal_inventory[warehouse['id']] = {}
                for product in products:
                    optimal_inventory[warehouse['id']][product['id']] = value(inventory_vars[(warehouse['id'], product['id'])])

            total_cost = value(prob.objective)

            return {
                'success': True,
                'status': 'Optimal solution found',
                'optimal_inventory_levels': optimal_inventory,
                'total_holding_cost': total_cost,
                'optimization_details': {
                    'warehouses_optimized': len(warehouses),
                    'products_optimized': len(products),
                    'constraints_added': len(prob.constraints)
                }
            }
        else:
            return {
                'success': False,
                'error': f'Optimization failed with status: {LpStatus[status]}'
            }
