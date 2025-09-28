#!/usr/bin/env python3
"""
Test script for futuristic inventory management module
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from futuristic_inventory_management import FuturisticInventoryManagementModule

def test_ai_demand_prediction():
    """Test AI-powered demand prediction"""
    print("Testing AI-powered demand prediction...")

    # Create sample historical data
    dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
    demands = [100 + 10 * np.sin(i/5) + np.random.normal(0, 5) for i in range(30)]

    historical_data = pd.DataFrame({
        'date': dates,
        'demand': demands
    })

    # External factors
    external_factors = {
        'weather': {
            'temperature': 22,
            'precipitation': 0,
            'severity': 0
        },
        'economic': {
            'inflation': 0.025,
            'unemployment': 0.045,
            'confidence': 105
        }
    }

    # Initialize module
    fim = FuturisticInventoryManagementModule()

    # Test prediction
    result = fim.predict_demand_with_ai(historical_data, external_factors)

    print(f"Prediction result: {result}")
    print("✓ AI demand prediction test completed\n")

def test_risk_assessment():
    """Test supply chain risk assessment"""
    print("Testing supply chain risk assessment...")

    inventory_data = {
        'average_daily_demand': 150,
        'lead_time_days': 7,
        'current_stock': 1000
    }

    supplier_data = {
        'reliability_score': 0.92,
        'backup_suppliers': 2
    }

    fim = FuturisticInventoryManagementModule()
    risk_assessment = fim.assess_supply_chain_risks(inventory_data, supplier_data)

    print(f"Overall risk score: {risk_assessment['overall_risk_score']:.3f}")
    print(f"Scenario impacts: {len(risk_assessment['scenario_impacts'])} scenarios analyzed")
    print("✓ Risk assessment test completed\n")

def test_sustainability_optimization():
    """Test sustainability optimization"""
    print("Testing sustainability optimization...")

    inventory_decisions = {
        'PROD001': {
            'transport_distance_km': 500,
            'warehouse_days': 15,
            'manufacturing_intensity': 1.2
        },
        'PROD002': {
            'transport_distance_km': 800,
            'warehouse_days': 20,
            'manufacturing_intensity': 0.8
        }
    }

    environmental_data = {
        'carbon_price_per_ton': 50,
        'renewable_energy_available': True,
        'recycling_programs': ['plastic', 'metal']
    }

    fim = FuturisticInventoryManagementModule()
    sustainability_opt = fim.optimize_sustainability(inventory_decisions, environmental_data)

    print(f"Carbon footprint reduction: {sustainability_opt['carbon_footprint_reduction']:.1f}%")
    print(f"Sustainability score: {sustainability_opt['sustainability_score']:.1f}")
    print("✓ Sustainability optimization test completed\n")

def test_iot_integration():
    """Test IoT sensor data integration"""
    print("Testing IoT sensor integration...")

    sensor_data = {
        'sensor_001': {
            'type': 'inventory_level',
            'current_stock': 850,
            'predicted_stock': 820,
            'threshold': 50
        },
        'sensor_002': {
            'type': 'environmental',
            'temperature': 18,
            'humidity': 65
        },
        'sensor_003': {
            'type': 'equipment',
            'vibration': 2.1,
            'equipment_temp': 45,
            'equipment_type': 'forklift'
        }
    }

    fim = FuturisticInventoryManagementModule()
    iot_result = fim.integrate_iot_data(sensor_data)

    print(f"Sensors active: {iot_result['sensors_active']}")
    print(f"Anomalies detected: {len(iot_result['anomalies_detected'])}")
    print(f"Inventory accuracy improvement: {iot_result['inventory_accuracy_improvement']:.1f}%")
    print("✓ IoT integration test completed\n")

def test_autonomous_decisions():
    """Test autonomous decision making"""
    print("Testing autonomous decision making...")

    current_state = {
        'inventory_levels': {
            'PROD001': {
                'current_stock': 120,
                'predicted_demand': 25,
                'reorder_point': 100
            },
            'PROD002': {
                'current_stock': 80,
                'predicted_demand': 20,
                'reorder_point': 150
            }
        }
    }

    historical_performance = {
        'accuracy_rate': 0.94,
        'cost_savings': 12500,
        'stockout_events': 2
    }

    fim = FuturisticInventoryManagementModule()
    decisions = fim.make_autonomous_decisions(current_state, historical_performance)

    print(f"Decisions made: {len(decisions['decisions_made'])}")
    print(f"Learning insights: {len(decisions['learning_insights'])}")
    print("✓ Autonomous decisions test completed\n")

def test_futuristic_metrics():
    """Test futuristic metrics calculation"""
    print("Testing futuristic metrics...")

    fim = FuturisticInventoryManagementModule()
    metrics = fim.get_futuristic_metrics()

    print(f"Predictive accuracy: {metrics.predictive_accuracy:.1f}")
    print(f"Sustainability score: {metrics.sustainability_score:.1f}")
    print(f"Risk resilience index: {metrics.risk_resilience_index:.1f}")
    print(f"Autonomous decisions: {metrics.autonomous_decisions}")
    print(f"IoT sensors active: {metrics.iot_sensors_active}")
    print(f"Carbon footprint reduction: {metrics.carbon_footprint_reduction:.1f}%")
    print("✓ Futuristic metrics test completed\n")

def main():
    """Run all tests"""
    print("🚀 Testing Futuristic Inventory Management Module\n")
    print("=" * 60)

    try:
        test_ai_demand_prediction()
        test_risk_assessment()
        test_sustainability_optimization()
        test_iot_integration()
        test_autonomous_decisions()
        test_futuristic_metrics()

        print("=" * 60)
        print("✅ All futuristic inventory management tests completed successfully!")
        print("\n🎯 Key Features Implemented:")
        print("  • AI-powered demand prediction with external factors")
        print("  • Comprehensive supply chain risk assessment")
        print("  • Sustainability optimization with carbon tracking")
        print("  • IoT sensor integration for real-time monitoring")
        print("  • Autonomous decision making with learning")
        print("  • Advanced metrics and performance tracking")

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
