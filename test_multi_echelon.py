import requests
import json

# Test script for multi-echelon optimization endpoint

def test_multi_echelon_optimization():
    base_url = "http://localhost:5000"

    # First, login to get JWT token
    login_data = {
        "email": "admin@supplychain.com",
        "password": "admin"
    }

    try:
        # Login
        login_response = requests.post(f"{base_url}/api/login", json=login_data)
        print(f"Login status: {login_response.status_code}")

        if login_response.status_code == 200:
            token = login_response.json()['access_token']
            headers = {'Authorization': f'Bearer {token}'}

            # Test multi-echelon optimization
            optimization_data = {
                "warehouses": [
                    {"id": "WH001", "name": "Main Warehouse"},
                    {"id": "WH002", "name": "Regional Warehouse"}
                ],
                "products": [
                    {"id": "PROD001", "name": "Product A", "holding_cost": 1.0},
                    {"id": "PROD002", "name": "Product B", "holding_cost": 1.5}
                ],
                "demand_forecasts": {
                    "PROD001": {"WH001": 100, "WH002": 50},
                    "PROD002": {"WH001": 80, "WH002": 40}
                },
                "supplier_data": {
                    "SUP001": {"lead_time": 5, "cost": 10.0}
                }
            }

            print("Testing multi-echelon optimization...")
            response = requests.post(
                f"{base_url}/api/inventory/multi-echelon-optimize",
                json=optimization_data,
                headers=headers
            )

            print(f"Optimization status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print("Success! Optimization result:")
                print(json.dumps(result, indent=2))
            else:
                print(f"Error: {response.text}")

        else:
            print(f"Login failed: {login_response.text}")

    except requests.exceptions.ConnectionError:
        print("Connection error: Make sure the server is running on localhost:5000")
    except Exception as e:
        print(f"Test error: {str(e)}")

if __name__ == "__main__":
    test_multi_echelon_optimization()
