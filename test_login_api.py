import requests
import json

url = 'http://localhost:5000/api/login'
data = {
    'email': 'admin@supplychain.com',
    'password': 'admin123'
}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print(f"Response Body: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")
