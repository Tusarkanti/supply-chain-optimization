import requests
import json
import time

url = 'http://localhost:5000/api/login'

def test_login(email, password, expected_status=200, expected_error=None):
    data = {'email': email, 'password': password}
    try:
        response = requests.post(url, json=data)
        print(f"Testing {email} / {password}")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Login successful")
        else:
            resp_json = response.json()
            print(f"Error: {resp_json.get('error')}")
            if 'remaining_attempts' in resp_json:
                print(f"Remaining attempts: {resp_json['remaining_attempts']}")
        print("---")
        return response.status_code, resp_json if response.status_code != 200 else None
    except Exception as e:
        print(f"Request failed: {e}")
        return None, None

print("Testing login functionality after removing account locking...")

# Test correct login
print("1. Testing correct login for admin")
status, error = test_login('admin@supplychain.com', 'admin', 200)
if status != 200:
    print("ERROR: Correct login failed")
else:
    print("SUCCESS: Correct login worked")

# Test incorrect password
print("2. Testing incorrect password for admin")
status, error = test_login('admin@supplychain.com', 'wrong', 401)
if status == 401 and error and error.get('error') == 'Invalid email or password':
    print("SUCCESS: Incorrect password handled correctly")
else:
    print("ERROR: Incorrect password not handled properly")

# Test multiple failures
print("3. Testing multiple failed attempts (should not lock)")
for i in range(6):
    print(f"Attempt {i+1}")
    status, error = test_login('admin@supplychain.com', 'wrong', 401)
    if status == 423:
        print("ERROR: Account locked after failures!")
        break
    time.sleep(1)  # small delay
else:
    print("SUCCESS: No locking after multiple failures")

# Test correct login again after failures
print("4. Testing correct login after failures")
status, error = test_login('admin@supplychain.com', 'admin', 200)
if status == 200:
    print("SUCCESS: Login works after failures")
else:
    print("ERROR: Login failed after failures")

# Test demo user
print("5. Testing demo user correct login")
status, error = test_login('demo@supplychain.com', 'demo123', 200)
if status == 200:
    print("SUCCESS: Demo login worked")
else:
    print("ERROR: Demo login failed")

print("Testing complete.")
