import requests

try:
    response = requests.get("http://localhost:8501")
    print(response.status_code)
except Exception as e:
    print(e)
