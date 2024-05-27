import requests

# URL of the FastAPI endpoint
url = 'http://0.0.0.0:5002/health'
response = requests.get(url, files=files)

# Print the response
print(response.json())