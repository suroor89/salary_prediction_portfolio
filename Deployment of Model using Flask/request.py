import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'yearsExperience':1, 'milesFromMetropolis':10})

print(r.json())