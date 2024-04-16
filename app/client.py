import requests

response = requests.post('http://localhost:8000/generate/invoke',json={"input":{"prompt":"provide me an essay about the topic of the day"},"config": {},
  "kwargs": {}})


print(response.content)