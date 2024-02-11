import requests
resp = requests.post('https://textbelt.com/text', {
  'phone': '+919550260588',
  'message': 'Suspicious Activity Detected Please check up on your items !!!',
  'key': 'textbelt',
})
print(resp.json())