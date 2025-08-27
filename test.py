import requests

url = "https://api.reccobeats.com/v1/audio-features"
song_ids = ['1idGsoSIxhGaFFE7eteyUP', 'adf']
ids_param = ",".join(song_ids)
params = {"ids": ids_param}

# Make API request
response = requests.get(url, params=params, timeout=30)
print(response.json())