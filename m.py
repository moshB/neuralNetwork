import json

with open("data.json", "r") as json_file:
  data = json.load(json_file)
  # Now you can access the data using keys
  print(data["city"])  # Assuming "name" is a key in your JSON
