import requests

# Define the URL of the endpoint
URL = "http://localhost:5000/predict"

# Define the path to the test image
IMAGE_PATH = "test_image.jpg"

# Open the test image file
with open(IMAGE_PATH, "rb") as f:
    image_data = f.read()

# Define the payload to send to the server
payload = {"image": image_data}

# Send a POST request to the server with the image data
response = requests.post(URL, files=payload)

# Print the prediction result
print(response.text)