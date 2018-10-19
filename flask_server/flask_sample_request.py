# importing the requests library
import argparse
import base64

import requests

API_ENDPOINT = "http://localhost:5000/imageclassifier/predict/"

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path of the image")
args = vars(ap.parse_args())

# defining the api-endpoint
image_path = args['image']
b64_image = ""
with open(image_path, "rb") as imageFile:
    b64_image = base64.b64encode(imageFile.read())

# data to be sent to api
data = {'b64': b64_image}

# sending post request and saving response as response object
r = requests.post(url=API_ENDPOINT, data=data)

# extracting the response
print("{}".format(r.text))
