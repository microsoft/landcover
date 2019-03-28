import base64

with open("yourfile.ext", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())