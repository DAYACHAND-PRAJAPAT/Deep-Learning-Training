# predict_client.py
import requests
import sys

def predict(image_path, url="http://localhost:5000/predict"):
    with open(image_path, "rb") as f:
        files = {"file": (image_path, f)}
        try:
            resp = requests.post(url, files=files)
            print("Status Code:", resp.status_code)
            print("Response Body:", resp.text)
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to the server: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_client.py path/to/image.jpg")
    else:
        predict(sys.argv[1])