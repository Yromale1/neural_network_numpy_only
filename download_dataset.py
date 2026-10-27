
import urllib.request
import zipfile
import os

os.makedirs("mnist", exist_ok=True)

base = "https://raw.githubusercontent.com/phoebetronic/mnist/master/"

for filename in ["mnist_train.csv.zip", "mnist_test.csv.zip"]:
    zip_path = os.path.join("mnist", filename)

    urllib.request.urlretrieve(base + filename, zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("mnist")

    print("Downloaded:", filename)