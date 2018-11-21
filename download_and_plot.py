import sentinel_requests
import images
import os

SAMPLE_LOCATION = (50.083333, 14.416667) # Prague
DATA_PATH = "data"
IMAGE_PATH = "saved_images"

ARCHIVE_EXT = ".zip"

print("Forming request...")
response = sentinel_requests.request_sentinel_2_data(SAMPLE_LOCATION)

if response.status_code == 200:
    print("Request successful.")

entry = response.json()['feed']['entry'][0]

dataset_name = sentinel_requests.download(entry)

archive_path = os.path.join(DATA_PATH,
                            dataset_name+ARCHIVE_EXT)

image = images.get_tci_image(archive_path)

print("Plotting...")
images.plot_and_save(image, dataset_name)