import sentinel_requests
import images
import os
from matplotlib import pyplot as plt

SAMPLE_LOCATION = (50.083333, 14.416667) # Prague
DATA_PATH = "data"
IMAGE_PATH = "saved_images"

ARCHIVE_EXT = ".zip"

print("Forming request...")
response = sentinel_requests.get_latest(SAMPLE_LOCATION)

if response.status_code == 200:
    print("Request successful.")

entries = response.json()['feed']['entry']

entry = sentinel_requests.get_latest_with_cloud_limit(entries, 2)

print(f"Latest image from: {entry['date'][0]['content']}")

dataset_name = sentinel_requests.download(entry)

archive_path = os.path.join(DATA_PATH,
                            dataset_name+ARCHIVE_EXT)

image = images.get_tci_image(archive_path)

print("Plotting...")
images.plot_and_save(image, dataset_name)

# plt.imsave(os.path.join(IMAGE_PATH, f"{dataset_name}.png"),
#         image,
#         format='png')