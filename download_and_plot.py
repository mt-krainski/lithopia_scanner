import request_last_dataset
import plot_RGB_image
import os

SAMPLE_LOCATION = (50.083333, 14.416667) # Prague
DATA_PATH = "data"
IMAGE_PATH = "saved_images"

ARCHIVE_EXT = ".zip"

print("Forming request...")
response = request_last_dataset.request_sentinel_2_data(SAMPLE_LOCATION)

if response.status_code == 200:
    print("Request successful.")

entry = response.json()['feed']['entry'][0]

dataset_name = request_last_dataset.download(entry)

archive_path = os.path.join(DATA_PATH,
                            dataset_name+ARCHIVE_EXT)

image = plot_RGB_image.get_rgb_from_archive(archive_path)

print("Plotting...")
plot_RGB_image.plot_and_save(image, dataset_name)