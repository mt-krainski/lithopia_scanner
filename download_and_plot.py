import request_last_dataset
import plot_RGB_image
import os

SAMPLE_LOCATION = (50.083333, 14.416667) # Prague
DATA_PATH = "data"
IMAGE_PATH = "saved_images"

ARCHIVE_EXT = ".zip"

print("Forming request...")
response = request_last_dataset.request_sentinel_2_data(SAMPLE_LOCATION)
entry = response.json()['feed']['entry'][0]
download_link = request_last_dataset.get_data_link(entry)

request_last_dataset.download_if_not_present(*download_link)

archive_path = os.path.join(DATA_PATH,
                            download_link[1])

dataset_name = download_link[1].split('.')[0]

image = plot_RGB_image.get_rgb_from_archive(archive_path)

print("Plotting...")
plot_RGB_image.plot_and_save(image, dataset_name)