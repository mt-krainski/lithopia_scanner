import rasterio
import numpy as np
import zipfile
import warnings
import os
import matplotlib.pyplot as plt

class ArchiveContentsWarning(ResourceWarning):
    pass

print("Starting script...")

DATA_PATH = "data"
SAT_TYPE = "S2A"
PROD_LEVEL = "MSIL1C"
SENSING_START = "20181016T101021"
PROC_BASELINE = "N0206"
RELATIVE_ORBIT_NUMBER = "R022"
TILE_NUMBER_FIELD = "T33UVR"
TIMESTAMP = "20181016T121930"
DATASET_NAME = "_".join([
        SAT_TYPE,
        PROD_LEVEL,
        SENSING_START,
        PROC_BASELINE,
        RELATIVE_ORBIT_NUMBER,
        TILE_NUMBER_FIELD,
        TIMESTAMP,
])
#"S2A_MSIL1C_20181016T101021_N0206_R022_T33UVR_20181016T121930"
ARCHIVE_EXT = ".zip"

TCI_FILE_KEYWORD = "TCI.jp2"
RED_CHANNEL_KEYWORD = "B04.jp2"
GREEN_CHANNEL_KEYWORD = "B03.jp2"
BLUE_CHANNEL_KEYWORD = "B02.jp2"

VALUE_THRESHOLD = 3000

IMAGE_PATH = ""

ARCHIVE_PATH = os.path.join(DATA_PATH, DATASET_NAME + ARCHIVE_EXT)

def get_image_filename(archive, keyword=TCI_FILE_KEYWORD):
    image_file = None
    for file in archive.infolist():
        if keyword in file.filename:
            if image_file is None:
                image_file = file.filename
            else:
                warnings.warn("Warning! More than one TCI file found!", ArchiveContentsWarning)
    return image_file

print("Extracting image path...")

archive = zipfile.ZipFile(ARCHIVE_PATH)

tci_image_filename = get_image_filename(archive)
red_image_filename = get_image_filename(archive, RED_CHANNEL_KEYWORD)
green_image_filename = get_image_filename(archive, GREEN_CHANNEL_KEYWORD)
blue_image_filename = get_image_filename(archive, BLUE_CHANNEL_KEYWORD)

print("Loading image...")

def read_archive_file(image_filename, archive_path = ARCHIVE_PATH):
    # Based on https://geopyspark.readthedocs.io/en/latest/tutorials/reading-in-sentinel-data.html
    with rasterio.open("zip://" + archive_path + "!" + image_filename) as f:
        image = f.read(1)
    return image

red_image = read_archive_file(red_image_filename)
print("\tRED read")
green_image = read_archive_file(green_image_filename)
print("\tGREEN read")
blue_image = read_archive_file(blue_image_filename)
print("\tBLUE read")

print("Composing RGB image...")

color_image = np.stack(
        (red_image, green_image, blue_image),
        axis=2)

del red_image, green_image, blue_image

np.place(
    color_image,
    color_image>VALUE_THRESHOLD,
    VALUE_THRESHOLD)

norm_factor = np.max(color_image) / 255

# Processing row by row to save memory on conversion from int
# to float due to division
color_image = np.array(
    [(row/norm_factor).astype(np.uint8) for row in color_image]
)

print("Plotting...")
imgplot = plt.imshow(color_image)
plt.show()

def read_zip_file(filepath):
    zfile = zipfile.ZipFile(filepath)
    for finfo in zfile.infolist():
        ifile = zfile.open(finfo)
        line_list = ifile.readlines()
        print(line_list)

