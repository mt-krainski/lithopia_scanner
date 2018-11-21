import rasterio
import numpy as np
import zipfile
import warnings
import os
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from lxml import etree

class ArchiveContentsWarning(ResourceWarning):
    pass

Image.MAX_IMAGE_PIXELS = 1000000000

print("Starting script...")

DATA_PATH = "data"
IMAGE_PATH = "saved_images"

ARCHIVE_EXT = ".zip"

TCI_FILE_KEYWORD = "TCI.jp2"
RED_CHANNEL_KEYWORD = "B04.jp2"
GREEN_CHANNEL_KEYWORD = "B03.jp2"
BLUE_CHANNEL_KEYWORD = "B02.jp2"

VALUE_THRESHOLD = 2000

# Example dataset definition
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

ARCHIVE_PATH = os.path.join(
        DATA_PATH,
        DATASET_NAME + ARCHIVE_EXT)

def get_image_filename(archive, keyword=TCI_FILE_KEYWORD):
    image_file = None
    for file in archive.infolist():
        if keyword in file.filename:
            if image_file is None:
                image_file = file.filename
            else:
                warnings.warn("Warning! More than one TCI file found!", ArchiveContentsWarning)
    return image_file


def read_archive_file(image_filename,
                      archive_path = ARCHIVE_PATH):
    # Based on https://geopyspark.readthedocs.io/en/latest/tutorials/reading-in-sentinel-data.html
    with rasterio.open("zip://" + archive_path +
                       "!" + image_filename) as f:
        return f.read(1)


def get_rgb_from_archive(archive_path = ARCHIVE_PATH):

    print("Extracting image path...")

    archive = zipfile.ZipFile(archive_path)

    red_image_filename = get_image_filename(
            archive,
            RED_CHANNEL_KEYWORD)

    green_image_filename = get_image_filename(
            archive,
            GREEN_CHANNEL_KEYWORD)

    blue_image_filename = get_image_filename(
            archive,
            BLUE_CHANNEL_KEYWORD)

    print("Loading image...")

    red_image = read_archive_file(red_image_filename, archive_path)
    print("\tRED read")
    green_image = read_archive_file(green_image_filename, archive_path)
    print("\tGREEN read")
    blue_image = read_archive_file(blue_image_filename, archive_path)
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

    # scaling to 8-bit uints
    norm_factor = np.max(color_image) / 255

    # Processing row by row to save memory on conversion from int
    # to float due to division
    color_image = np.array(
        [(row/norm_factor).astype(np.uint8)
                for row in color_image] )

    return color_image


def get_tci_image(archive_path = ARCHIVE_PATH):
    img = None
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            if TCI_FILE_KEYWORD in entry.filename:
                # with archive.open(entry) as file:
                image_data = archive.read(entry)
                fh = BytesIO(image_data)
                img = Image.open(fh)
                break
    return img


def get_inspire_metadata(archive_path = ARCHIVE_PATH):
    INSPIRE_FILENAME = "INSPIRE.xml"
    inspire_file = None
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            if INSPIRE_FILENAME == os.path.basename(entry.filename):
                with archive.open(entry) as file:
                    inspire_file = etree.parse(file)
    return inspire_file


def get_namespaces_from_xml(xml_file):
    nsmap = {}
    for ns in xml_file.xpath('//namespace::*'):
        if ns[0]:  # Removes the None namespace, neither needed nor supported.
            nsmap[ns[0]] = ns[1]
    return nsmap


def get_bounding_box(inspire_xml):
    BOUNDING_BOX_ELEMENT = "gmd:EX_GeographicBoundingBox"
    ELEMENTS = {"west"  : "gmd:westBoundLongitude",
                "east"  : "gmd:eastBoundLongitude",
                "north" : "gmd:northBoundLatitude",
                "south" : "gmd:southBoundLatitude"}
    VALUE_ELEMENT = "gco:Decimal"

    limits = {}

    nsmap = get_namespaces_from_xml(inspire_xml)

    bounding_box = inspire_xml.xpath(
            f"//{BOUNDING_BOX_ELEMENT}", namespaces=nsmap)[0]

    for element_id in ELEMENTS:
        element = bounding_box.find(ELEMENTS[element_id], namespaces=nsmap)
        limits[element_id] = float(element.find(VALUE_ELEMENT, namespaces=nsmap).text)

    return limits


def plot_and_save(image, dataset_name = DATASET_NAME, limits = None):
    fig = plt.figure(figsize=(10, 10), dpi=300)

    if limits is not None:
        plt.imshow(image, extent=[
                limits["east"], limits["west"],
                limits["south"], limits["north"]
            ], aspect="auto")
    else:
        plt.imshow(image)
    fig.savefig(
        os.path.join(IMAGE_PATH, f"{dataset_name}.png")
    )
    plt.close(fig)

if __name__ == "__main__":
    image = get_tci_image()
    limits = get_bounding_box(get_inspire_metadata())

    print("Plotting...")

    plot_and_save(image, limits = limits)

