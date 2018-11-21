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


class ImageOutOfBoundsException(Exception):
    pass


Image.MAX_IMAGE_PIXELS = 1000000000

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
    print("Getting TCI image...")
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


def read_xml_from_archive(archive_path, xml_filename):
    xml_file = None
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            if xml_filename == os.path.basename(entry.filename):
                with archive.open(entry) as file:
                    xml_file = etree.parse(file)
    return xml_file


def get_inspire_metadata(archive_path = ARCHIVE_PATH):
    INSPIRE_FILENAME = "INSPIRE.xml"
    return read_xml_from_archive(archive_path, INSPIRE_FILENAME)


def get_manifest(archive_path = ARCHIVE_PATH):
    MANIFEST_FILENAME = 'manifest.safe'
    return read_xml_from_archive(archive_path, MANIFEST_FILENAME)


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

    inspire_root = inspire_xml.getroot()

    bounding_box = inspire_root.xpath(
            f"//{BOUNDING_BOX_ELEMENT}", namespaces=inspire_root.nsmap)[0]

    for element_id in ELEMENTS:
        element = bounding_box.find(ELEMENTS[element_id], namespaces=inspire_root.nsmap)
        limits[element_id] = float(element.find(VALUE_ELEMENT, namespaces=inspire_root.nsmap).text)

    return limits


def get_coordinates(manifest_xml):
    COORDINATES_TAG = "gml:coordinates"

    manifest_root = manifest_xml.getroot()

    coordinates = manifest_xml.xpath(f"//{COORDINATES_TAG}/text()", namespaces=manifest_root.nsmap)[0]

    coordinates_float = [float(x) for x in coordinates.strip().split(" ")]

    coordinates_mapped = [coordinates_float[1::2], coordinates_float[0::2]]

    return coordinates_mapped


def plot_and_save(image, dataset_name = DATASET_NAME, limits = None):

    fig = plt.figure(figsize=(10, 10), dpi=300)

    if limits is not None:
        plt.imshow(image, extent=[
            limits["west"], limits["east"],
            limits["south"], limits["north"]
        ], aspect="auto")
    else:
        plt.imshow(image)
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        # Stores the image in full resolution
        # plt.imsave(
        #     os.path.join(IMAGE_PATH, f"{dataset_name}.png"),
        #     image,
        #     format='png')

    fig.savefig(
        os.path.join(IMAGE_PATH, f"{dataset_name}.png"),
        bbox_inches="tight"
    )
    plt.close(fig)


def get_ratio(limits, value):
    return (value-min(limits))/abs(limits[0] - limits[1])


def crop(image, bounding_box, crop_bounding_box):
    if bounding_box["east"] < crop_bounding_box["east"] or \
            bounding_box["west"] > crop_bounding_box["west"] or \
            bounding_box["south"] > crop_bounding_box["south"] or \
            bounding_box["north"] < crop_bounding_box["north"]:
        raise ImageOutOfBoundsException("Requested bounding box is outside of image bounds")
    cutout_percent = {
        "east" : get_ratio((bounding_box["east"], bounding_box["west"]), crop_bounding_box["east"]),
        "west": get_ratio((bounding_box["east"], bounding_box["west"]), crop_bounding_box["west"]),
        "north": 1.0 - get_ratio((bounding_box["north"], bounding_box["south"]), crop_bounding_box["north"]), ## images are indexed from top
        "south": 1.0 - get_ratio((bounding_box["north"], bounding_box["south"]), crop_bounding_box["south"]),
    }
    print(cutout_percent)
    cutout_pixel = {}
    width, height = image.size
    if cutout_percent["east"] < cutout_percent["west"]:
        cutout_pixel["left"] = int(np.floor(cutout_percent["east"] * width))
        cutout_pixel["right"] = int(np.ceil(cutout_percent["west"] * width))
    else:
        cutout_pixel["right"] = int(np.ceil(cutout_percent["east"] * width))
        cutout_pixel["left"] = int(np.floor(cutout_percent["west"] * width))
    if cutout_percent["south"] < cutout_percent["north"]:
        cutout_pixel["upper"] = int(np.ceil(cutout_percent["north"] * height))
        cutout_pixel["lower"] = int(np.floor(cutout_percent["south"] * height))
    else:
        cutout_pixel["lower"] = int(np.floor(cutout_percent["north"] * height))
        cutout_pixel["upper"] = int(np.ceil(cutout_percent["south"] * height))
    print(cutout_pixel)
    crop_result = image.crop((
        cutout_pixel["left"],
        cutout_pixel["lower"],
        cutout_pixel["right"],
        cutout_pixel["upper"],
    ))
    return crop_result


if __name__ == "__main__":
    print("Starting script...")
    image = get_tci_image()
    bounding_box = get_bounding_box(get_inspire_metadata())

    print("Plotting...")

    plot_and_save(image, limits = bounding_box)


    TEST_CUTOUT_BOX = {
        'east': 14.357152,
        'north' : 50.055195,
        'west' : 14.314761,
        'south' : 50.024767
    }
