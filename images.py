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
INSPIRE_FILENAME = "INSPIRE.xml"
MANIFEST_FILENAME = 'manifest.safe'

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
                warnings.warn(f"Warning! More than one {keyword} file found!", ArchiveContentsWarning)
    return image_file


def get_tci_image(archive_path):
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


def get_inspire_metadata(archive_path):
    return read_xml_from_archive(archive_path, INSPIRE_FILENAME)


def get_manifest(archive_path):
    return read_xml_from_archive(archive_path, MANIFEST_FILENAME)


def get_bounding_box(inspire_xml):
    BOUNDING_BOX_ELEMENT = "gmd:EX_GeographicBoundingBox"
    ELEMENTS = {"west"  : "gmd:westBoundLongitude",
                "east"  : "gmd:eastBoundLongitude",
                "north" : "gmd:northBoundLatitude",
                "south" : "gmd:southBoundLatitude"}
    VALUE_ELEMENT = "gco:Decimal"

    limits = {}

    inspire_root = inspire_xml.getroot()
    nsmap = inspire_root.nsmap.copy()
    if None in nsmap:
        del nsmap[None]

    bounding_box = inspire_root.xpath(
            f"//{BOUNDING_BOX_ELEMENT}", namespaces=nsmap)[0]

    for element_id in ELEMENTS:
        element = bounding_box.find(ELEMENTS[element_id], namespaces=nsmap)
        limits[element_id] = float(element.find(VALUE_ELEMENT, namespaces=nsmap).text)

    return limits


def get_coordinates(manifest_xml):
    COORDINATES_TAG = "gml:coordinates"

    manifest_root = manifest_xml.getroot()

    nsmap = manifest_root.nsmap.copy()
    if None in nsmap:
        del nsmap[None]

    coordinates = manifest_xml.xpath(f"//{COORDINATES_TAG}/text()", namespaces=nsmap)[0]

    coordinates_float = [float(x) for x in coordinates.strip().split(" ")]

    coordinates_mapped = [coordinates_float[1::2], coordinates_float[0::2]]

    return coordinates_mapped


def crop_by_coords(bounding_box, image, transform_function):
    """
    Crops the image to a bounding box given in geographical coordinates
    :param bounding_box: dict storing keys 'lower', 'left', 'upper', 'right',
        holding geo coordinates of the bounding box
    :param image: pillow image to be cropped
    :param transform_function: function capable of transorming geo coordinates
        to image pixels
    :return: cropped image
    """
    bounding_pixels = transform_function(
        [[bounding_box['left'], bounding_box['right']],
         [bounding_box['upper'], bounding_box['lower']]])

    bounding_pixels = {
        'left': bounding_pixels[0][0],
        'right': bounding_pixels[0][1],
        'upper': bounding_pixels[1][0],
        'lower': bounding_pixels[1][1]
    }

    return image.crop((
        bounding_pixels['left'],
        bounding_pixels['upper'],
        bounding_pixels['right'],
        bounding_pixels['lower']
    ))


def plot_and_save(image, dataset_name):

    fig = plt.figure(figsize=(10, 10), dpi=300)

    plt.imshow(image)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                    hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    fig.savefig(
        os.path.join(IMAGE_PATH, f"{dataset_name}.png"),
        bbox_inches="tight"
    )
    plt.close(fig)


if __name__ == "__main__":
    print("Starting script...")
    image = get_tci_image(ARCHIVE_PATH)

    print("Plotting...")

    plot_and_save(image, DATASET_NAME)
