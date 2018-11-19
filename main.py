import rasterio
import geopyspark as gps
import numpy as np
import zipfile
import warnings
import os

from pyspark import SparkContext

class ArchiveContentsWarning(ResourceWarning):
    pass

print("Starting script...")

conf = gps.geopyspark_conf(
    master="local[*]",
    appName="sentinel-ingest-example",)
pysc = SparkContext(conf=conf)

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

IMAGE_PATH = ""

def get_tci_image_filename(archive):
    image_file = None
    for file in archive.infolist():
        if TCI_FILE_KEYWORD in file.filename:
            if image_file is None:
                image_file = file.filename
            else:
                warnings.warn("Warning! More than one TCI file found!", ArchiveContentsWarning)
    return image_file

print("Extracting image path...")

archive_path = os.path.join(DATA_PATH, DATASET_NAME+ARCHIVE_EXT)

archive = zipfile.ZipFile(archive_path)

image_file = get_tci_image_filename(archive)

print("Loading image...")

# Based on https://geopyspark.readthedocs.io/en/latest/tutorials/reading-in-sentinel-data.html
with rasterio.open("zip://" + archive_path + "!" + image_file) as f:
    image = f.read(1)
    extent = gps.Extent(*f.bounds)
    projected_extent = gps.ProjectedExtent(
        extent=extent,
        epsg=int(f.crs.to_dict()['init'][5:]))
    tile = gps.Tile.from_numpy_array(
        numpy_array=image,
        no_data_value=f.nodata)
    rdd = pysc.parallelize([(projected_extent, tile)])
    raster_layer = gps.RasterLayer.from_numpy_rdd(
        layer_type=gps.LayerType.SPATIAL,
        numpy_rdd=rdd)


def read_zip_file(filepath):
    zfile = zipfile.ZipFile(filepath)
    for finfo in zfile.infolist():
        ifile = zfile.open(finfo)
        line_list = ifile.readlines()
        print(line_list)

