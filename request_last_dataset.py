import credentials
from lxml import etree

SAMPLE_LOCATION = (50.083333, 14.416667) # Prague

MASTER_URI = "https://scihub.copernicus.eu/dhus/search?"

PLATFORM_NAME = "Sentinel-2"
PRODUCT_TYPE = "S2MSI1C"


def request_sentinel_2_data(location):
    request_uri = f"{MASTER_URI}start=0&rows=10&" \
                  f"q=footprint:\"Intersects({location[0]}, {location[1]})\" AND " \
                  f"platformname:{PLATFORM_NAME} AND " \
                  f"producttype: {PRODUCT_TYPE}&" \
                  f"orderby=beginposition desc"
    response = credentials.request(request_uri)
    return response

