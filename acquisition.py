from bs4 import BeautifulSoup
import requests
from lxml import etree
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class Acquisition:
    def __init__(self, polygon, observation_start, observation_end):
        self.polygon = polygon
        self.observation_start = observation_start
        self.observation_end = observation_end

SENTINEL_URL_BASE = "https://sentinel.esa.int"
ACQUISITION_PLANS = "/web/sentinel/missions/sentinel-2/acquisition-plans"

acquisition_html = requests.get(f"{SENTINEL_URL_BASE}{ACQUISITION_PLANS}").text

page_parser = BeautifulSoup(acquisition_html, 'html.parser')
plan_link_a = page_parser.find('div', 'sentinel-2a').find('a')['href']
plan_link_b = page_parser.find('div', 'sentinel-2b').find('a')['href']

kml_a = etree.XML(requests.get(f"{SENTINEL_URL_BASE}{plan_link_a}").content)
kml_b = etree.XML(requests.get(f"{SENTINEL_URL_BASE}{plan_link_b}").content)

def get_acquisition_plan(kml_file):
    PLACEMARK = "Placemark"
    KML = "kml"
    nsmap = kml_file.nsmap
    if None in nsmap:
        del nsmap[None] ## emtpy namespaces are not supported

    placemarks = kml_file.xpath(f"//{KML}:{PLACEMARK}", namespaces=nsmap)



