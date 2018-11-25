from bs4 import BeautifulSoup
import requests
from lxml import etree
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import datetime

class AcquisitionSwath:
    def __init__(self,
                 name = None,
                 polygon = None,
                 observation_start = None,
                 observation_end = None,
                 id = None,
                 timeliness = None,
                 station = None,
                 mode = None,
                 orbit_absolute = None,
                 orbit_relative = None,
                 scenes = None):
        self.polygon = polygon
        self.ObservationTimeStart = observation_start
        self.ObservationTimeStop = observation_end
        self.name = name
        self.ID = id
        self.Timeliness = timeliness
        self.Station = station
        self.Mode = mode
        self.OrbitAbsolute = orbit_absolute
        self.OrbitRelative = orbit_relative
        self.Scenes = scenes

    NAME_PATH = 'kml:name'
    DATA_PATH = 'kml:ExtendedData'
    POLYGON_PATH = 'kml:Polygon/kml:outerBoundaryIs/kml:LinearRing/kml:coordinates'

    @staticmethod
    def from_placemark_xml(placemark_xml):
        nsmap = placemark_xml.nsmap
        if None in nsmap:
            del nsmap[None]  ## emtpy namespaces are not supported

        class_object = AcquisitionSwath()
        class_object.name = placemark_xml.xpath(AcquisitionSwath.NAME_PATH, namespaces=nsmap)[0].text

        for data_object in placemark_xml.find(AcquisitionSwath.DATA_PATH).iter():
            class_object.__dict__[data_object.get('name')] = data_object.find('kml:value', namespaces=nsmap).text

        class_object.parse_dates()
        polygon_coordinates = placemark_xml.find(AcquisitionSwath.POLYGON_PATH,
                                                 namespaces=nsmap).text
        polygon_definition = []

        for coordinate in polygon_coordinates.split(" "):
            point = [float(x) for x in coordinate.split(',')]
            polygon_definition.append(point[0:2])

        class_object.polygon = Polygon(polygon_definition)

        return class_object

    def parse_dates(self):
        self.ObservationTimeStart = datetime.datetime.strptime(self.ObservationTimeStart, "%Y-%m-%dT%H:%M:%S.%fZ")
        self.ObservationTimeStop = datetime.datetime.strptime(self.ObservationTimeStop, "%Y-%m-%dT%H:%M:%S.%fZ")

    def get_acquisition_period(self):
        return [self.ObservationTimeStart, self.ObservationTimeStop]

    def is_point_in_polygon(self, point):
        if not isinstance(point, Point):
            point = Point(point)

        return self.polygon.contains(point)


def get_acquisition_plan(kml_file):
    PLACEMARK_PATH = "kml:Placemark"
    nsmap = kml_file.nsmap
    if None in nsmap:
        del nsmap[None] ## emtpy namespaces are not supported
    placemarks = kml_file.xpath(f"//{PLACEMARK_PATH}", namespaces=nsmap)
    placemarks_decoded = []
    for placemark in placemarks:
        placemarks_decoded.append(AcquisitionSwath.from_placemark_xml(placemark))
    return placemarks_decoded



SAMPLE_LOCATION = (50.083333, 14.416667) # Prague

SENTINEL_URL_BASE = "https://sentinel.esa.int"
ACQUISITION_PLANS = "/web/sentinel/missions/sentinel-2/acquisition-plans"

acquisition_html = requests.get(f"{SENTINEL_URL_BASE}{ACQUISITION_PLANS}").text

page_parser = BeautifulSoup(acquisition_html, 'html.parser')
plan_link_a = page_parser.find('div', 'sentinel-2a').find('a')['href']
plan_link_b = page_parser.find('div', 'sentinel-2b').find('a')['href']

kml_a = etree.XML(requests.get(f"{SENTINEL_URL_BASE}{plan_link_a}").content)
kml_b = etree.XML(requests.get(f"{SENTINEL_URL_BASE}{plan_link_b}").content)

placemarks = get_acquisition_plan(kml_a) + get_acquisition_plan(kml_b)

matching_acquisitions = []

for placemark in placemarks:
    if placemark.is_point_in_polygon(SAMPLE_LOCATION):
        matching_acquisitions.append(placemark.get_acquisition_period())

print("Matching acquisitions:")
print(matching_acquisitions)