import unittest
from lxml import etree
from shapely.geometry.polygon import Polygon


from sentinel2 import acquisition

TEST_FILE_A = 'resources/_test.kml'
TEST_FILE_B = 'resources/_test2.kml'

class TestAcquisition(unittest.TestCase):
    def test_get_acquisition(self):
        result = acquisition.get_acquisition_plan()

        self.assertTrue(len(result)>0)

    def test_parse_kml(self):
        kml_content = etree.parse(TEST_FILE_A).getroot()
        result_a = acquisition.decode_kml_file(kml_content, "Sentinel-2A")

        self.assertIsInstance(result_a, list)
        self.assertIsInstance(result_a[0], acquisition.AcquisitionSwath)
        self.assertEqual(result_a[0].satellite, "Sentinel-2A")
        self.assertIsInstance(result_a[0].polygon, Polygon)


        kml_content = etree.parse(TEST_FILE_B).getroot()
        result_b = acquisition.decode_kml_file(kml_content, "Sentinel-2B")

        self.assertIsInstance(result_b, list)
        self.assertIsInstance(result_b[0], acquisition.AcquisitionSwath)
        self.assertEqual(result_b[0].satellite, "Sentinel-2B")
        self.assertIsInstance(result_b[0].polygon, Polygon)

