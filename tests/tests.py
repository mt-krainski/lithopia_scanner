import unittest
from sentinel2 import acquisition

class TestAcquisition(unittest.TestCase):
    def test_get_acquisition(self):
        result = acquisition.get_acquisition_plan()

        self.assertTrue(len(result)>0)
