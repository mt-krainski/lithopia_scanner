![](https://travis-ci.org/mt-krainski/lithopia_scanner.svg?branch=master)

## Lithopia scanner

Application for the Lithopia project, which extracts latest Sentinel-2 image for a given area and analyses it for a presence of a visual marker. If such marker is detected, the application is triggerring a blockchain contract.

## Functionalities:
- script for storing and retrieving credentials in a local file
- querying Copernicus Open Data Hub and retrieving Sentinel 2 images
- retrieving, plotting, and trimming images from the Sentinel 2 datasets
- finding a mapping from geo coordinates to pixel coordinates (coarse, but works)
- checking acquisition plans for a given location

## Notes:
- Sentinel 2 image resolution is 10m per pixel, revisit time is 5 days.
- [Copernicus data portal](https://scihub.copernicus.eu/dhus/#/home)
- [Copernicus data API](https://scihub.copernicus.eu/userguide/APIsOverview)
- [Sentinel-2 acquisition plans](https://sentinel.esa.int/web/sentinel/missions/sentinel-2/acquisition-plans)
- [Sentinel Application Platform](http://step.esa.int/main/toolboxes/snap/) - useful tool for initial visualization of data

## Used libraries:

*note: Development was done on Windows 10 and that's the system to which the installation notes apply*

- [rasterio](https://rasterio.readthedocs.io/en/latest/index.html) - *Also needed to be installed manually from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#rasterio) and required GDAL, which can be found [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal)*
- [pillow](https://pillow.readthedocs.io/en/5.2.x/handbook/image-file-formats.html#jpeg-2000) is used to read JPEG2000 data. It uses OpenJPEG
