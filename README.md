## Lithopia scanner

Application for the Lithopia project, which extracts latest Sentinel-2 image for a given area and analyses it for a presence of a visual marker. If such marker is detected, the application is triggerring a blockchain contract.

## Notes:
- Sentinel 2 image resolution is 10m per pixel, revisit time is 5 days.
- [Copernicus data portal](https://scihub.copernicus.eu/dhus/#/home)
- [Copernicus data API](https://scihub.copernicus.eu/userguide/APIsOverview)
- [Sentinel-2 acquisition plans](https://sentinel.esa.int/web/sentinel/missions/sentinel-2/acquisition-plans)
- [Sentinel Application Platform](http://step.esa.int/main/toolboxes/snap/) - useful tool for initial visualization of data

## Used libraries:

*note: Development was done on Windows 10 and that's the system to which the installation notes apply*

- [rasterio](https://rasterio.readthedocs.io/en/latest/index.html) - *Also needed to be installed manually from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#rasterio) and required GDAL, which can be found [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal)*