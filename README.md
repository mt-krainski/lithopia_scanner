## Lithopia scanner

Application for the Lithopia project, which extracts latest Sentinel-2 image for a given area and analyses it for a presence of a visual marker. If such marker is detected, the application is triggerring a blockchain contract.

## Notes:
- Sentinel 2 image resolution is 10m per pixel, revisit time is 5 days.
- [Copernicus data portal](https://scihub.copernicus.eu/dhus/#/home)
- [Copernicus data API](https://scihub.copernicus.eu/userguide/APIsOverview)
- [Sentinel-2 acquisition plans](https://sentinel.esa.int/web/sentinel/missions/sentinel-2/acquisition-plans)
- [Sentinel Application Platform](http://step.esa.int/main/toolboxes/snap/) - useful tool for initial visualization of data
- `geopyspark` requires spark (can be installed with `pip install pyspark`). It also requires an environmental variable SPARK\_HOME to be set. If installed with `pip`, pyspark creates a script `find_spark_home.py`, which can be used to determine to correct path
- `geopyspark` requires to run `geopyspark install-jars` after it's installation
- `spark` requires `winutils.exe` under Windows. Download winutils.exe from [here](http://public-repo-1.hortonworks.com/hdp-win-alpha/winutils.exe), copy it to a folder (e.g. `C:\winutils\bin`), then point `HADOOP_HOME` to that folder

## Used libraries:

*note: Development was done on Windows 10 and that's the system to which the installation notes apply*

- [geopyspark] (https://geopyspark.readthedocs.io/en/latest/index.html) - *I needed to install shapely manually from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely). For some reason, installation of egg\_info was failing*
- [rasterio] (https://rasterio.readthedocs.io/en/latest/index.html) - *Also needed to be installed manually from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#rasterio) and required GDAL, which can be found [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal)*
