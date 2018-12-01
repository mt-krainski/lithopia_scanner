from sentinel2 import credentials
from os import path
from sys import stdout
from io import BytesIO
from PIL import Image

from threading import Thread, Lock

SAMPLE_LOCATION = (50.083333, 14.416667) # Prague

MASTER_URI = "https://scihub.copernicus.eu/dhus/search?"

PLATFORM_NAME = "Sentinel-2"
PRODUCT_TYPE = "S2MSI1C"

DATA_PATH = "data"
ARCHIVE_EXT = ".zip"


def get_latest(location):
    request_uri = f"{MASTER_URI}start=0&rows=50&" \
                  f"q=footprint:\"Intersects({location[0]}, {location[1]})\" AND " \
                  f"platformname:{PLATFORM_NAME} AND " \
                  f"producttype: {PRODUCT_TYPE}&" \
                  f"orderby=beginposition desc&" \
                  f"format=json"
    response = credentials.request(request_uri)
    return response


def get_latest_with_cloud_limit(entries, cloudcoverpercentage_limit = 10.0):
    cloud_cover_percentage_variable_name = "cloudcoverpercentage"
    for entry in entries:
        if type(entry['double']) is list:
            for item in entry['double']:
                if item['name'] == cloud_cover_percentage_variable_name:
                    if float(item['content']) <= cloudcoverpercentage_limit:
                        return entry
        elif type(entry['double']) is dict:
            if entry['double']['name'] == cloud_cover_percentage_variable_name:
                if float(entry['double']['content']) < cloudcoverpercentage_limit:
                    return entry


def get_entries(response):
    return response.json()['feed']['entry']


def get_data_link(entry):
    return entry['link'][0]['href']


def get_data_name(entry):
    return entry['title']


def print_data_summary(entry):
    print(entry['summary'])


def get_data_size(entry):
    for item in entry['str']:
        if item['name'] == 'size':
            return item['content']


class DownloadWrapper:
    def __init__(self, entry):
        self.download_link = get_data_link(entry)
        self.dataset_name = get_data_name(entry)
        self.filesize = size_to_float(get_data_size(entry))

        self.file_path = path.join(DATA_PATH, self.dataset_name + ARCHIVE_EXT)
        self.lock = Lock()
        self.progress = 0.0
        self.chunk_size = 1024

    def _download(self):
        with credentials.request(self.download_link, stream=True) as r:
            with open(self.file_path, 'wb') as f:
                chunk_id = 0
                for chunk in r.iter_content(chunk_size=self.chunk_size):
                    size_downloaded = chunk_id * self.chunk_size
                    with self.lock:
                        self.progress = size_downloaded / self.filesize
                    if chunk:
                        f.write(chunk)
                    chunk_id += 1

    def start_download(self, overwrite=False):
        file_exists = path.isfile(self.file_path)
        if overwrite or not file_exists:
            thread = Thread(target=self._download)
            thread.daemon = True
            thread.start()
        else:
            with self.lock:
                self.progress = 1

    def get_progress(self):
        with self.lock:
            return self.progress


def download_file(url, filename="temp.zip", filesize=None):
    local_filename = path.join(DATA_PATH, filename+ARCHIVE_EXT)
    # NOTE the stream=True parameter
    print("Downloading file...")
    with credentials.request(url, stream=True) as r:
        with open(local_filename, 'wb') as f:
            chunk_id = 0
            for chunk in r.iter_content(chunk_size=1024):
                if chunk_id%888==0:
                    stdout.write(f"\rDownloaded {format_size(chunk_id*1024)}")
                    if filesize is not None:
                        stdout.write(f" of {filesize}")
                    else:
                        stdout.write(f"...")
                    stdout.write(10*" ")
                    stdout.flush()
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    #f.flush() commented by recommendation from J.F.Sebastian
                chunk_id += 1

    stdout.write(f"\rDownloaded {format_size(chunk_id*1024)}")
    if filesize is not None:
        stdout.write(f" of {filesize}")
    else:
        stdout.write(f"...")
    stdout.write("\n")
    print("Done!")
    return local_filename


def quicklook(entry):
    for item in entry['link']:
        if item.get('rel', None) == 'icon':
            data = credentials.request(item['href'])
            img = Image.open(BytesIO(data.content))
            return img


def download(entry, overwrite=False):
    download_link = get_data_link(entry)
    dataset_name = get_data_name(entry)
    filesize = get_data_size(entry)

    file_path = path.join(DATA_PATH, dataset_name+ARCHIVE_EXT)

    file_exists = path.isfile(file_path)

    if overwrite or not file_exists:
        download_file(
                download_link,
                dataset_name,
                filesize)
    else:
        print("File exists, skipping download...")

    return dataset_name

def format_size(size):
    #2**10 = 1024
    power = 2**10
    n = 0
    Dic_powerN = {0 : '', 1: 'k', 2: 'M', 3: 'G', 4: 'T'}
    while size > power:
        size /=  power
        n += 1
    return f"{size:.2f} {Dic_powerN[n]}B"


def size_to_float(size_str):
    #2**10 = 1024
    power = 2**10
    n = 0
    Dic_powerN = {'kB': 1, 'MB': 2, 'GB': 3, 'TB': 4}
    for key in Dic_powerN:
        if key in size_str:
            value = float(size_str.split(' ')[0])
            return value*(power**Dic_powerN[key])

    return float(size_str.split(' ')[0])


if __name__ == "__main__":
    response = get_latest(SAMPLE_LOCATION)
    entry = response.json()['feed']['entry'][0]
    dataset_name = download(entry)
