import credentials
from os import path
from sys import stdout

SAMPLE_LOCATION = (50.083333, 14.416667) # Prague

MASTER_URI = "https://scihub.copernicus.eu/dhus/search?"

PLATFORM_NAME = "Sentinel-2"
PRODUCT_TYPE = "S2MSI1C"

DATA_PATH = "data"
ARCHIVE_EXT = ".zip"


def request_sentinel_2_data(location):
    request_uri = f"{MASTER_URI}start=0&rows=10&" \
                  f"q=footprint:\"Intersects({location[0]}, {location[1]})\" AND " \
                  f"platformname:{PLATFORM_NAME} AND " \
                  f"producttype: {PRODUCT_TYPE}&" \
                  f"orderby=beginposition desc&" \
                  f"format=json"
    response = credentials.request(request_uri)
    return response


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


if __name__ == "__main__":
    response = request_sentinel_2_data(SAMPLE_LOCATION)
    entry = response.json()['feed']['entry'][0]
    dataset_name = download(entry)
