import argparse


DATA_PATH_DEFAULT = "data"
IMAGE_PATH_DEFAULT = "saved_images"
QUALITY_LOW = 'low'
QUALITY_HIGH = 'high'
QUALITY_DEFAULT = QUALITY_LOW
CLOUD_LIMIT_DEFAULT = 5

def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="Sentinel 2 acquisition checker",
        description="This script allows you to quickly check "
                    "when either one of the Sentinel 2 satellites "
                    "will be taking pictures on a given location. "
                    "Information is extracted directly from "
                    "Sentinel 2 acquisition plans, available at "
                    "https://sentinel.esa.int "
                    "Sample location: 50.083333 14.416667 (Prague)",
        usage=f"python {__file__} --location lat lon",
    )

    parser.add_argument('--location', '-l', nargs=2, type=float, required=True)

    parser.add_argument('--data-path', '-d',
                        nargs=1, type=str,
                        default=DATA_PATH_DEFAULT,
                        help="Path to where the datasets "
                             f"are stored (default: {DATA_PATH_DEFAULT})")

    parser.add_argument('--image-path', '-i',
                        nargs=1, type=str,
                        default=IMAGE_PATH_DEFAULT,
                        help="Path to where the images "
                             f"are to be stored (default: {IMAGE_PATH_DEFAULT})")

    parser.add_argument('--image-quality', '-q',
                        nargs=1,
                        default=QUALITY_DEFAULT,
                        choices=(QUALITY_LOW, QUALITY_HIGH),
                        help=f"Saved image quality (default: {QUALITY_DEFAULT})")

    parser.add_argument('--cloud-limit', '-c',
                        nargs=1, type=float,
                        default=CLOUD_LIMIT_DEFAULT,
                        help="Downloaded datasets are filtered by amount "
                             "of clouds in the image. Only images with "
                             "amount of clouds less than this value %% "
                             f"are accepted (default: {CLOUD_LIMIT_DEFAULT})")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    import sentinel_requests
    import images
    import os
    from matplotlib import pyplot as plt

    location = args.location
    data_path = args.data_path
    image_path = args.image_path
    image_quality = args.image_quality[0]
    cloud_limit = args.cloud_limit

    ARCHIVE_EXT = ".zip"

    print("Forming request...")
    response = sentinel_requests.get_latest(location)

    if response.status_code == 200:
        print("Request successful.")

    entries = response.json()['feed']['entry']

    entry = sentinel_requests.get_latest_with_cloud_limit(entries, cloud_limit)

    print(f"Latest image from: {entry['date'][0]['content']}")

    dataset_name = sentinel_requests.download(entry)

    archive_path = os.path.join(data_path,
                                dataset_name + ARCHIVE_EXT)

    image = images.get_tci_image(archive_path)

    print("Plotting...")
    if image_quality == QUALITY_LOW:
        images.plot_and_save(image, dataset_name)
    elif image_quality == QUALITY_HIGH:
        plt.imsave(os.path.join(image_path, f"{dataset_name}.png"),
                   image,
                   format='png')