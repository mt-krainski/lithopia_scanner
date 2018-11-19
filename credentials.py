import json
import requests
import getpass

SAMPLE_REQUEST = 'https://scihub.copernicus.eu/dhus/search?' \
             'q=footprint:"Intersects(41.9000, 12.5000)"'

CREDENTIALS_FILE = ".credentials.json"

def get_credentials():
    credentials = {}

    try:
        with open(CREDENTIALS_FILE) as cred_file:
            credentials = json.load(cred_file)
    except (FileNotFoundError, PermissionError):
        print("File error! Have you created the "
              "credentials by running credentials.py?")

    return credentials


def store_credentials(username, password):
    credential_dict = {
        "username" : username,
        "password" : password
    }

    with open(CREDENTIALS_FILE, "w") as cred_file:
        json.dump(credential_dict, cred_file)


def test_credentials(username, password):
    res = requests.get(
        SAMPLE_REQUEST,
        auth=(username,
              password))

    if res.status_code != 200:
        return True
    else:
        return False


def request(uri, **kwargs):
    credentials = get_credentials()
    return requests.get(
        uri,
        auth=(credentials["username"],
              credentials["password"]),
        **kwargs)


if __name__ == "__main__":

    import argparse

    argument_parser = argparse.ArgumentParser(
        prog='Update credentials',
        description='Script to update credentials used for '
                    'downloading data from the '
                    'Copernicus Open Access Hub')

    argument_parser.add_argument(
        '--username', '-u',
        help="Your username"
    )

    argument_parser.add_argument(
        '--password', '-p',
        help="Your password"
    )

    credentials = argument_parser.parse_args()

    if credentials.username is None:
        username = input("Username: ")
    else:
        username = credentials.username

    if username == "":
        print("Invalid username! Error...")
        exit()

    if credentials.password is None:
        password = getpass.getpass("Password: ")
    else:
        password = credentials.password

    if password == "":
        print("Invalid password! Error...")
        exit()

    if not test_credentials(username, password):
        print("Request failed! Credentials invalid...")
        exit()

    store_credentials(username, password)
