import os
import gdown
import zipfile
import logging

def check_dir(dir_name: str) -> bool:
    return os.path.isdir(dir_name)

def download_data(dir_name: str = "data") -> None:
    if not check_dir(dir_name):
        os.mkdir(dir_name)

    os.chdir(dir_name)
    logging.info("Downloading data....")
    gdown.download(
        "https://drive.google.com/uc?id=1HWFhyaR9a3HtjIqG2WSItl6vqOFtwpy6", quiet=False
    )
    logging.info("Extracting zip file....")
    with zipfile.ZipFile("UTKFace.zip", 'r') as zip_ref:
        zip_ref.extractall("")
    os.remove("UTKFace.zip")
    os.chdir("..")