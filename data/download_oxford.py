"""
Dataset ingestion script for the Oxford 102 Flowers Dataset.
Pipeline Stage: 1 (Data Ingestion)
"""
import os
import tarfile
import urllib.request
import logging

# Set up professional logging for the ML pipeline
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define relative paths so this works on Colab, your local PC, or a university server
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'raw')
EXTRACT_DIR = os.path.join(RAW_DATA_DIR, 'oxford_102')

# Oxford 102 Flowers Official VGG URL
DATASET_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
ARCHIVE_PATH = os.path.join(RAW_DATA_DIR, "102flowers.tgz")

def download_and_extract():
    """Downloads and extracts the dataset if it doesn't already exist."""
    
    # Step 1: Ensure the target directories exist
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    
    # Step 2: Download the dataset
    if not os.path.exists(ARCHIVE_PATH):
        logging.info(f"Downloading Oxford 102 Flowers from {DATASET_URL}...")
        urllib.request.urlretrieve(DATASET_URL, ARCHIVE_PATH)
        logging.info("Download complete!")
    else:
        logging.info(f"Archive already exists at {ARCHIVE_PATH}. Skipping download.")

    # Step 3: Extract the dataset
    # The archive contains a single folder named 'jpg' full of the images
    expected_img_dir = os.path.join(EXTRACT_DIR, "jpg")
    if not os.path.exists(expected_img_dir):
        logging.info(f"Extracting {ARCHIVE_PATH} to {EXTRACT_DIR}...")
        with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
            tar.extractall(path=EXTRACT_DIR)
        logging.info("Extraction complete! Dataset is ready for the pipeline.")
    else:
        logging.info("Dataset already extracted and verified. Ready to use.")

if __name__ == "__main__":
    download_and_extract()
