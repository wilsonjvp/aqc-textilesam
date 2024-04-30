from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import os

import pandas as pd
import json
import boto3
import tqdm
import csv
import argparse
import numpy as np

from sqlalchemy import create_engine
from sqlalchemy import URL
from dotenv import load_dotenv

AWS_BUCKET = "aqc-lambda-tiles-prod"
OUTPUT_DIR = "data/train"
OUTPUT_DIR_VIS = "evaluation/visible/test/good"
OUTPUT_DIR_IR = "evaluation/infrared/test/good"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_VIS, exist_ok=True)
os.makedirs(OUTPUT_DIR_IR, exist_ok=True)


def download_one_file(bucket: str, output: str, client: boto3.client, s3_file: str):
    """
    Download a single file from S3
    Args:
        bucket (str): S3 bucket where images are hosted
        output (str): Dir to store the images
        client (boto3.client): S3 client
        s3_file (str): S3 object name
    """
    client.download_file(
        Bucket=bucket,
        Key=s3_file,
        Filename=os.path.join(output, s3_file.split("/")[-1]),
    )


def get_files_to_download(coco_file_dir):
    path_to_annotation = coco_file_dir
    with open(path_to_annotation, "r") as j:
        data = json.load(j)

    df = pd.DataFrame(data["images"])
    files = df["coco_url"].values
    return files


def parse_user_arguments():
    """Parse user arguments for the evaluation of a method on the MVTec AD
    dataset.

    Returns:
        Parsed user arguments.
    """
    parser = argparse.ArgumentParser(description="""Parse user arguments.""")

    parser.add_argument('--coco_file_dir',
                        default="data/annotations_qualitex_reviewed_22_03_2024.json",
                        help="""Path to the json file containing the annotations in COCO format""")


    args = parser.parse_args()

    return args

def download_db():
    # Load DB variables
    load_dotenv()
    DATABASE = os.getenv("PROD_DATABASE")
    USER = os.getenv("PROD_USER")
    PASSWORD = os.getenv("PROD_PASSWORD")
    HOST = os.getenv("PROD_HOST")
    PORT = os.getenv("PROD_PORT")

    # Connection to DB
    url_object = URL.create(
                "postgresql+psycopg",
                username=USER,
                password=PASSWORD,
                host=HOST,
                port=PORT,
                database=DATABASE,
            )

    alchemyEngine = create_engine(url_object)

    dbConnection = alchemyEngine.connect()

    # Reading Tables from DB
    df_tiles = pd.read_sql('select * from "Tiles"', dbConnection)
    df_rolls = pd.read_sql('select * from "Rolls"', dbConnection)

    # Close DB connection
    dbConnection.close()    

    return df_tiles, df_rolls

def download_good_tiles(args):
    # Extract file names from COCO file
    coco_files = get_files_to_download(args.coco_file_dir)
    machine_roll_counters = np.unique([x.split("/")[2] for x in coco_files])

    # Download tiles table from DB
    df_tiles, df_rolls = download_db()

    # Counting total training number of images per roll id
    total_count = {}
    print("total coco files", len(coco_files))
    for counter in machine_roll_counters:
        image_count = len([1 for x in coco_files if counter + "/" in x])
        roll_id = df_tiles[df_tiles["AbsVisImageFileName"].str.contains(counter + "/")]["RollId"].values[0]
        d_number = df_rolls[df_rolls["Id"] == roll_id]["DispositionNumber"].values[0]
        d_row = df_rolls[df_rolls["Id"] == roll_id]["DispositionRowNumber"].values[0]
        disposition = f"{d_number}_{d_row}"

        if disposition not in total_count.keys():
            total_count[disposition] = image_count
        else:
            total_count[disposition] += image_count
    
    files_to_download_visible = []
    files_to_download_infrared = []
    for disposition in total_count.keys():
        max_count = total_count[disposition]
        d_number = int(disposition.split("_")[0])
        d_row = int(disposition.split("_")[1])
        roll_ids = df_rolls[(df_rolls["DispositionNumber"] == d_number)
        & (df_rolls["DispositionRowNumber"] == d_row)]["Id"].values

        tiles = df_tiles[(df_tiles["RollId"].isin(roll_ids)) &
        (df_tiles["Tags"].apply(lambda x: "training" in x))].reset_index(drop=True)

        count = 0
        for i in range(len(tiles)):
            if count < max_count:
                files_to_download_visible.append(tiles["AbsVisImageFileName"].values[i])
                files_to_download_infrared.append(tiles["AbsIrImageFileName"].values[i])
                count += 2

    # Creating only one session and one client
    session = boto3.Session()
    client = session.client("s3")
    # The client is shared between threads
    func_vis = partial(download_one_file, AWS_BUCKET, OUTPUT_DIR_VIS, client)
    func_ir = partial(download_one_file, AWS_BUCKET, OUTPUT_DIR_IR, client)

    # List for storing possible failed downloads to retry later
    failed_downloads = []

    with tqdm.tqdm(
        desc="Downloading visible evaluation images from S3", total=len(files_to_download_visible)
    ) as pbar:
        with ThreadPoolExecutor(max_workers=32) as executor:
            # Using a dict for preserving the downloaded file for each future, to store it as a failure if we need that
            futures = {
                executor.submit(func_vis, file_to_download): file_to_download
                for file_to_download in files_to_download_visible
            }
            for future in as_completed(futures):
                if future.exception():
                    failed_downloads.append(futures[future])
                pbar.update(1)
    if len(failed_downloads) > 0:
        print("Some downloads have failed. Saving ids to csv")
        with open(
            os.path.join(OUTPUT_DIR_VIS, "failed_downloads.csv"), "w", newline=""
        ) as csvfile:
            wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            wr.writerow(failed_downloads)
    
    with tqdm.tqdm(
        desc="Downloading infrared evaluation images from S3", total=len(files_to_download_infrared)
    ) as pbar:
        with ThreadPoolExecutor(max_workers=32) as executor:
            # Using a dict for preserving the downloaded file for each future, to store it as a failure if we need that
            futures = {
                executor.submit(func_ir, file_to_download): file_to_download
                for file_to_download in files_to_download_infrared
            }
            for future in as_completed(futures):
                if future.exception():
                    failed_downloads.append(futures[future])
                pbar.update(1)
    if len(failed_downloads) > 0:
        print("Some downloads have failed. Saving ids to csv")
        with open(
            os.path.join(OUTPUT_DIR_VIS, "failed_downloads.csv"), "w", newline=""
        ) as csvfile:
            wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            wr.writerow(failed_downloads)
    



def download_defective_tiles(args):
    files_to_download = get_files_to_download(args.coco_file_dir)
    # Creating only one session and one client
    session = boto3.Session()
    client = session.client("s3")
    # The client is shared between threads
    func = partial(download_one_file, AWS_BUCKET, OUTPUT_DIR, client)

    # List for storing possible failed downloads to retry later
    failed_downloads = []

    with tqdm.tqdm(
        desc="Downloading training images from S3", total=len(files_to_download)
    ) as pbar:
        with ThreadPoolExecutor(max_workers=32) as executor:
            # Using a dict for preserving the downloaded file for each future, to store it as a failure if we need that
            futures = {
                executor.submit(func, file_to_download): file_to_download
                for file_to_download in files_to_download
            }
            for future in as_completed(futures):
                if future.exception():
                    failed_downloads.append(futures[future])
                pbar.update(1)
    if len(failed_downloads) > 0:
        print("Some downloads have failed. Saving ids to csv")
        with open(
            os.path.join(OUTPUT_DIR, "failed_downloads.csv"), "w", newline=""
        ) as csvfile:
            wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            wr.writerow(failed_downloads)

def main():
    # parse arguments
    args = parse_user_arguments()

    # Download good tiles for evaluation purpose
    download_good_tiles(args)

    # Download defective tiles for training
    download_defective_tiles(args)


if __name__ == "__main__":
    main()
