import shutil
import json
from tqdm import tqdm
import numpy as np


def split_val_test_set(
    annotations_file="/opt/cocoapi/annotations/captions_val2014.json.old",
    images_dir="/opt/cocoapi/images/val2014.old/",
    seed=2137,
):
    """Split validation dataset in half to create validation and test datasets.

        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        !!! RUN THIS FUNCTION DIRECTLY AS A ROOT !!!
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    Args:
        annotations_file (str, optional): [absolute path to validation captions]. Defaults to "/opt/cocoapi/annotations/captions_val2014.json.old".
        images_dir (str, optional): [absolute path to validation images directory]. Defaults to "/opt/cocoapi/images/val2014.old/".
        seed (int, optional): [seed for shuffling]. Defaults to 2137.
    """

    # get chunks
    validations = json.loads(open(annotations_file).read())
    validations_info = validations["info"]
    validations_images = validations["images"]
    validations_licenses = validations["licenses"]
    validations_annotations = validations["annotations"]

    # split validation in half to create test dataset
    pics_ids = [i["id"] for i in validations_images]
    split = len(validations_images) // 2

    # shuffle ids and prepare filnames
    np.random.seed(seed)
    np.random.shuffle(pics_ids)
    val_ids = pics_ids[:split]
    test_ids = pics_ids[split:]
    val_files = [f"COCO_val2014_{str(i).zfill(12)}.jpg" for i in val_ids]
    test_files = [f"COCO_val2014_{str(i).zfill(12)}.jpg" for i in test_ids]

    ## ANNOTATIONS
    # split to new validation and test
    validation = {
        "info": validations_info,
        "images": [i for i in validations_images if i["id"] in val_ids],
        "licenses": validations_licenses,
        "annotations": [i for i in validations_annotations if i["image_id"] in val_ids],
    }
    test = {
        "info": validations_info,
        "images": [i for i in validations_images if i["id"] in test_ids],
        "licenses": validations_licenses,
        "annotations": [
            i for i in validations_annotations if i["image_id"] in test_ids
        ],
    }
    # save new annotation files
    with open("captions_val2014.json", "w") as f:
        json.dump(validation, f)
    shutil.os.system("mv captions_val2014.json /opt/cocoapi/annotations")

    with open("captions_test2014.json", "w") as f:
        json.dump(test, f)
    shutil.os.system("mv captions_test2014.json /opt/cocoapi/annotations")

    ## IMAGES
    # make new folders and move files
    shutil.os.system("mkdir /opt/cocoapi/images/val2014")
    for file in tqdm(val_files):
        shutil.os.system(f"cp {images_dir+file} /opt/cocoapi/images/val2014")

    shutil.os.system("mkdir /opt/cocoapi/images/test2014")
    for file in tqdm(test_files):
        shutil.os.system(f"cp {images_dir+file} /opt/cocoapi/images/test2014")

    ## REMOVE redundant files
    shutil.os.system(f"rm -rf {images_dir}")
    shutil.os.system(f"rm -f {annotations_file}")

    print("All done.")


if __name__ == "__main__":
    split_val_test_set()
