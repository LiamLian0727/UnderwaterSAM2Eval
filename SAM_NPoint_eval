import json
import os
import time

import matplotlib
import numpy as np
import pycocotools.mask as mask_util
import torch

from PIL import Image
from matplotlib import pyplot as plt
from numpy import random
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from segment_anything import sam_model_registry, SamPredictor

# ------------------------------ Setting ------------------------------
ANNOTATION_PATH = r"/root/segment-anything-2/data/USIS10K/multi_class_annotations/multi_class_test_annotations.json"
IMAGE_PATH = r"/root/segment-anything-2/data/USIS10K/test"
sam_checkpoint = "/root/segment-anything-2/checkpoints/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
N_Point = 3
# ------------------------------ Setting ------------------------------

if __name__ == '__main__':
    matplotlib.use('Agg')
    dataset_results, all_time = [], 0
    # ------------------------ Load SAM2 model ------------------------
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)


    # ------------------------ Load COCO dataset -----------------------

    cocoGT = COCO(ANNOTATION_PATH)
    categories = cocoGT.dataset['categories']
    classes = dict([(ann["id"], ann["name"]) for ann in categories])
    print("categories:", classes)

    # ------------------------ Evaluate SAM2 model ----------------------

    for img_id in tqdm(cocoGT.getImgIds()):
        # --------------------- Load and embed image --------------------
        start = time.perf_counter()  # image encoder time begin
        img_dict = cocoGT.loadImgs(img_id)[0]
        file_name, height, width = img_dict["file_name"], img_dict["height"], img_dict["width"]
        image = np.array(Image.open(os.path.join(IMAGE_PATH, file_name)).convert("RGB"))

        # plt.imshow(image)
        # plt.savefig(f"../fig/uiis10k.jpg")
        predictor.set_image(image)
        end = time.perf_counter()
        all_time += end - start  # image encoder time ends
        # print("file_name:", file_name, "Image", image.shape, "height:", height, "width:", width)

        # --------------------- Use SAM2 to predict masks ----------------
        ann_ids = cocoGT.getAnnIds(imgIds=img_dict['id'], iscrowd=None)
        anns = cocoGT.loadAnns(ann_ids)
        for ann in anns:
            x, y, w, h = ann['bbox']
            input_point = [[x + w / 2, y + h / 2]]
            if N_Point > 1:
                bound_point = ann['segmentation'][0]
                x_list, y_list = np.array(bound_point[::2]), np.array(bound_point[1::2])
                shuffle_index = random.randint(len(x_list), size=(N_Point-1))
                input_point += [*zip(x_list[shuffle_index], y_list[shuffle_index])]

            start = time.perf_counter()  # prompt encoder and mask decoder time begins
            masks, scores, logits = predictor.predict(
                point_coords=np.array(input_point),
                point_labels=np.array([1] * N_Point),
                multimask_output=True,
            )
            end = time.perf_counter()
            all_time += end - start  # prompt encoder and mask decoder time ends

            sorted_ind = np.argmax(scores)
            masks, scores = masks[sorted_ind], scores[sorted_ind]
            rle = mask_util.encode(np.array(masks[:, :, np.newaxis], dtype=np.uint8, order="F"))[0]
            rle['counts'] = rle["counts"].decode("utf-8")
            dataset_results.append({
                'image_id': ann['image_id'], 'category_id': ann['category_id'],
                'segmentation': rle, "score": float(scores)
            })

    # ------------------------- Save the results -------------------------
    with open(f"SAM_{N_Point}point_test.json", "w") as f:
        json.dump(dataset_results, f)

    # ------------------------- Evaluate the results ---------------------
    cocoDt = cocoGT.loadRes(f"SAM_{N_Point}point_test.json")
    cocoEval = COCOeval(cocoGT, cocoDt, "segm")
    cocoEval.evaluate()
    cocoEval.accumulate()
    print("Average time per image (FPS) is:", len(cocoGT.getImgIds()) / all_time)
    cocoEval.summarize()
