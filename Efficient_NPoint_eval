import json
import os
import time
import sys
sys.path.append("..")
import matplotlib
import numpy as np
import pycocotools.mask as mask_util
import torch
from numpy import random
from torchvision.transforms import ToTensor
from PIL import Image
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits

 # ------------------------------ Setting ------------------------------
ANNOTATION_PATH = r"/root/segment-anything-2/data/UIIS/UDW/annotations/val.json"
IMAGE_PATH = r"/root/segment-anything-2/data/UIIS/UDW/val"
scale_to_original_image_size = True
N_Point = 3
# ------------------------------ Setting ------------------------------

if __name__ == '__main__':
    matplotlib.use('Agg')
    dataset_results, all_time = [], 0
    # ------------------------ Load EfficientSAM model -----------------

    efficient_sam = build_efficient_sam_vitt().cuda()

    # ------------------------ Load COCO dataset -----------------------

    cocoGT = COCO(ANNOTATION_PATH)
    categories = cocoGT.dataset['categories']
    classes = dict([(ann["id"], ann["name"]) for ann in categories])
    print("categories:", classes)

    # ------------------------ Evaluate EfficientSAM model --------------

    for img_id in tqdm(cocoGT.getImgIds()):
        img_dict = cocoGT.loadImgs(img_id)[0]
        file_name, height, width = img_dict["file_name"], img_dict["height"], img_dict["width"]
        # print("file_name:", file_name, "height:", height, "width:", width)
        image_path = os.path.join(IMAGE_PATH, file_name)

        # ------------------ Use EfficientSAM to predict masks ----------
        ann_ids = cocoGT.getAnnIds(imgIds=img_dict['id'], iscrowd=None)
        anns = cocoGT.loadAnns(ann_ids)

        start = time.perf_counter()  # efficient sam encoder time begins
        image_np = np.array(Image.open(image_path))
        img_tensor = ToTensor()(image_np)[None, ...].cuda()
        batch_size, _, input_h, input_w = img_tensor.shape
        image_embeddings = efficient_sam.get_image_embeddings(img_tensor)
        end = time.perf_counter()
        all_time += end - start  # efficient sam encoder time ends


        for ann in anns:
            x, y, w, h = ann['bbox']
            input_point = [[x + w / 2, y + h / 2]]
            if N_Point > 1:
                bound_point = ann['segmentation'][0]
                x_list, y_list = np.array(bound_point[::2]), np.array(bound_point[1::2])
                shuffle_index = random.randint(len(x_list), size=(N_Point - 1))
                input_point += [*zip(x_list[shuffle_index], y_list[shuffle_index])]
            pts_sampled = np.array(input_point)
            pts_labels = np.array([1] * N_Point)
            pts_sampled = torch.reshape(torch.tensor(pts_sampled), [1, 1, -1, 2]).cuda()
            pts_labels = torch.reshape(torch.tensor(pts_labels), [1, 1, -1]).cuda()

            start = time.perf_counter()  # prompt encoder and mask decoder time begins
            predicted_logits, predicted_iou = efficient_sam.predict_masks(
                image_embeddings,
                pts_sampled,
                pts_labels,
                multimask_output=True,
                input_h=input_h,
                input_w=input_w,
                output_h=input_h if scale_to_original_image_size else -1,
                output_w=input_w if scale_to_original_image_size else -1,
            )
            end = time.perf_counter()
            all_time += end - start  # prompt encoder and mask decoder time ends

            sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
            predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
            predicted_logits = torch.take_along_dim(
                predicted_logits, sorted_ids[..., None, None], dim=2
            )
            mask = torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()
            iou = predicted_iou[0, 0, 0].cpu().detach().numpy()

            rle = mask_util.encode(np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F"))[0]
            rle['counts'] = rle["counts"].decode("utf-8")
            dataset_results.append({
                'image_id': ann['image_id'], 'category_id': ann['category_id'],
                'segmentation': rle, "score": float(iou)
            })

    # ------------------------- Save the results -------------------------
    with open("EfficientSAM_bbox_test.json", "w") as f:
        json.dump(dataset_results, f)

    # ------------------------- Evaluate the results ---------------------
    cocoDt = cocoGT.loadRes("EfficientSAM_bbox_test.json")
    cocoEval = COCOeval(cocoGT, cocoDt, "segm")
    cocoEval.evaluate()
    cocoEval.accumulate()
    print("Average time per image (FPS) is:", len(cocoGT.getImgIds()) / all_time)
    cocoEval.summarize()
