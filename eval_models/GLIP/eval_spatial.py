import matplotlib.pyplot as plt

import requests
from io import BytesIO
from PIL import Image
import numpy as np
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

import os
from collections import defaultdict
import json
from tqdm import tqdm
import sys
import pdb

sys.path.append("../../") 
from utils import *

def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def load(dir):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    pil_image = Image.open(dir).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img, caption):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)


def list_grid(list1, list2):
    return [(l1, l2) for l1 in list1 for l2 in list2]


def eval_image(bbox_by_entities, img_gt):
    bbox1 = [coord for _, coord in bbox_by_entities[img_gt['obj1'][0]]]
    bbox2 = [coord for _, coord in bbox_by_entities[img_gt['obj2'][0]]]
    gt_rel = img_gt['relation']
    all_relations = []

    if len(bbox1) != 0 and len(bbox2) != 0:
        for b1, b2 in list_grid(bbox1, bbox2):
            relation = eval_spatial_relation(b1, b2)
            all_relations.append(relation)
        if gt_rel in all_relations:
            return True
        if gt_rel == 'next to' and ('left' in all_relations or 'right' in all_relations):
            return True
        return False
    return False


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-d", "--dir", type=str)
    parser.add_argument("-t", "--thresh", type=float, default=0.7)
    parser.add_argument("--annotations", type=str, default="../../dataset/NSR-1K/spatial/spatial.val.json")
    parser.add_argument("--output_dir", type=str, default="spatial")
    args = parser.parse_args()

    with open(args.annotations, "r") as file:
        gt = json.load(file)
    gt = {d['id']: d for d in gt}
    
    folder = args.output_dir
    os.makedirs(f"outputs/{folder}/{args.thresh}", exist_ok=True)
    result_file = f"outputs/{folder}/GLIP{args.thresh}_results.json"
    image_names = sorted(os.listdir(args.dir))

    if not os.path.exists(result_file):
        # start loading GLIP
        config_file = "configs/pretrain/glip_Swin_L.yaml"
        weight_file = "MODEL/glip_large_model.pth"

        # update the config options with the config file
        # manual override some options
        cfg.local_rank = 0
        cfg.num_gpus = 1
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
        cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

        glip_demo = GLIPDemo(
            cfg,
            min_image_size=800,
            confidence_threshold=0.7,
            show_mask_heatmaps=False
        )

        plus = 1 if glip_demo.cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD" else 0

        grounding_results = {}
        blockPrint()
        n_correct = 0
        for file in tqdm(image_names):
            image = load(os.path.join(args.dir, file))
            image_id, n_iter = [int(x) for x in os.path.splitext(file)[0].split("_")]

            caption = f"{gt[image_id]['obj1'][0]}, {gt[image_id]['obj2'][0]}"

            result, top_predictions = glip_demo.run_on_web_image(image, caption, args.thresh)
            fig = plt.figure(figsize=(5,5))
            plt.imshow(result[:, :, [2, 1, 0]])
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(f"outputs/{folder}/{args.thresh}/{file}")
            plt.close()

            scores = top_predictions.get_field("scores")
            labels = top_predictions.get_field("labels")
            bbox = top_predictions.bbox
            entities = glip_demo.entities
            
            new_labels = []
            for i in labels:
                if i <= len(entities):
                    new_labels.append(entities[i-plus])
                else:
                    new_labels.append("object")

            bbox_by_entities = defaultdict(list)
            for l, score, coord in zip(new_labels, scores, bbox):
                bbox_by_entities[l.strip()].append((score.item(), coord.tolist()))
            grounding_results[file] = bbox_by_entities

            n_correct += 1 if eval_image(bbox_by_entities, gt[image_id]) else 0

        with open(result_file, "w") as file:
            json.dump(grounding_results, file, indent=4, separators=(",",":"), sort_keys=True)

    else:
        n_correct = 0
        grounding_results = json.load(open(result_file, "r"))

        for file in tqdm(image_names):
            image_id, n_iter = [int(x) for x in os.path.splitext(file)[0].split("_")]
            bbox_by_entities = grounding_results[file]
            n_correct += 1 if eval_image(bbox_by_entities, gt[image_id]) else 0

    enablePrint()
    print(folder, " Spatial Accuracy: {:.04f}".format(n_correct / len(image_names)))

    