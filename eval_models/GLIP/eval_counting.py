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


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-d", "--dir", type=str)
    parser.add_argument("-t", "--thresh", type=float, default=0.7)
    parser.add_argument("--annotations", type=str, default="../../dataset/NSR-1K/counting/counting.val.json")
    parser.add_argument("--output_dir", type=str, default="counting")
    args = parser.parse_args()

    # get list of eval images
    image_names = sorted(os.listdir(args.dir))

    # ground truth annotations
    with open(args.annotations, "r") as file:
        gt = json.load(file)
    gt = {d['id']:d for d in gt}

    # prepare dir for detection results
    folder = args.output_dir
    save_dir = os.path.join("outputs", folder, str(args.thresh))
    os.makedirs(save_dir, exist_ok=True)
    result_file = f"{os.path.dirname(save_dir)}/GLIP{args.thresh}_results.json"
    result_exists = os.path.exists(result_file)
    if result_exists:
        detection_results = json.load(open(result_file))

    # GLIP config
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
    # blockPrint()
    n_correct = 0
    n_instances = 0

    for file in tqdm(image_names):
        image_id, n_iter = [int(x) for x in os.path.splitext(file)[0].split("_")]

        num_objects = {obj:n for obj, n in gt[image_id]['num_object']}
        n_instances += len(num_objects)
        caption = ", ".join(list(num_objects.keys()))

        if result_exists:
            bbox_by_entities = detection_results[file]
        else:
            image = load(os.path.join(args.dir, file))
            result, top_predictions = glip_demo.run_on_web_image(image, caption, args.thresh)
            fig = plt.figure(figsize=(5,5))
            plt.imshow(result[:, :, [2, 1, 0]])
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{file}")
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
        
        if gt[image_id]['sub-type'] == 'comparison':
            anchor_obj, anchor_cnt = gt[image_id]['num_object'][0]
            if len(bbox_by_entities[anchor_obj]) == anchor_cnt:
                if gt[image_id]['num_object'][1][1] == anchor_cnt:
                    if len(bbox_by_entities[gt[image_id]['num_object'][1][0]]) == anchor_cnt:
                        n_correct += 1
                elif gt[image_id]['num_object'][1][1] != anchor_cnt:
                    if (len(bbox_by_entities[gt[image_id]['num_object'][1][0]]) - anchor_cnt) * (gt[image_id]['num_object'][1][1] - anchor_cnt) > 0:
                        n_correct += 1
        else:
            for obj in num_objects.keys():
                if num_objects[obj] == len(bbox_by_entities[obj]):
                    n_correct += 1
        
    enablePrint()
    print("Counting Accuracy: {:.04f}".format(n_correct / n_instances))
    
    if not result_exists:
        with open(result_file, "w") as file:
            json.dump(grounding_results, file, indent=4, separators=(",",":"), sort_keys=True)