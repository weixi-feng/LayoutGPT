import json
import pdb

import numpy as np


def load_json(fname):
    with open(fname, "r") as file:
        data = json.load(file)
    return data


def write_json(fname, data):
    with open(fname, "w") as file:
        json.dump(data, file, indent=4, separators=(",",":"), sort_keys=True)


def bb_relative_position(boxA, boxB):
    xA_c = (boxA[0]+boxA[2])/2
    yA_c = (boxA[1]+boxA[3])/2
    xB_c = (boxB[0]+boxB[2])/2
    yB_c = (boxB[1]+boxB[3])/2
    dist = np.sqrt((xA_c - xB_c)**2 + (yA_c - yB_c)**2)
    cosAB = (xA_c-xB_c) / dist
    sinAB = (yB_c-yA_c) / dist
    return cosAB, sinAB
    

def eval_spatial_relation(bbox1, bbox2):
    theta = np.sqrt(2)/2
    relation = 'diagonal'

    if bbox1 == bbox2:
        return relation
    
    cosine, sine = bb_relative_position(bbox1, bbox2)

    if cosine > theta:
        relation = 'right'
    elif sine > theta:
        relation = 'top'
    elif cosine < -theta:
        relation = 'left'
    elif sine < -theta:
        relation = 'bottom'
    
    return relation


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou