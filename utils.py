import json
import numpy as np
import os
import argparse
import re
import cssutils
from tqdm import tqdm
from string import digits

import pdb

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


def parse_layout(string, canvas_size=64, no_integer=False):
    idx = string.find(' {')
    category = re.sub(r'[0-9]', '', string[:idx].replace(' ', '-'))
    string = category + string[idx:]
    try:
        sheet = cssutils.parseString(string)
        rule = sheet.cssRules[0]
        bbox = [b.strip().strip(";").strip() for b in rule.style.cssText.split("\n")]
        text = rule.selectorText
    except:
        try:
            text, bbox = string.split("{")
            bbox = bbox.strip().strip("}").strip().strip(";").split(";")
            assert len(bbox) == 4
            bbox = [b.strip().strip(";").strip() for b in bbox]
        except:
            return None, None

    category = text.strip()
    parsed_category = category.translate(category.maketrans('', '', digits)).strip()

    bbox = [b.split(":") for b in bbox]
    if no_integer:
        bbox = {k.strip():float(v.lstrip().rstrip()) for k, v in bbox}
    else:
        bbox = {k.strip():int(v.lstrip().rstrip("px")) for k, v in bbox}

    if sorted(bbox.keys()) != ['height', 'left', 'top', 'width']:
        print(string)
        bbox = [0] * 4
        return parsed_category, bbox

    bbox = [bbox['left'], bbox['top'], min(bbox['left']+bbox['width'], canvas_size), min(bbox['top']+bbox['height'], canvas_size)]
    if bbox[0] >= canvas_size or bbox[1] >= canvas_size:
        return None, None
    bbox = [float(b)/canvas_size for b in bbox]

    return parsed_category, bbox


def parse_3D_layout(string, unit='m'):
    # cannot use cssutils due to self-defined properties
    try:
        text, bbox = string.split("{")
        bbox = bbox.strip().strip("}").strip().strip(";").split(";")
        assert len(bbox) == 7
        bbox = [b.strip().strip(";").rstrip("degrees").strip() for b in bbox]
    except:
        return None, None

    category = text.strip()
    parsed_category = category.translate(category.maketrans('', '', digits)).strip()

    bbox = [b.split(":") for b in bbox]

    bbox = {k.strip():float(v.lstrip().rstrip(unit)) for k, v in bbox}
    if sorted(bbox.keys()) != sorted(['height', 'width', 'length', 'orientation', 'left', 'top', 'depth']):
        print(string)
        bbox = {k:0 for k in bbox.keys()}
        return parsed_category, bbox

    return parsed_category, bbox



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--files", nargs="+")
    args = parser.parse_args()
    
    for fname in args.files:
        basename = os.path.basename(fname)
        dirname = os.path.dirname(fname)

        assert "raw" not in basename
        response = load_json(fname)

        print(f"Parsing {basename}")
        for r in tqdm(response):
            layout = r['text'].strip().strip("\n").strip().split("\n")
            assert len(layout) >= 2
            r['objects'] = []

            for elm in layout:
                selector_text, bbox = parse_layout(elm)
                if selector_text == None:
                    continue
                if sum(bbox) == 0:
                    print("Failed")
                r['objects'].append([selector_text, bbox])
        
        write_json(os.path.join(dirname, "parsed_"+basename), response)
                
