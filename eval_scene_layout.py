import os
import os.path as op
import sys
import numpy as np
from numpy import *
import json
from tqdm import tqdm
import copy
import argparse


parser = argparse.ArgumentParser("Scene layout evaluation")
parser.add_argument("-f", "--file", type=str)
parser.add_argument("-r", "--room", type=str)
parser.add_argument('--dataset_dir', type=str)
parser.add_argument("--is_atiss", action='store_true')
args = parser.parse_args()

dataset_prefix = args.dataset_dir
with open(os.path.join(dataset_prefix, "dataset_stats.txt"), "r") as fin:
    stats = json.load(fin)
splits = json.load(open(f"./dataset/3D/{args.room}_splits.json", "r"))


def load_room_boxes(prefix, id, stats):
    data = np.load(op.join(prefix, id, 'boxes.npz'))
    x_c, y_c = data['floor_plan_centroid'][0], data['floor_plan_centroid'][2]
    x_offset  = min(data['floor_plan_vertices'][:,0])
    y_offset = min(data['floor_plan_vertices'][:,2])
    room_length = max(data['floor_plan_vertices'][:,0]) - min(data['floor_plan_vertices'][:,0])
    room_width = max(data['floor_plan_vertices'][:,2]) - min(data['floor_plan_vertices'][:,2])    
    vertices = np.stack((data['floor_plan_vertices'][:,0]-x_offset, data['floor_plan_vertices'][:,2]-y_offset), axis=1)
    vertices = np.asarray([list(nxy) for nxy in set(tuple(xy) for xy in vertices)])
    vertices = [f'({v[0]:.2f}, {v[1]:.2f})' for v in vertices]

    objects = []
    for label, size, angle, loc in zip(data['class_labels'], data['sizes'], data['angles'], data['translations']):
        label_idx = np.where(label)[0][0]
        if label_idx >= len(stats['object_types']):
            continue
        cat = stats['object_types'][label_idx]
        length, height, width = size
        orientation = round(angle[0] / 3.1415926 * 180)
        dx,dz,dy = loc
        objects.append([cat, length, width, height, dx+x_c-x_offset, dy+y_c-y_offset, dz, angle])
    return room_length, room_width, objects


def roty(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, s],
                     [-s,c]])


def invalid_object_size(output, stats, normalized=False, pixel=False, margin=0.1):
    invalid_objects = []
    counter = 0
    invalid_scenes = []
    for out in output:
        invalid_scene = False
        rl, rw, _ = load_room_boxes(dataset_prefix, out['query_id'], stats)
        data = np.load(op.join(dataset_prefix, out['query_id'], 'boxes.npz'))
        x_c, y_c = data['floor_plan_centroid'][0], data['floor_plan_centroid'][2]
        
        pred_objects = copy.deepcopy(out['object_list'])
        
        norm = 1
        if pixel:
            norm = min(rl, rw) / 256.
        else:
            if normalized:
                norm = min(rl, rw)
                
        for _, box in pred_objects:
            for k,v in box.items():
                if k == 'orientation': continue
                box[k] = v*norm
        
        for cat, box in pred_objects:
            R = roty(box['orientation']/180*pi)
            box_vertices = np.asarray([[-box['length']/2, -box['width']/2],
                                       [box['length']/2, -box['width']/2],
                                       [-box['length']/2, box['width']/2],
                                       [box['length']/2, box['width']/2]])
            box_vertices = box_vertices@R
            box_vertices += np.asarray([[box['left'], box['top']]])
            
            # if box['left'] + box['length']/2 > rl + 0.1 or box['top'] + box['width']/2 > rw + 0.1 or box['left'] - box['length']/2 < -0.1 or box['top'] - box['width']/2 < -0.1: 
            if max(box_vertices[:, 0]) > rl+margin or min(box_vertices[:, 0]) < -margin or max(box_vertices[:, 1]) > rw+margin or min(box_vertices[:, 1]) < -margin:
                invalid_objects.append([out['query_id'], cat, 
                                        max(box['left'] + box['length']/2 - rl, box['top'] + box['width']/2 - rw,
                                            (box['left'] - box['length']/2)*-1, (box['top'] - box['width']/2)*-1)])
                invalid_scene = True
            counter += 1
        if invalid_scene:
            invalid_scenes.append(out['query_id'])

    return len(invalid_objects) / counter, invalid_scenes, invalid_objects


def categorical_kl(p, q):
        return (p * (np.log(p + 1e-6) - np.log(q + 1e-6))).sum()

def object_category_KL_divergence(output, gt_data, stats):
    all_categories = stats['object_types']
    gt_label_freq = {c: 0 for c in all_categories}
    pred_label_freq = {c: 0 for c in all_categories}

    for d in gt_data.values():
        for obj in d[2]:
            gt_label_freq[obj[0]] += 1
    gt_label_freq = np.asarray([gt_label_freq[k]/sum(list(gt_label_freq.values())) for k in sorted(all_categories)])
    
    for out in output:
        for obj in out['object_list']:
            if obj[0] not in all_categories: continue
            pred_label_freq[obj[0]] += 1
    pred_label_freq = np.asarray([pred_label_freq[k]/sum(list(pred_label_freq.values())) for k in sorted(all_categories)])
    
    kl_div = categorical_kl(gt_label_freq, pred_label_freq)

    return kl_div, gt_label_freq, pred_label_freq, sorted(all_categories)


def postprocess_atiss(output):
    id2full = {x.split("_")[-1]:x for x in splits['test']}
    for o in output:
        o['query_id'] = id2full[o['id']]

    regular_output = []
    for out in output:
        if out['query_id'] in splits['rect_test']:
            data = np.load(os.path.join(dataset_prefix, out['query_id'], 'boxes.npz'))
            x_c, y_c = data['floor_plan_centroid'][0], data['floor_plan_centroid'][2]
            x_offset  = min(data['floor_plan_vertices'][:,0])
            y_offset = min(data['floor_plan_vertices'][:,2])
            for _, o in out['object_list']:
                for k in ['length', 'width', 'height']:
                    o[k] = o[k]*2
                o['left'] = o['left'] + x_c - x_offset
                o['top'] = o['top'] + y_c - y_offset
                o['orientation'] = round((o['orientation']/np.pi) * 180)
            regular_output.append(out)
    return regular_output


if __name__ == '__main__':
    test_ids_regular = splits['rect_test']
    test_data_regular = {id:load_room_boxes(dataset_prefix, id, stats) for id in tqdm(test_ids_regular)}
    output = json.load(open(args.file))
    if args.is_atiss:
        output = postprocess_atiss(output)
        
    invalid_objs, invalid_scenes, laa = invalid_object_size(output, stats, 
                                                        normalized=True, pixel=True, 
                                                        margin=0.1)
    print("Out-of-bound rate: ", len(invalid_scenes)/len(output))

    KL_divergence, a, b, cats = object_category_KL_divergence(output, {k:v for k, v in test_data_regular.items() if k in splits['rect_test']}, stats)
    print("KL Divergence: ", KL_divergence)