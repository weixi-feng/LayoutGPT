from collections import defaultdict
import difflib
import os
import argparse
import pdb
import random
import cssutils
from tqdm import tqdm
from utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str)
    args = parser.parse_args()

    ref_file = load_json("dataset/NSR-1K/spatial/spatial.val.json")
    ref_file = {x['id']: x for x in ref_file}

    fname = args.file
    basename = os.path.basename(fname)
    dirname = os.path.dirname(fname)

    assert "raw" not in basename
    responses = load_json(fname)

    n_correct = defaultdict(lambda: 0)
    n_miss = defaultdict(lambda: 0)
    n_type = defaultdict(lambda: 0)

    print(f"Evaluating {basename}")
    for r in tqdm(responses):
        try:
            ref_sample = ref_file[int(r['query_id'])]
        except:
            ref_sample = ref_file[int(r['id'])]
        ref_relation = ref_sample['relation']
        obj1, _ = ref_sample['obj1']
        obj2, _ = ref_sample['obj2']
        prompt_type = ref_sample['type']
        n_type[prompt_type] += 1
        
        pred_objects = [obj for obj in r['object_list'] if obj[1] != [0]*4 and obj[0] != None]
        
        all_objects = [pred_obj[0] for pred_obj in pred_objects]
        close_obj1 = difflib.get_close_matches(obj1, all_objects)[:1]
        pred_bbox1 = [obj[1] for i, obj in enumerate(pred_objects) if obj[0] in close_obj1]
        
        if len(pred_bbox1) == 0:
            n_miss[prompt_type] += 1
            continue
        
        close_obj2 = difflib.get_close_matches(obj2, all_objects)[:1]
        pred_bbox2 = [obj[1] for i, obj in enumerate(pred_objects) if obj[0] in close_obj2]
        if len(pred_bbox2) == 0:
            n_miss[prompt_type] += 1
            continue
        
        all_relations = [eval_spatial_relation(b1, b2) for b1 in pred_bbox1 for b2 in pred_bbox2]

        if ref_relation in all_relations:
            n_correct[prompt_type] += 1
        else:
            if ref_relation == 'next to' and ('left' in all_relations or 'right' in all_relations):
                n_correct[prompt_type] += 1
            else:
                pass
    
    for prompt_type in n_correct.keys():
        print(f'{basename} {prompt_type} (#eg: {n_type[prompt_type]})')
        acc = n_correct[prompt_type]/n_type[prompt_type]
        score_info = {'acc': acc, 'n_miss': n_miss[prompt_type]}
        print(f'\tAcc = {acc*100:.2f} %, #miss = {n_miss[prompt_type]}')

        # save output
        args.output_dir = os.path.join('./eval_score/spatial/')
        os.makedirs(args.output_dir, exist_ok=True)
        output_filename = os.path.join(args.output_dir, 'layout_eval.'+basename)
        with open(output_filename, 'w') as fout:
            json.dump(score_info, fout)

    print("{}, Overall, acc: {:.4f}, missing: {}".format(basename, sum(n_correct.values())/len(responses), sum(n_miss.values())))
