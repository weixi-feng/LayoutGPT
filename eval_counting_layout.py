import os
import json
import pdb
import numpy as np
from collections import defaultdict
import argparse


gpt_name = {
    'gpt3.5': 'text-davinci-003',
    'gpt3.5-chat': 'gpt-3.5-turbo',
    'gpt4': 'gpt-4',
}

parser = argparse.ArgumentParser(prog='Layout Evaluation', description='Layout evaluation for counting prompts.')
parser.add_argument('--input_info_dir', type=str, default='./dataset/NSR-1K/counting/')
parser.add_argument("-f", "--file", type=str)
parser.add_argument('--score_dir', type=str, default='./eval_score/counting/')
parser.add_argument('--setting', type=str, default='counting', choices=['counting', 'counting.single_category', 'counting.two_categories', 'counting.reasoning', 'counting.mscoco'])
parser.add_argument('--verbose', default=False, action='store_true')
args = parser.parse_args()


def _main(args):
    # load prediction results
    # pred_filename = f'{args.gpt_type}.{args.setting}.{args.icl_type}.k_{args.K}.px_{args.canvas_size}.json'
    # prediction_list = json.load(open(os.path.join(args.prediction_dir, pred_filename)))
    pred_filename = os.path.basename(args.file)
    prediction_list = json.load(open(args.file))

    # load gt val examples
    val_example_files = os.path.join(
        args.input_info_dir,
        f'{args.setting}.val.json',
    )
    val_example_list = json.load(open(val_example_files))
    id2subtype = {d['id']:d['sub-type'] for d in val_example_list}
    ref_file = {x['id']: x for x in val_example_list}


    precision_list = []
    recall_list = []
    iou_list = []
    mae_list = []
    acc_list = []
    for pred_eg in prediction_list:
        val_eg = ref_file[int(pred_eg['query_id'])]
        pred_object_count = defaultdict(lambda: 0)
        for category, _ in pred_eg['object_list']:
            if category is None: continue
            for x in val_eg['num_object']:
                if category.lstrip("a ").lstrip("an ").lstrip("the ") in x[0] or x[0] in category.lstrip("a ").lstrip("an ").lstrip("the "):
                    category = x[0]
            pred_object_count[category] += 1
        
        if id2subtype[pred_eg['query_id']] == 'comparison':
            (obj1, gt_num1), (obj2, gt_num2) = val_eg['num_object']
            pred_num1 = pred_object_count[obj1]
            pred_num2 = pred_object_count[obj2]
            
            # equal cases
            if gt_num1 == gt_num2 == pred_num1 == pred_num2:
                acc_list.append(1)
            # < or >
            elif gt_num1 == pred_num1 and (gt_num1-gt_num2)*(pred_num1-pred_num2) > 0:
                acc_list.append(1)
            else:
                acc_list.append(0)

        else:
            cnt_gt_total = 0
            cnt_pred_total = sum(pred_object_count.values())
            cnt_intersection_total = 0
            cnt_union_total = 0
            absolute_error = 0
            appeared_category_list = []
            all_matched = True

            for category, gt_cnt in val_eg['num_object']:
                cnt_gt_total += gt_cnt
                pred_cnt = pred_object_count[category]
                cnt_intersection_total += min(pred_cnt, gt_cnt)
                cnt_union_total += max(pred_cnt, gt_cnt)
                absolute_error += abs(pred_cnt - gt_cnt)
                appeared_category_list.append(category)
                if pred_cnt != gt_cnt:  # check if all the mentioned objects are predicted correctly
                    all_matched = False

            # accuracy
            acc_list.append(1 if all_matched else 0)

            # MAE
            if not len(appeared_category_list):
                mae_list.append(0)
            else:
                mae_list.append(float(absolute_error) / len(appeared_category_list))

            # precision, recall, IoU
            if not cnt_intersection_total:
                precision_list.append(0)
                recall_list.append(0)
                iou_list.append(0)
            else:
                precision_list.append(float(cnt_intersection_total) / cnt_pred_total)
                recall_list.append(float(cnt_intersection_total) / cnt_gt_total)
                iou_list.append(float(cnt_intersection_total) / cnt_union_total)

    # print results
    # print(f'Setting: {args.setting} (#eg: {len(prediction_list)})\tGPT-3: {args.gpt_type} - {args.icl_type}\tk = {args.K}')
    print(f'{pred_filename}, #eg: {len(prediction_list)}')
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_acc = np.mean(acc_list)
    avg_mae = np.mean(mae_list)
    avg_iou = np.mean(iou_list)
    score_info = {
        'precision': avg_precision,
        'recall': avg_recall,
        'precision_list': precision_list,
        'recall_list': recall_list,
        'acc': avg_acc,
        'acc_list': acc_list,
        'mae': avg_mae,
        'mae_list': mae_list,
        'iou': avg_iou,
        'iou_list': iou_list,
    }
    print(f'\tPrecision = {avg_precision*100:.2f} %\n\tRecall = {avg_recall*100:.2f} %' \
            f'\n\tIoU = {avg_iou*100:.2f} %\n\tMAE = {avg_mae:.2f}\n\tacc = {avg_acc*100:.2f} %')
    # save output
    os.makedirs(args.score_dir, exist_ok=True)
    output_filename = os.path.join(args.score_dir, f'layout_eval.{pred_filename}')
    with open(output_filename, 'w') as fout:
        json.dump(score_info, fout)
    return score_info


if __name__ == '__main__':
    _main(args)