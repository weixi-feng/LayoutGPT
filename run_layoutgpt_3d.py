import os
import os.path as op
import json
import pdb
import clip
import torch
import numpy as np
from tqdm import tqdm
import time
import random
from PIL import Image
import argparse
import openai
from utils import *

from transformers import GPT2TokenizerFast

from parse_llm_output import parse_3D_layout

openai.organization = ""
openai.api_key = "" 
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


parser = argparse.ArgumentParser(prog='LayoutGPT for scene synthesis', description='Use GPTs to predict 3D layout for indoor scenes.')
parser.add_argument('--room', type=str, default='bedroom', choices=['bedroom','livingroom'])
parser.add_argument('--dataset_dir', type=str)
parser.add_argument('--gpt_type', type=str, default='gpt4', choices=['gpt3.5', 'gpt3.5-chat', 'gpt4'])
parser.add_argument('--icl_type', type=str, default='k-similar', choices=['fixed-random', 'k-similar'])
parser.add_argument('--base_output_dir', type=str, default='./llm_output/3D/')
parser.add_argument('--K', type=int, default=8)
parser.add_argument('--gpt_input_length_limit', type=int, default=7000)
parser.add_argument('--unit', type=str, choices=['px', 'm', ''], default='px')
parser.add_argument("--n_iter", type=int, default=1)
parser.add_argument("--test", action='store_true')
parser.add_argument('--verbose', default=False, action='store_true')
parser.add_argument("--suffix", type=str, default="")
parser.add_argument("--normalize", action='store_true')
parser.add_argument("--regular_floor_plan", action='store_true')
parser.add_argument("--temperature", type=float, default=0.7)
args = parser.parse_args()

# GPT Type
gpt_name = {
    'gpt3.5': 'text-davinci-003',
    'gpt3.5-chat': 'gpt-3.5-turbo',
    'gpt4': 'gpt-4',
}


def load_room_boxes(prefix, id, stats, unit):
    data = np.load(op.join(prefix, id, 'boxes.npz'))
    x_c, y_c = data['floor_plan_centroid'][0], data['floor_plan_centroid'][2]
    x_offset  = min(data['floor_plan_vertices'][:,0])
    y_offset = min(data['floor_plan_vertices'][:,2])
    room_length = max(data['floor_plan_vertices'][:,0]) - min(data['floor_plan_vertices'][:,0])
    room_width = max(data['floor_plan_vertices'][:,2]) - min(data['floor_plan_vertices'][:,2])    
    vertices = np.stack((data['floor_plan_vertices'][:,0]-x_offset, data['floor_plan_vertices'][:,2]-y_offset), axis=1)
    vertices = np.asarray([list(nxy) for nxy in set(tuple(xy) for xy in vertices)])
    
    # normalize
    if args.normalize:
        norm = min(room_length, room_width)
        room_length, room_width = room_length/norm, room_width/norm
        vertices /= norm
        if unit in ['px', '']:
            scale_factor = 256
            room_length, room_width = int(room_length*scale_factor), int(room_width*scale_factor)

    vertices = [f'({v[0]:.2f}, {v[1]:.2f})' for v in vertices]

    if unit in ['px', '']:
        condition = f"Condition:\n"
        if args.room == 'livingroom':
            if 'dining' in id.lower():
                condition += f"Room Type: living room & dining room\n"
            else:
                condition += f"Room Type: living room\n"
        else:
            condition += f"Room Type: {args.room}\n"
        condition += f"Room Size: max length {room_length}{unit}, max width {room_width}{unit}\n"
    else:
        condition = f"Condition:\n" \
                    f"Room Type: {args.room}\n" \
                    f"Room Size: max length {room_length:.2f}{unit}, max width {room_width:.2f}{unit}\n"

    layout = 'Layout:\n'
    for label, size, angle, loc in zip(data['class_labels'], data['sizes'], data['angles'], data['translations']):
        label_idx = np.where(label)[0][0]
        if label_idx >= len(stats['object_types']): # NOTE:
            continue
        cat = stats['object_types'][label_idx]
        
        length, height, width = size # NOTE: half the actual size
        length, height, width = length*2, height*2, width*2
        orientation = round(angle[0] / 3.1415926 * 180)
        dx,dz,dy = loc # NOTE: center point
        dx = dx+x_c-x_offset
        dy = dy+y_c-y_offset

        # normalize
        if args.normalize:
            length, width, height = length/norm, width/norm, height/norm
            dx, dy, dz = dx/norm, dy/norm, dz/norm
            if unit in ['px', '']:
                length, width, height = int(length*scale_factor), int(width*scale_factor), int(height*scale_factor)
                dx, dy, dz = int(dx*scale_factor), int(dy*scale_factor), int(dz*scale_factor)

        if unit in ['px', '']:
            layout += f"{cat} {{length: {length}{unit}; " \
                                f"width: {width}{unit}; " \
                                f"height: {height}{unit}; " \
                                f"left: {dx}{unit}; " \
                                f"top: {dy}{unit}; " \
                                f"depth: {dz}{unit};" \
                                f"orientation: {orientation} degrees;}}\n"                                
        else:
            layout += f"{cat} {{length: {length:.2f}{unit}; " \
                                f"height: {height:.2f}{unit}; " \
                                f"width: {width:.2f}{unit}; " \
                                f"orientation: {orientation} degrees; " \
                                f"left: {dx:.2f}{unit}; " \
                                f"top: {dy:.2f}{unit}; " \
                                f"depth: {dz:.2f}{unit};}}\n" 

    return condition, layout, dict(data)


def load_set(prefix, ids, stats, unit):
    id2prompt = {}
    meta_data = {}
    for id in tqdm(ids):
        condition, layout, data = load_room_boxes(prefix, id, stats, unit)
        id2prompt[id] = [condition, layout]
        meta_data[id] = data
    return id2prompt, meta_data


def load_features(meta_data, floor_plan=True):
    features = {}
    for id, data in meta_data.items():
        if floor_plan:
            features[id] = np.asarray(Image.fromarray(data['room_layout'].squeeze()).resize((64,64)))
        else:
            room_length = max(data['floor_plan_vertices'][:,0]) - min(data['floor_plan_vertices'][:,0])
            room_width = max(data['floor_plan_vertices'][:,2]) - min(data['floor_plan_vertices'][:,2])  
            features[id] = np.asarray([room_length, room_width])
    return features


def get_closest_room(train_features, val_feature):
    '''
    train_features
    '''
    distances = [[id, ((feat-val_feature)**2).mean()] for id, feat in train_features.items()]
    distances = sorted(distances, key=lambda x: x[1])
    sorted_ids, _ = zip(*distances)
    return sorted_ids


def create_prompt(sample):
    return sample[0] + sample[1] + "\n\n"


def form_prompt_for_gpt3(text_input, top_k, stats, supporting_examples,
                            train_features=None, val_feature=None):
    unit_name = 'pixel' if args.unit in ['px', ''] else 'meters'
    class_freq = [f"{obj}: {round(stats['class_frequencies'][obj], 4)}" for obj in stats['object_types']]
    rtn_prompt = 'Instruction: synthesize the 3D layout of an indoor scene. ' \
                'The generated 3D layout should follow the CSS style, where each line starts with the furniture category ' \
                'and is followed by the 3D size, orientation and absolute position. ' \
                "Formally, each line should follow the template: \n" \
                f"FURNITURE {{length: ?{args.unit}: width: ?{args.unit}; height: ?{args.unit}; left: ?{args.unit}; top: ?{args.unit}; depth: ?{args.unit}; orientation: ? degrees;}}\n" \
                f'All values are in {unit_name} but the orientation angle is in degrees.\n\n' \
                f"Available furnitures: {', '.join(stats['object_types'])} \n" \
                f"Overall furniture frequencies: ({'; '.join(class_freq)})\n\n"               
                
    last_example = f'{text_input[0]}Layout:\n'
    prompting_examples = ''
    total_length = len(tokenizer(rtn_prompt + last_example)['input_ids'])

    if args.icl_type == 'k-similar':
        assert train_features is not None
        sorted_ids = get_closest_room(train_features, val_feature)
        supporting_examples = [supporting_examples[id] for id in sorted_ids[:top_k]]
        if args.test:
            print("retrieved examples:")
            print("\n".join(sorted_ids[:top_k]))
            pdb.set_trace()

    # loop through the related supporting examples, check if the prompt length exceed limit
    for i, supporting_example in enumerate(supporting_examples[:top_k]):
        current_prompting_example = create_prompt(supporting_example)
        cur_len = len(tokenizer(current_prompting_example)['input_ids'])
        if total_length + cur_len > args.gpt_input_length_limit:  # won't take the input that is too long
            print(f"{i+1}th exemplar exceed max length")
            break
        prompting_examples = current_prompting_example + prompting_examples 
        total_length += cur_len
    
    prompting_examples += last_example
    rtn_prompt += prompting_examples
    
    return rtn_prompt


def form_prompt_for_chatgpt(text_input, top_k, stats, supporting_examples,
                            train_features=None, val_feature=None):
    message_list = []
    unit_name = 'pixel' if args.unit in ['px', ''] else 'meters'
    class_freq = [f"{obj}: {round(stats['class_frequencies'][obj], 4)}" for obj in stats['object_types']]
    rtn_prompt = 'You are a 3D indoor scene designer. \nInstruction: synthesize the 3D layout of an indoor scene. ' \
                'The generated 3D layout should follow the CSS style, where each line starts with the furniture category ' \
                'and is followed by the 3D size, orientation and absolute position. ' \
                "Formally, each line should follow the template: \n" \
                f"FURNITURE {{length: ?{args.unit}: width: ?{args.unit}; height: ?{args.unit}; orientation: ? degrees; left: ?{args.unit}; top: ?{args.unit}; depth: ?{args.unit};}}\n" \
                f'All values are in {unit_name} but the orientation angle is in degrees.\n\n' \
                f"Available furnitures: {', '.join(stats['object_types'])} \n" \
                f"Overall furniture frequencies: ({'; '.join(class_freq)})\n\n"

    message_list.append({'role': 'system', 'content': rtn_prompt})
    last_example = f'{text_input[0]}Layout:\n'
    total_length = len(tokenizer(rtn_prompt + last_example)['input_ids'])


    if args.icl_type == 'k-similar':
        assert train_features is not None
        sorted_ids = get_closest_room(train_features, val_feature)
        supporting_examples = [supporting_examples[id] for id in sorted_ids[:top_k]]
        if args.test:
            print("retrieved examples:")
            print("\n".join(sorted_ids[:top_k]))

    # loop through the related supporting examples, check if the prompt length exceed limit
    for i, supporting_example in enumerate(supporting_examples[:top_k]):
        cur_len = len(tokenizer(supporting_example[0]+supporting_example[1])['input_ids'])
        if total_length + cur_len > args.gpt_input_length_limit:  # won't take the input that is too long
            print(f"{i+1}th exemplar exceed max length")
            break
        total_length += cur_len

        current_messages = [
            {'role': 'user', 'content': supporting_example[0]+"Layout:\n"},
            {'role': 'assistant', 'content': supporting_example[1].lstrip("Layout:\n")},
        ]
        message_list = message_list + current_messages
    
    # concatename prompts for gpt4
    message_list.append({'role': 'user', 'content': last_example})

    return message_list


def _main(args):
    dataset_prefix = f"{args.dataset_dir}"
    with open(f"dataset/3D/{args.room}_splits.json", "r") as file:
        splits = json.load(file)
        
    with open(f"{args.dataset_dir}/dataset_stats.txt", "r") as file:
        stats = json.load(file)

    if args.regular_floor_plan:
        args.suffix += '_regular'

    # check if have been processed
    args.output_dir = args.base_output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = os.path.join(args.output_dir, f'{args.gpt_type}.{args.room}.{args.icl_type}.k_{args.K}.{args.unit}{args.suffix}.json')
    os.makedirs(os.path.join(args.output_dir, 'raw'), exist_ok=True)
    raw_output_filename = os.path.join(args.output_dir, 'raw', f'raw_{args.gpt_type}.{args.room}.{args.icl_type}.k_{args.K}.{args.unit}{args.suffix}.json')

    # load train examples
    train_ids = splits['rect_train'] if args.regular_floor_plan else splits['train']
    train_data, meta_train_data = load_set(dataset_prefix, train_ids, stats, args.unit)

    # load val examples
    val_ids = splits['rect_test'] if args.regular_floor_plan else splits['test']
    val_data, meta_val_data = load_set(dataset_prefix, val_ids, stats, args.unit)
    val_features = load_features(meta_val_data)
    print(f"Loaded {len(train_data)} train samples and {len(val_data)} validation samples")

    if args.test:
        val_data = {k:v for k, v in list(val_data.items())[:5]}
        args.verbose = True
        args.n_iter = 1

    if args.icl_type == 'fixed-random':
        # load fixed supporting examples
        all_supporting_examples = list(train_data.values())
        supporting_examples = all_supporting_examples[:args.K]
        train_features = None
    elif args.icl_type == 'k-similar':
        supporting_examples = train_data
        train_features = load_features(meta_train_data)
    
    # GPT-3 prediction process
    args.gpt_name = gpt_name[args.gpt_type]
    all_prediction_list = []
    all_responses = []
    top_k = args.K

    n_lines = []
    n_furnitures = []
    for val_id, val_example in tqdm(val_data.items(), total=len(val_data), desc='gpt3'):
        # predict
        while True:
            if args.gpt_type == 'gpt3.5':
                prompt_for_gpt3 = form_prompt_for_gpt3(
                    text_input=val_example,
                    top_k=top_k,
                    stats=stats,
                    supporting_examples=supporting_examples,
                    train_features=train_features,
                    val_feature=val_features[val_id]
                )
            elif args.gpt_type in ['gpt3.5-chat', 'gpt4']:
                prompt_for_gpt3 = form_prompt_for_chatgpt(
                    text_input=val_example,
                    top_k=top_k,
                    stats=stats,
                    supporting_examples=supporting_examples,
                    train_features=train_features,
                    val_feature=val_features[val_id]
                )
            else:
                raise NotImplementedError

            if args.verbose:
                print(val_id)
                print(prompt_for_gpt3)
                print('\n' + '-'*30)
                pdb.set_trace()

            if op.exists(op.join(args.output_dir, 'tmp', args.gpt_type, f"{val_id}.json")):
                response = json.load(open(op.join(args.output_dir, 'tmp', args.gpt_type, f"{val_id}.json")))
                break

            try:
                if args.gpt_type == 'gpt3.5':
                    response = openai.Completion.create( # use openai.ChatCompletion for GPT-4
                        model=args.gpt_name,
                        prompt=prompt_for_gpt3,
                        temperature=args.temperature,
                        max_tokens=1024 if args.room=='livingroom' else 512,
                        top_p=1.0,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        stop="Condition:",
                        n=args.n_iter,
                    )
                elif args.gpt_type in ['gpt3.5', 'gpt4']:
                    response = openai.ChatCompletion.create(
                        model=args.gpt_name,
                        messages=prompt_for_gpt3,
                        temperature=0.7,
                        max_tokens=1024 if args.room=='livingroom' else 512,
                        top_p=1.0,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        stop="Condition:",
                        n=args.n_iter,
                    )
                else:
                    raise NotImplementedError
                break
            except openai.error.ServiceUnavailableError:
                print('OpenAI ServiceUnavailableError.\tWill try again in 5 seconds.')
                time.sleep(5)
            except openai.error.RateLimitError:
                print('OpenAI RateLimitError.\tWill try again in 5 seconds.')
                time.sleep(5)
            except openai.error.InvalidRequestError as e:
                print(e)
                print('Input too long. Will shrink the prompting examples.')
                top_k -= 1
            except openai.error.APIError as e:
                print('OpenAI Bad Gateway Error.\tWill try again in 5 seconds.')
                time.sleep(5)
        
        os.makedirs(op.join(args.output_dir, 'tmp', args.gpt_type), exist_ok=True)
        write_json(op.join(args.output_dir, 'tmp', args.gpt_type, f"{val_id}.json"), response)
        response['prompt'] = prompt_for_gpt3
        all_responses.append(response)

        for i_iter in range(args.n_iter):
            # parse output
            if args.verbose:
                try:
                    print(response['choices'][i_iter]['text'])
                except:
                    print(response['choices'][i_iter]['message']['content'])

            predicted_object_list = []
            if args.gpt_type == 'gpt3.5':
                line_list = response['choices'][i_iter]['text'].split('\n')
            else:
                line_list = response['choices'][i_iter]['message']['content'].split('\n')

            n_lines.append(len(line_list))
            for line in line_list:
                if line == '':
                    continue
                try:
                    selector_text, bbox = parse_3D_layout(line, args.unit)
                    if selector_text == None:
                        print(line)
                        continue
                    predicted_object_list.append([selector_text, bbox])
                except ValueError as e:
                    pass
            n_furnitures.append(len(predicted_object_list))
            all_prediction_list.append({
                'query_id': val_id,
                'iter': i_iter,
                'prompt': val_example[0],
                'object_list': predicted_object_list,
            })

            if args.gpt_type in ['gpt4']:
                time.sleep(3)

    # # save output
    with open(raw_output_filename, 'w') as fout:
        json.dump(all_responses, fout, indent=4, sort_keys=True)

    with open(output_filename, 'w') as fout:
        json.dump(all_prediction_list, fout, indent=4, sort_keys=True)
    print(f'GPT-3 ({args.gpt_type}) prediction results written to {output_filename}')
    print(f"{np.mean(n_lines)}, {np.mean(n_furnitures)}")

if __name__ == '__main__':
    _main(args)
