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
import argparse
import openai
from transformers import GPT2TokenizerFast
from utils import *

from parse_llm_output import parse_layout

openai.organization = ""
openai.api_key = ""
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# GPT-3 Type
gpt_name = {
    'gpt3.5': 'text-davinci-003',
    'gpt3.5-chat': 'gpt-3.5-turbo',
    'gpt4': 'gpt-4',
}

parser = argparse.ArgumentParser(prog='LayoutGPT: text-based image layout planning', description='Use LayoutGPT to generate image layouts.')
parser.add_argument('--input_info_dir', type=str, default='./dataset/NSR-1K')
parser.add_argument('--base_output_dir', type=str, default='./llm_output')
parser.add_argument('--setting', type=str, default='counting', choices=['counting', 'spatial'])
parser.add_argument('--matching_content_type', type=str, default='visual')
parser.add_argument('--gpt_type', type=str, default='gpt4', choices=list(gpt_name.keys()))
parser.add_argument('--icl_type', type=str, default='k-similar', choices=['fixed-random', 'k-similar'])
parser.add_argument('--K', type=int, default=8)
parser.add_argument('--gpt_input_length_limit', type=int, default=3000)
parser.add_argument('--canvas_size', type=int, default=64)
parser.add_argument("--n_iter", type=int, default=5)
parser.add_argument("--test", action='store_true')
parser.add_argument('--verbose', default=False, action='store_true')
args = parser.parse_args()


if args.icl_type == 'k-similar':
    # Load CLIP model
    clip_feature_name = 'ViT-L/14'.lower().replace('/', '-')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load('ViT-L/14', device=device)
    clip_model = clip_model.to(device)


def load_features(content):
    """Load visual/text features from npz file"""
    np_filename = os.path.join(
        args.input_info_dir, args.setting,
        f'train.{args.setting}.{clip_feature_name}.{content}.npz',
    )
    feature_list = np.load(np_filename)['feature_list']
    features = torch.HalfTensor(feature_list).to(device)
    features /= features.norm(dim=-1, keepdim=True)
    return features


def create_exemplar_prompt(caption, object_list, canvas_size, is_chat=False):
    if is_chat:
        prompt = ''
    else:
        prompt = f'\nPrompt: {caption}\nLayout:\n'

    for obj_info in object_list:
        category, bbox = obj_info
        coord_list = [int(i*canvas_size) for i in bbox]
        x, y, w, h = coord_list
        prompt += f'{category} {{height: {h}px; width: {w}px; top: {y}px; left: {x}px; }}\n'
    return prompt


def form_prompt_for_chatgpt(text_input, top_k, supporting_examples=None, features=None):
    message_list = []
    system_prompt = 'Instruction: Given a sentence prompt that will be used to generate an image, plan the layout of the image.' \
                'The generated layout should follow the CSS style, where each line starts with the object description ' \
                'and is followed by its absolute position. ' \
                'Formally, each line should be like "object {{width: ?px; height: ?px; left: ?px; top: ?px; }}". ' \
                'The image is {}px wide and {}px high. ' \
                'Therefore, all properties of the positions should not exceed {}px, ' \
                'including the addition of left and width and the addition of top and height. \n'.format(args.canvas_size, args.canvas_size, args.canvas_size)
    message_list.append({'role': 'system', 'content': system_prompt})
    final_prompt = f'Prompt: {text_input}\nLayout:'
    total_length = len(tokenizer(system_prompt + final_prompt)['input_ids'])

    if args.icl_type == 'k-similar':
        # find most related supporting examples
        text_inputs = clip.tokenize(text_input, truncate=True).to(device)
        text_features = clip_model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * text_features @ features.T).softmax(dim=-1)
        _, indices = similarity[0].topk(top_k)
        supporting_examples = [supporting_examples[idx] for idx in indices]

    # loop through the related supporting examples, check if the prompt length exceed limit
    for supporting_example in supporting_examples:
        user_prompt = f'Prompt: {supporting_example["prompt"]}\nLayout:'
        if args.setting == 'counting':
            answer = create_exemplar_prompt(
                caption=supporting_example['prompt'],
                object_list=supporting_example['object_list'],
                canvas_size=args.canvas_size,
                is_chat=True
            )
        else:
            answer = create_exemplar_prompt(
                caption=supporting_example['prompt'],
                object_list=[supporting_example['obj1'], supporting_example['obj2']],
                canvas_size=args.canvas_size,
                is_chat=True
            )

        cur_len = len(tokenizer(user_prompt+answer)['input_ids'])
        if total_length + cur_len > args.gpt_input_length_limit:  # won't take the input that is too long
            break
        total_length += cur_len
        
        cur_messages = [
            {'role': 'user', 'content': user_prompt},
            {'role': 'assistant', 'content': answer},
        ]
        message_list = message_list[:1] + cur_messages + message_list[1:]
    
    # add final question
    message_list.append({'role': 'user', 'content': final_prompt})
    
    return message_list


def form_prompt_for_gpt3(text_input, top_k, supporting_examples=None, features=None):
    rtn_prompt = 'Instruction: Given a sentence prompt that will be used to generate an image, plan the layout of the image.' \
                'The generated layout should follow the CSS style, where each line starts with the object description ' \
                'and is followed by its absolute position. ' \
                'Formally, each line should be like "object {{width: ?px; height: ?px; left: ?px; top: ?px; }}". ' \
                'The image is {}px wide and {}px high. ' \
                'Therefore, all properties of the positions should not exceed {}px, ' \
                'including the addition of left and width and the addition of top and height. \n'.format(args.canvas_size, args.canvas_size, args.canvas_size)
    last_example = f'\nPrompt: {text_input}\nLayout:'
    prompting_examples = ''
    total_length = len(tokenizer(rtn_prompt + last_example)['input_ids'])

    if args.icl_type == 'k-similar':
        # find most related supporting examples
        text_inputs = clip.tokenize(text_input, truncate=True).to(device)
        text_features = clip_model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * text_features @ features.T).softmax(dim=-1)
        _, indices = similarity[0].topk(top_k)
        supporting_examples = [supporting_examples[idx] for idx in indices]

    # loop through the related supporting examples, check if the prompt length exceed limit
    for supporting_example in supporting_examples:
        if args.setting == 'counting':
            current_prompting_example = create_exemplar_prompt(
                caption=supporting_example['prompt'],
                object_list=supporting_example['object_list'],
                canvas_size=args.canvas_size,
            )
        else:
            current_prompting_example = create_exemplar_prompt(
                caption=supporting_example['prompt'],
                object_list=[supporting_example['obj1'], supporting_example['obj2']],
                canvas_size=args.canvas_size,
            )

        cur_len = len(tokenizer(current_prompting_example)['input_ids'])
        if total_length + cur_len > args.gpt_input_length_limit:  # won't take the input that is too long
            break
        prompting_examples = current_prompting_example + prompting_examples  # most similar example appear first
        total_length += cur_len
    
    # concatename prompts
    prompting_examples += last_example
    rtn_prompt += prompting_examples
    
    return rtn_prompt



def _main(args):
    # check if have been processed
    args.output_dir = os.path.join(args.base_output_dir, args.setting)
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = os.path.join(args.output_dir, f'{args.gpt_type}.{args.setting}.{args.icl_type}.k_{args.K}.px_{args.canvas_size}.json')
    if os.path.exists(output_filename):
        print(f'{output_filename} have been processed.')
        return

    # load val examples
    val_example_files = os.path.join(
        args.input_info_dir, args.setting,
        f'{args.setting}.val.json',
    )
    val_example_list = json.load(open(val_example_files))
    if args.test:
        val_example_list = val_example_list[:3]

    # load all training examples
    train_example_files = os.path.join(
        args.input_info_dir, args.setting,
        f'{args.setting}.train.json',
    )
    train_examples = json.load(open(train_example_files))
    if args.icl_type == 'fixed-random':
        random.seed(42)
        random.shuffle(train_examples)
        supporting_examples = train_examples[:args.K]
        features = None
    elif args.icl_type == 'k-similar':
        supporting_examples = train_examples
        features = load_features(args.matching_content_type)

    # GPT-3 prediction process
    args.gpt_name = gpt_name[args.gpt_type]
    all_prediction_list = []
    all_responses = []
    f_form_prompt = form_prompt_for_gpt3 if args.gpt_type == 'gpt3.5' else form_prompt_for_chatgpt
    f_gpt_create = openai.Completion.create if args.gpt_type == 'gpt3.5' else openai.ChatCompletion.create

    for val_example in tqdm(val_example_list, total=len(val_example_list), desc='test'):
        while True:
            top_k = args.K
            prompt_for_gpt = f_form_prompt(
                text_input=val_example['prompt'],
                top_k=top_k,
                supporting_examples=supporting_examples,
                features=features
            )
            if args.verbose:
                print(prompt_for_gpt)
                print('\n' + '-'*30)
                pdb.set_trace()

            try:
                input_kwargs = {
                    "model": args.gpt_name,
                    "temperature": 0.7,
                    "max_tokens": 256,
                    "top_p": 1.0,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                    "stop": "Prompt:",
                    "n": args.n_iter,
                }
                if args.gpt_type == 'gpt3.5':
                    input_kwargs["prompt"] = prompt_for_gpt
                else:
                    input_kwargs["messages"] = prompt_for_gpt
                response = f_gpt_create(**input_kwargs)
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

        all_responses.append(response)
        for i_iter in range(args.n_iter):
            # parse output
            predicted_object_list = []
            if args.gpt_type == 'gpt3.5':
                line_list = response.choices[i_iter]["text"].split('\n')
            else:
                line_list = response.choices[i_iter]["message"]["content"].split('\n')
                
            for line in line_list:
                if line == '':
                    continue
                try:
                    selector_text, bbox = parse_layout(line, canvas_size=args.canvas_size)
                    if selector_text == None:
                        print(line)
                        continue
                    predicted_object_list.append([selector_text, bbox])
                except ValueError as e:
                    pass
            all_prediction_list.append({
                'query_id': val_example['id'],
                'iter': i_iter,
                'prompt': val_example['prompt'],
                'object_list': predicted_object_list,
            })

    # save output
    with open(output_filename, 'w') as fout:
        json.dump(all_prediction_list, fout, indent=4, sort_keys=True)
    print(f'LayoutGPT ({args.gpt_type}) prediction results written to {output_filename}')


if __name__ == '__main__':
    _main(args)
