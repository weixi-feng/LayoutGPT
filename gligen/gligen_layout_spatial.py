import argparse
import json
import math
import pdb
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from tqdm import tqdm
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import os
from transformers import CLIPProcessor, CLIPModel
from copy import deepcopy
import torch 
from ldm.util import instantiate_from_config
from trainer import read_official_ckpt, batch_to_device
from inpaint_mask_func import draw_masks_from_boxes
import numpy as np
import clip 
from functools import partial
import torchvision.transforms.functional as F
import cv2

device = "cuda"


def set_alpha_scale(model, alpha_scale):
    from ldm.modules.attention import GatedCrossAttentionDense, GatedSelfAttentionDense
    for module in model.modules():
        if type(module) == GatedCrossAttentionDense or type(module) == GatedSelfAttentionDense:
            module.scale = alpha_scale


def alpha_generator(length, type=None):
    """
    length is total timestpes needed for sampling. 
    type should be a list containing three values which sum should be 1
    
    It means the percentage of three stages: 
    alpha=1 stage 
    linear deacy stage 
    alpha=0 stage. 
    
    For example if length=100, type=[0.8,0.1,0.1]
    then the first 800 stpes, alpha will be 1, and then linearly decay to 0 in the next 100 steps,
    and the last 100 stpes are 0.    
    """
    if type == None:
        type = [1,0,0]

    assert len(type)==3 
    assert type[0] + type[1] + type[2] == 1
    
    stage0_length = int(type[0]*length)
    stage1_length = int(type[1]*length)
    stage2_length = length - stage0_length - stage1_length
    
    if stage1_length != 0: 
        decay_alphas = np.arange(start=0, stop=1, step=1/stage1_length)[::-1]
        decay_alphas = list(decay_alphas)
    else:
        decay_alphas = []
        
    
    alphas = [1]*stage0_length + decay_alphas + [0]*stage2_length
    
    assert len(alphas) == length
    
    return alphas



def load_ckpt(ckpt_path):
    
    saved_ckpt = torch.load(ckpt_path)
    config = saved_ckpt["config_dict"]["_content"]
    
    model = instantiate_from_config(config['model']).to(device).eval()
    autoencoder = instantiate_from_config(config['autoencoder']).to(device).eval()
    text_encoder = instantiate_from_config(config['text_encoder']).to(device).eval()
    diffusion = instantiate_from_config(config['diffusion']).to(device)

    # donot need to load official_ckpt for self.model here, since we will load from our ckpt
    model.load_state_dict( saved_ckpt['model'] )
    autoencoder.load_state_dict( saved_ckpt["autoencoder"]  )
    text_encoder.load_state_dict( saved_ckpt["text_encoder"]  )
    diffusion.load_state_dict( saved_ckpt["diffusion"]  )

    return model, autoencoder, text_encoder, diffusion, config




def project(x, projection_matrix):
    """
    x (Batch*768) should be the penultimate feature of CLIP (before projection)
    projection_matrix (768*768) is the CLIP projection matrix, which should be weight.data of Linear layer 
    defined in CLIP (out_dim, in_dim), thus we need to apply transpose below.  
    this function will return the CLIP feature (without normalziation)
    """
    return x@torch.transpose(projection_matrix, 0, 1)


def get_clip_feature(model, processor, input, is_image=False):
    which_layer_text = 'before'
    which_layer_image = 'after_reproject'

    if is_image:
        if isinstance(input, list):
            if None in input: return None
        else:
            if input == None: return None
        image = Image.open(input).convert("RGB")
        inputs = processor(images=[image],  return_tensors="pt", padding=True)
        inputs['pixel_values'] = inputs['pixel_values'].cuda() # we use our own preprocessing without center_crop 
        inputs['input_ids'] = torch.tensor([[0,1,2,3]]).cuda()  # placeholder
        outputs = model(**inputs)
        feature = outputs.image_embeds 
        if which_layer_image == 'after_reproject':
            feature = project( feature, torch.load('projection_matrix').cuda().T ).squeeze(0)
            feature = ( feature / feature.norm() )  * 28.7 
            feature = feature.unsqueeze(0)
    else:
        if isinstance(input, list):
            if None in input: return None
        else:
            if input == None: return None
        inputs = processor(text=input,  return_tensors="pt", padding=True)
        inputs['input_ids'] = inputs['input_ids'].cuda()
        inputs['pixel_values'] = torch.ones(1,3,224,224).cuda() # placeholder 
        inputs['attention_mask'] = inputs['attention_mask'].cuda()
        outputs = model(**inputs)
        if which_layer_text == 'before':
            feature = outputs.text_model_output.pooler_output
    return feature


def complete_mask(has_mask, max_objs):
    mask = torch.ones(1,max_objs)
    if has_mask == None:
        return mask 

    if type(has_mask) == int or type(has_mask) == float:
        return mask * has_mask
    else:
        for idx, value in enumerate(has_mask):
            mask[0,idx] = value
        return mask



@torch.no_grad()
def prepare_batch(meta, batch=1, max_objs=30):
    phrases, images = meta.get("phrases"), meta.get("images")
    images = [None]*len(phrases) if images==None else images 
    phrases = [None]*len(images) if phrases==None else phrases 

    version = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(version).cuda()
    processor = CLIPProcessor.from_pretrained(version)

    boxes = torch.zeros(max_objs, 4)
    masks = torch.zeros(max_objs)
    text_masks = torch.zeros(max_objs)
    image_masks = torch.zeros(max_objs)
    text_embeddings = torch.zeros(max_objs, 768)
    image_embeddings = torch.zeros(max_objs, 768)
    
    text_features = []
    image_features = []
    for phrase, image in zip(phrases, images):
        text_features.append(  get_clip_feature(model, processor, phrase, is_image=False) )
        image_features.append( get_clip_feature(model, processor, image,  is_image=True) )

    for idx, (box, text_feature, image_feature) in enumerate(zip( meta['locations'], text_features, image_features)):
        boxes[idx] = torch.tensor(box)
        masks[idx] = 1
        if text_feature is not None:
            text_embeddings[idx] = text_feature
            text_masks[idx] = 1 
        if image_feature is not None:
            image_embeddings[idx] = image_feature
            image_masks[idx] = 1 
    out = {
        "boxes" : boxes.unsqueeze(0).repeat(batch,1,1),
        "masks" : masks.unsqueeze(0).repeat(batch,1),
        "text_masks" : text_masks.unsqueeze(0).repeat(batch,1)*complete_mask( meta.get("text_mask"), max_objs ),
        "image_masks" : image_masks.unsqueeze(0).repeat(batch,1)*complete_mask( meta.get("image_mask"), max_objs ),
        "text_embeddings"  : text_embeddings.unsqueeze(0).repeat(batch,1,1),
        "image_embeddings" : image_embeddings.unsqueeze(0).repeat(batch,1,1)
    }

    return batch_to_device(out, device) 


@torch.no_grad()
def prepare_batch_hetero(model, processor, metas, max_objs=30):
    batch_boxes = []
    batch_masks = []
    batch_text_masks = []
    batch_image_masks = []
    batch_text_embeddings = []
    batch_image_embeddings = []

    for meta in metas:
        phrases, images = meta.get("phrases"), meta.get("images")
        images = [None]*len(phrases) if images==None else images 
        phrases = [None]*len(images) if phrases==None else phrases 

        boxes = torch.zeros(max_objs, 4)
        masks = torch.zeros(max_objs)
        text_masks = torch.zeros(max_objs)
        image_masks = torch.zeros(max_objs)
        text_embeddings = torch.zeros(max_objs, 768)
        image_embeddings = torch.zeros(max_objs, 768)
    
        text_features = get_clip_feature(model, processor, phrases, is_image=False)
        image_features = get_clip_feature(model, processor, images,  is_image=True)

        n_obj = len(meta['locations'])
        boxes[:n_obj] = torch.tensor(meta['locations'])
        masks[:n_obj] = 1
        if text_features is not None:
            text_embeddings[:n_obj] = text_features
            text_masks[:n_obj] = 1
        if image_features is not None:
            image_embeddings[:n_obj] = image_features
            image_masks[:n_obj] = 1
        
        batch_boxes.append(boxes)
        batch_masks.append(masks)
        batch_text_masks.append(text_masks)
        batch_image_masks.append(image_masks)
        batch_text_embeddings.append(text_embeddings)
        batch_image_embeddings.append(image_embeddings)

    out = {
        "boxes" : torch.stack(batch_boxes),
        "masks" : torch.stack(batch_masks),
        "text_masks" : torch.stack(batch_text_masks)*complete_mask( meta.get("text_mask"), max_objs ),
        "image_masks" : torch.stack(batch_image_masks)*complete_mask( meta.get("image_mask"), max_objs ),
        "text_embeddings"  : torch.stack(batch_text_embeddings),
        "image_embeddings" : torch.stack(batch_image_embeddings)
    }

    return batch_to_device(out, device) 



@torch.no_grad()
def prepare_batch_kp(meta, batch=1, max_persons_per_image=8):
    
    points = torch.zeros(max_persons_per_image*17,2)
    idx = 0 
    for this_person_kp in meta["locations"]:
        for kp in this_person_kp:
            points[idx,0] = kp[0]
            points[idx,1] = kp[1]
            idx += 1
    
    # derive masks from points
    masks = (points.mean(dim=1)!=0) * 1 
    masks = masks.float()

    out = {
        "points" : points.unsqueeze(0).repeat(batch,1,1),
        "masks" : masks.unsqueeze(0).repeat(batch,1),
    }

    return batch_to_device(out, device) 


def fake_loader(data_list, batch_size=1):
    n_batch = int(math.ceil(len(data_list) / batch_size))
    return [data_list[i*batch_size:(i+1)*batch_size] for i in range(n_batch)]


@torch.no_grad()
def run(meta_list, config, starting_noise=None):
    meta = meta_list[0]

    # - - - - - prepare models - - - - - # 
    model, autoencoder, text_encoder, diffusion, config = load_ckpt(meta["ckpt"])

    grounding_tokenizer_input = instantiate_from_config(config['grounding_tokenizer_input'])
    model.grounding_tokenizer_input = grounding_tokenizer_input

    # - - - - - update config from args - - - - - # 
    config.update( vars(args) )
    config = OmegaConf.create(config)

    # load CLIP text encoder
    version = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(version).cuda()
    processor = CLIPProcessor.from_pretrained(version)


    output_folder = os.path.join( args.folder,  meta["save_folder_name"])
    output_folder_clean = os.path.join(output_folder, "clean")
    output_folder_layout = os.path.join(output_folder, "w_layout")
    os.makedirs( output_folder_clean, exist_ok=True)
    os.makedirs( output_folder_layout, exist_ok=True)

    for metas in tqdm(fake_loader(meta_list, config.batch_size)):
        # - - - - - prepare batch - - - - - #
        real_batch_size = len(metas)
        if "keypoint" in meta["ckpt"]:
            batch = prepare_batch_kp(meta, config.batch_size)
        else:
            batch = prepare_batch_hetero(clip_model, processor, metas)

        context = text_encoder.encode(  [meta["prompt"] for meta in metas]  )
        uc = text_encoder.encode( real_batch_size*[""] )

        # - - - - - sampler - - - - - # 
        alpha_generator_func = partial(alpha_generator, type=metas[0].get("alpha_type"))
        if config.no_plms:
            sampler = DDIMSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
            steps = 250 
        else:
            sampler = PLMSSampler(diffusion, model, alpha_generator_func=alpha_generator_func, set_alpha_scale=set_alpha_scale)
            steps = 50 
    
        # - - - - - input for gligen - - - - - #
        grounding_input = grounding_tokenizer_input.prepare(batch)
        input = dict(
                    x = starting_noise, 
                    timesteps = None, 
                    context = context, 
                    grounding_input = grounding_input,
                    inpainting_extra_input = None
                )


        # - - - - - start sampling - - - - - #
        shape = (real_batch_size, model.in_channels, model.image_size, model.image_size)

        samples_fake = sampler.sample(S=steps, shape=shape, input=input,  uc=uc, guidance_scale=config.guidance_scale, mask=None, x0=None)
        samples_fake = autoencoder.decode(samples_fake)


        # - - - - - save - - - - - #
        for image_id, (sample, meta) in enumerate(zip(samples_fake, metas)):
            img_name = meta['file_name'] + ".jpg"
            sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
            sample = sample.cpu().numpy().transpose(1,2,0) * 255 
            sample = Image.fromarray(sample.astype(np.uint8))
            sample.save(  os.path.join(output_folder_clean, img_name)   )

            for sent_str, bbox in zip(meta['phrases'], meta['locations']):
                x1, y1, x2, y2 = [int(x*512) for x in bbox]
                sample = cv2.rectangle(np.array(sample), (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)
                sample = cv2.putText(sample, sent_str, (x1+5, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)    
            sample = Image.fromarray(sample.astype(np.uint8))
            sample.save(  os.path.join(output_folder_layout, img_name)   )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str,  default="generation_samples", help="root folder for output")
    parser.add_argument("--batch_size", type=int, default=5, help="This will overwrite the one in yaml.")
    parser.add_argument("--no_plms", action='store_true', help="use DDIM instead")
    parser.add_argument("--guidance_scale", type=float,  default=7.5, help="")
    parser.add_argument("-f", "--file", type=str, required=True)
    parser.add_argument("-a1", "--alpha1", type=float, default=0.3, choices=[0.3, 0.5, 0.8, 1.0])
    args = parser.parse_args()
    
    alpha1 = args.alpha1
    alpha2 = 1-alpha1

    file_name= os.path.basename(args.file)
    out_dir = os.path.splitext(file_name)[0].replace("parsed_text_", "")

    with open(args.file, "r") as file:
        input_data = json.load(file)
    data = []
    for d in input_data:
        if os.path.basename(args.file)=='spatial.val.json': # ground truth layout
            img_name = f"{d['id']}_0"
            prompt = d['prompt']
            coord1 = d['obj1'][1]
            coord1 = [coord1[0], coord1[1], coord1[0]+coord1[2], coord1[1]+coord1[3]]
            coord2 = d['obj2'][1]
            coord2 = [coord2[0], coord2[1], coord2[0]+coord2[2], coord2[1]+coord2[3]]
            locations = [coord1, coord2]
            phrases = ["a "+d['obj1'][0], "a "+d['obj2'][0]]
        else:
            img_name = f"{d['query_id']}_{d['iter']}"
            prompt = d['prompt']
            try:
                locations = [loc for _, loc in d['objects']]
                phrases = ["a "+phrase for phrase, _ in d['objects']]
            except:
                locations = [loc for _, loc in d['object_list'] if loc != None]
                phrases = ["a "+phrase for phrase, _ in d['object_list'] if phrase != None]
            if len(phrases) == 0:
                phrases = [""]
                locations = [[0,0,0,0]]

        data.append({
            "ckpt": "gligen_checkpoints/checkpoint_generation_text.pth",
            "prompt": prompt,
            "phrases": phrases,
            "locations": locations,
            "alpha_type": [alpha1, 0.0, alpha2],
            "save_folder_name": out_dir,
            "file_name": img_name,
        })
        
    meta_list = data

    starting_noise = torch.randn(args.batch_size, 4, 64, 64).to(device)
    starting_noise = None
    run(meta_list, args, starting_noise)

    



