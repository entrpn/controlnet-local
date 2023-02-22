from __future__ import annotations

import os
import argparse
import pathlib
import random
import shlex
import subprocess
import sys
import json

import numpy as np
from PIL import Image

from model import Model

a_prompt = "best quality, extremely detailed"
negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

base_url = 'https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/'
names = [
    'body_pose_model.pth',
    'dpt_hybrid-midas-501f0c75.pt',
    'hand_pose_model.pth',
    'mlsd_large_512_fp32.pth',
    'mlsd_tiny_512_fp32.pth',
    'network-bsds500.pth',
    'upernet_global_small.pth',
]

for name in names:
    command = f'wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/{name} -O {name}'
    out_path = pathlib.Path(f'ControlNet/annotator/ckpts/{name}')
    if out_path.exists():
        continue
    subprocess.run(shlex.split(command), cwd='ControlNet/annotator/ckpts/')

model = Model()

model_map = {
    "canny" : model.process_canny,
    "hough" : model.process_hough,
    "hed" : model.process_hed,
    "scribble" : model.process_scribble,
    # "scribble_interactive" : model.process_scribble_interactive,
    # "fake_scribble" : model.process_fake_scribble,
    "pose" : model.process_pose,
    "segmentation" : model.process_seg,
    "depth" : model.process_depth,
    "normal_map" : model.process_normal
}

def infer(opt, model):
    img_uri = opt.image_uri
    output_dir = opt.output_dir
    os.makedirs(opt.output_dir,exist_ok=True)
    model_fn = model_map[model]
    img = np.asarray(Image.open(img_uri).convert("RGB"),dtype=np.uint8)
    output_name = img_uri.split("/")[-1].split(".")[0]
    samples = model_fn(
            input_image=img, 
            prompt=opt.prompt,
            a_prompt=a_prompt, 
            n_prompt=negative_prompt, 
            num_samples=opt.num_samples, 
            image_resolution=opt.image_resolution,
            ddim_steps=opt.steps,
            scale=opt.scale,
            seed=opt.seed,
            eta=opt.eta,
            **opt.kwargs)
    i=0
    for sample in samples:
        sample = Image.fromarray(sample)
        sample.save(os.path.join(output_dir,f"{output_name}-{model}-out{i}.png"))
        i+=1

def run(opt):
    if not opt.kwargs:
        opt.kwargs = {}
    
    if opt.model == "all":
        opt.kwargs = {}
        for model in model_map.keys():
            infer(opt, model)
    else:
        infer(opt, opt.model)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-uri",
        type=str,
        required=True,
        help="Local image filepath."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Available models: canny, hough, hed, scribble, pose, segmentation, depth, normal_map and all ( all will run through all models)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=28166885,
        help="Seed"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Steps"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9,
        help="Scale"
    )
    parser.add_argument(
        "--eta",
        type=int,
        default=0,
        help="eta ddim"
    )
    parser.add_argument(
        "--image-resolution",
        type=float,
        default=512,
        help="Image resolution. Ex: 512"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output dir. Defaults to outputs"
    )
    parser.add_argument('--kwargs', 
        type=json.loads,
        required=False,
        help="Pass arguments to different models. Ex: If using canny, pass {'low_threshold' : 100, 'high_threshold'=200}")

    opt = parser.parse_args()
    run(opt)
