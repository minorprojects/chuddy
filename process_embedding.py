import aac_datasets
import numpy as np
import os
import sys
from joblib import Parallel, delayed
from tqdm import tqdm
import torch
import json
import pandas as pd
import argparse

import inspect
import json
import os
import signal
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
import io
import urllib
from typing import Literal

from tqdm import tqdm

import datasets
import PIL.Image
from einops import rearrange
import torch.utils.data
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor

from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent
from resize_right import resize
from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent
from Chuddy.dataset.utils import _Rescale
num_threads = 20
from aac_datasets import AudioCaps
# dset = load_dataset("conceptual_12m")



USER_AGENT = get_datasets_user_agent()


def _fetch_single_image(image_url, timeout=None, retries=0):
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
            break
        except Exception:
            image = None
    return image


def _fetch_images(batch, num_threads, timeout=None, retries=0):
    fetch_single_image_with_args = partial(_fetch_single_image, timeout=timeout, retries=retries)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image"] = list(executor.map(fetch_single_image_with_args, batch["image_url"]))
    return batch


# Load a slightly modified version of the Stable Diffusion pipeline.
# This allows us to extract text embeddings directly (without generating images).
from Chuddy.models.diffusion_pipelines.sd_pipeline import StableDiffusionPipeline
from Chuddy.models.diffusion_pipelines.vd_pipeline import TextToVideoSDPipeline
# from Chuddy.models.diffusion_pipelines.ad_pipeline import AudioLDM2Pipeline



def save_to_path(emb, path):
    """Save embeddings to disk."""
    try:
        with open(path, 'wb') as wf:
            np.save(wf, emb)
    except:
        print("Error with", path)
    return path


if __name__ == '__main__':

    batch_size = 128

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # clip_output_dir = './embed/'
    # synthesize_path = '../data/synthesize_data/synthesize_data.json'

    # video_path = '../data/T-X_pair_data/webvid/webvid.json'
    # audio_path = '../data/T-X_pair_data/audiocap/audiocap.json'
    # img_path = '../data/T-X_pair_data/cc3m/cc3m.json'

    # image_generation_ckpt_path = 'runwayml/stable-diffusion-v1-5'
    # video_generation_ckpt_path = 'cerspense/zeroscope_v2_576w'
    # audio_generation_ckpt_path = 'cvssp/audioldm-l-full'

    # data_path = sys.argv[1]
    modality = sys.argv[1]
    clip_output_dir = sys.argv[2]
    ckpt_path = sys.argv[3]

    if not os.path.exists(clip_output_dir):
        os.makedirs(clip_output_dir, exist_ok=True)

    # Get existing files, so that we don't recompute them.
    existing_files = set([f.strip('.npy') for f in os.listdir(clip_output_dir)])

    caption_list = []
    name_list = []
    if modality == 'audio':
        print('extract audio caption embedding')
        # with open(data_path, 'r', encoding='utf-8') as f:
        #     data = json.load(f)
        dataset = AudioCaps(download=True)
        data = dataset[0]
        for row in tqdm(data, total=len(data)):
            one_audio_name, one_caption = row["audio"], row["captions"]
            if one_audio_name not in existing_files:
                caption_list.append(one_caption)
                name_list.append(one_audio_name)
        pipe = AudioLDM2Pipeline.from_pretrained(ckpt_path, torch_dtype=dtype)
        if not torch.cuda.is_available():
            print('WARNING: using CPU, this will be slow!')
        else:
            pipe = pipe.to("cuda")
    elif modality == 'image':
        print('extract image caption embedding')
        # with open(data_path, 'r', encoding='utf-8') as f:
        #     data = json.load(f)
        dset = load_dataset('conceptual_12m')
        dset = dset.map(_fetch_images, batched=True, batch_size=100, fn_kwargs={"num_threads": num_threads})
        data = dset['train']
        for row in tqdm(data, total=len(data)):
            one_image_name, one_caption = _fetch_single_image(row["image_url"]), row["captions"]
            if one_image_name not in existing_files:
                caption_list.append(one_caption)
                name_list.append(one_image_name)
        pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=dtype)
        if not torch.cuda.is_available():
            print('WARNING: using CPU, this will be slow!')
        else:
            pipe = pipe.to("cuda")
    elif modality == 'video':
        print('extract video caption embedding')
        # with open(data_path, 'r', encoding='utf-8') as f:
        #     data = json.load(f)
        dset = load_dataset('HuggingFaceM4/webvid')
        data = dset['train']
        for row in tqdm(data, total=len(data)):
            one_video_name, one_caption = row["contentUrl"], row["name"]
            if one_video_name not in existing_files:
                caption_list.append(one_caption)
                name_list.append(one_video_name)
        pipe = TextToVideoSDPipeline.from_pretrained(ckpt_path, torch_dtype=dtype)
        if not torch.cuda.is_available():
            print('WARNING: using CPU, this will be slow!')
        else:
            pipe = pipe.to("cuda")

    print('Extract embeddings in batches.')
    num_batches = int(np.ceil(len(caption_list) / batch_size))
    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_captions = caption_list[start_idx:end_idx]
        batch_ids = name_list[8:-4]
        prompt_embeds = pipe(batch_captions, return_prompts_only=True).detach().cpu().numpy()

        # Save embeddings to disk in parallel.
        Parallel(n_jobs=8)(delayed(save_to_path)(
            prompt_embeds[j, :, ...], os.path.join(clip_output_dir, f'{batch_ids[j]}.npy')
        ) for j in range(prompt_embeds.shape[0]))
