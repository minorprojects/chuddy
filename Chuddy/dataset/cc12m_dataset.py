
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


USER_AGENT = get_datasets_user_agent()


def fetch_single_image(image_url, timeout=None, retries=0):
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


def fetch_images(batch, num_threads, timeout=None, retries=0):
    fetch_single_image_with_args = partial(fetch_single_image, timeout=timeout, retries=retries)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image"] = list(executor.map(fetch_single_image_with_args, batch["image_url"]))
    return batch




class CC12MDataset(BaseDataset):
    def __init__(self, data_path,mm_root_path: str, embed_path: str, train: bool = True, img_transform=None):
        super(CC12MDataset,self).__init__(data_path,mm_root_path,embed_path)
        split = "train" if train else "validation"

        self.urls = data_path[f"{split}"]['image_url']
        self.captions = data_path[f"{split}"]['caption']

        if img_transform is None:
            self.img_transform = Compose([ToTensor(), _Rescale(side_length)])
        else:
            self.img_transform = Compose([ToTensor(), _Rescale(side_length), img_transform])
        self.max_length = max_length

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        self.mm_path_list, self.caption_list = [], []
        img = _fetch_single_image(self.urls[idx])
        if img is None:
            return None
        elif self.img_transform:
            img = self.img_transform(img)

        # Have to check None again because `Resize` transform can return None
        if img is None:
            return None
        elif img.shape[0] != 3:
            return None

        return {'image': img, 'encoding': enc, 'mask': msk}

num_threads = 20
dset = load_dataset("conceptual_12m")
dset = dset.map(fetch_images, batched=True, batch_size=100, fn_kwargs={"num_threads": num_threads})


class CC3MDataset(BaseDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, mm_root_path: str, embed_path: str):
        super(CC3MDataset, self).__init__(data_path, mm_root_path, embed_path)
        self.embed_path = embed_path

        print('Load CC3M dataset ...')
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for row in tqdm(data, total=len(data)):
            image_id, one_caption = row["image_name"], row["caption"]
            self.mm_path_list.append(os.path.join(mm_root_path, image_id))
            self.caption_list.append(process_caption(one_caption))

        print(f'[!] collect {len(self.mm_path_list)} samples for training')
