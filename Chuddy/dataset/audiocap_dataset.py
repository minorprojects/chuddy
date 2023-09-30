#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import os
import json
from tqdm import tqdm
import ipdb
import random
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence

import torch
import torch.distributed as dist
import transformers
import numpy as np
from torch.utils.data import Dataset
from .base_dataset import BaseDataset
from tqdm import tqdm
import pandas as pd
from .utils import process_caption
from acc_datasets import AudioCaps

class AudioCapDataset(BaseDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, mm_root_path: str, embed_path: str):
        super(AudioCapDataset, self).__init__(data_path, mm_root_path, embed_path)
        self.embed_path = embed_path

        print('Load Audiocap dataset ...')
        self.mm_path_list, self.caption_list = [], []
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for row in tqdm(data, total=len(data)):
            audio_id, one_caption = row["audio_name"], row["caption"]
            self.mm_path_list.append(os.path.join(mm_root_path, audio_id))
            self.caption_list.append(process_caption(one_caption))

        print(f'[!] collect {len(self.mm_path_list)} samples for training')

class AudioDataset(torch.utils.data.DataLoader):
    def __init__(self, data_path,mm_root_path: str, embed_path: str):
        super(AudioDataset,self).__init__()
        # split = "train" if train else "validation"
        data = data_path[0]
        self.audio = data_path['audio']
        self.captions = data_path['captions']

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        with open(os.path.join(self.embed_path, str(os.path.basename(img)) + '.npy'), 'rb') as f:
            caption_embs = torch.from_numpy(np.load(f, allow_pickle=True)) 

        return dict(mm_paths=self.audio[idx], output_texts=self.captions[idx], caption_embs=caption_embs)

dataset = AudioCaps(root=".", download=True)
