#%%

import numpy as np
np.set_printoptions(precision=4)
import pandas as pd

import torch
import torch.nn as nn
from torch.distributions import Categorical

from datetime import datetime
import os
from copy import deepcopy
from random import sample
from collections import namedtuple
from tqdm import tqdm

import matplotlib.pyplot as plt

from typing import List, Dict, Optional, Tuple
import time

par_dir = os.path.abspath(os.path.join(__file__, * [os.pardir] * 2))
src_dir = os.path.join(par_dir, "Src")
result_dir = os.path.join(par_dir, "Result")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

def convert_size(size_info: pd.DataFrame):
    size_info_ = size_info.copy()
    size_info_.loc[size_info_["NET_WGT"] >= 250, "carrier_length"] = 14 # 더블 A 캐리어
    size_info_.loc[size_info_["NET_WGT"] < 250, "carrier_length"] = 11 # A 캐리어

    # 더 긴 쪽을 기준으로 캐리어를 두므로 짧은 쪽의 길이가 Carrier의 길이보다 작으면 Carrier 길이로 변경해준다.
    # 우선 Length 와 Width 중 어디가 더 짧은지 확인한다.
    changed_size = size_info_[["LTH", "BTH"]].values # 우선 원래 값 저장
    shorter_side_idx = np.argmin(changed_size, axis=1)

    # 짧은 쪽의 길이를 변경해야하는 값을 찾아 changed_length로 저장한다
    changed_length = np.max(np.vstack([np.min(changed_size, axis=1), size_info_["carrier_length"].values]).T, axis=1)

    # 어느 방향이 더 짧은지 계산했으니, 해당 경우를 변경된 길이로 변환
    changed_size[np.arange(len(shorter_side_idx)), shorter_side_idx] = changed_length
    # 적치 공간 확보를 위한 여유
    changed_size = np.ceil(changed_size) 
    size_info_[["LTH", "BTH"]] = changed_size
    
    # 그리고 length 기준으로 제일 긴 값이 오게끔 모든 블록 방향 통일
    mask = size_info_["BTH"] > size_info_["LTH"]
    future_len = size_info_.loc[mask, "BTH"].copy().values
    future_width = size_info_.loc[mask, "LTH"].copy().values
    size_info_.loc[mask, "BTH"] = future_width
    size_info_.loc[mask, "LTH"] = future_len
    
    return size_info_[[col for col in size_info_.columns if col != "carrier_length"]]

df = pd.read_csv(os.path.join(src_dir, "sorted_blocks.csv"), encoding="cp949")
del df["Unnamed: 0"]
print(df.columns)
#%%
result = convert_size(df.iloc[:50].copy())
rename_dict = {"PROJ_NO" : "vessel_id", "BLK_NO": "block_id", "LTH": "length", "BTH": "width", "NET_WGT": "weight", "HGT": "height"}
result.rename(columns=rename_dict, inplace=True)

#%%
result = result[list(rename_dict.values())]
result.to_csv(os.path.join(src_dir, "sorted_blocks_processed.csv"), encoding="cp949")
# %%
