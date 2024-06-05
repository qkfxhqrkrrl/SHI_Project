#%%

import numpy as np
np.set_printoptions(precision=4)
import pandas as pd

import torch
import torch.nn as nn
from torch.distributions import Categorical

from datetime import datetime
import os
import shutil
from copy import deepcopy
from random import sample
from collections import namedtuple
from tqdm import tqdm
from glob import glob

import matplotlib.pyplot as plt

from typing import List, Dict, Optional, Tuple
import time
import warnings
warnings.filterwarnings("ignore")
# %%


def convert_size(size_info: pd.DataFrame):
    size_info_ = size_info.copy()
    size_info_.loc[size_info_["weight"] >= 250, "carrier_length"] = 14 # 더블 A 캐리어
    size_info_.loc[size_info_["weight"] < 250, "carrier_length"] = 11 # A 캐리어

    # 더 긴 쪽을 기준으로 캐리어를 두므로 짧은 쪽의 길이가 Carrier의 길이보다 작으면 Carrier 길이로 변경해준다.
    # 우선 Length 와 Width 중 어디가 더 짧은지 확인한다.
    changed_size = size_info_[["length", "width"]].values # 우선 원래 값 저장
    shorter_side_idx = np.argmin(changed_size, axis=1)

    # 짧은 쪽의 길이를 변경해야하는 값을 찾아 changed_length로 저장한다
    changed_length = np.max(np.vstack([np.min(changed_size, axis=1), size_info_["carrier_length"].values]).T, axis=1)

    # 어느 방향이 더 짧은지 계산했으니, 해당 경우를 변경된 길이로 변환
    changed_size[np.arange(len(shorter_side_idx)), shorter_side_idx] = changed_length
    # 적치 공간 확보를 위한 여유
    changed_size = np.ceil(changed_size) 
    size_info_[["length", "width"]] = changed_size
    
    # 그리고 length 기준으로 제일 긴 값이 오게끔 모든 블록 방향 통일
    mask = size_info_["width"] > size_info_["length"]
    future_len = size_info_.loc[mask, "width"].copy().values
    future_width = size_info_.loc[mask, "length"].copy().values
    size_info_.loc[mask, "width"] = future_width
    size_info_.loc[mask, "length"] = future_len
    
    return size_info_[[col for col in size_info_.columns if col != "carrier_length"]]

def generate_empty_slots(mapping_dict):
    
    par_dir = os.path.abspath(os.path.join(__file__, * [os.pardir] * 1))
    data_dir = os.path.join(par_dir, "Data")
    
    empty_slots_dfs = []
    sectors = list(mapping_dict.keys())
    for sect in sectors:
        if sect == "사내": continue
        # n_rand_slots = np.random.randint(10, 20)
        n_rand_slots = np.random.randint(10, 20)
        
        empty_slots = np.zeros((n_rand_slots, 3))
        empty_slots[:, 0] = np.random.randint(24, 30, n_rand_slots)
        empty_slots[:, 1] = np.random.randint(18, 24, n_rand_slots)
        # empty_slots[:, 2] = np.round(np.random.uniform(2, 6, n_rand_slots), 2)
        # empty_slots[:, 2] = np.random.randint(2, 6, n_rand_slots)
        if mapping_dict[sect] == "봉암8" or mapping_dict[sect] == "봉암9":
            empty_slots[:, 2] = 5.8
        elif mapping_dict[sect] == "덕곡2":
            empty_slots[:, 2] = 5
        else:
            empty_slots[:, 2] = 1000
            
        
        empty_slots_smaller_value = np.min(empty_slots[:, :2], axis=-1)
        empty_slots_bigger_value = np.max(empty_slots[:, :2], axis=-1)
        empty_slots[:, 0] = empty_slots_bigger_value # Length 기준으로 긴 길이 정렬
        empty_slots[:, 1] = empty_slots_smaller_value # Width 기준으로 짧은 길이 정렬
        
        empty_slots_df = pd.DataFrame(empty_slots, columns=["length", "width", "height"]).value_counts().reset_index()
        empty_slots_df["location"] = mapping_dict[sect]
        empty_slots_dfs.append(empty_slots_df)
    
    empty_slots_dfs = pd.concat(empty_slots_dfs)
    empty_slots_dfs.to_csv(os.path.join(data_dir, "DB_empty_slots.csv"), index=False, encoding="cp949")

# generate_empty_slots()
def preproc_area_dict(src_dir):
    area_dictionary = pd.read_csv(os.path.join(src_dir, "area_before_proc.csv"), encoding="cp949")
    area_dictionary = area_dictionary.ffill()
    area_dictionary.rename(columns={"사내/외" : "isOut", "부문": "sector", "대지번": "WKA_CD"}, inplace=True)
    area_dictionary.to_csv(os.path.join(src_dir, "area_dict.csv"), encoding="cp949", index=False)
    

def preprocess(par_dir,isExp):
    
    par_dir = os.path.abspath(os.path.join(__file__, * [os.pardir] * 1))
    src_dir = os.path.join(par_dir, "Src")
    input_dir = os.path.join(par_dir, "Input")
    data_dir = os.path.join(par_dir, "Temp")
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
        os.makedirs(data_dir, exist_ok=True)
    else:
        os.makedirs(data_dir, exist_ok=True)
    
    df_db_block = pd.read_csv(os.path.join(input_dir, "DB_whole_block.csv"), encoding="cp949")
    df_db_block.loc[df_db_block["WKA_CD"] == "조립", "WKA_CD"] = df_db_block.loc[df_db_block["WKA_CD"] == "조립", "DTL_ARNO_CD"] # 조립의 경우 이상하게 기록되어있어서 변경
    
    df_db_take_in = pd.read_csv(os.path.join(input_dir, "shi_take_in.csv"), encoding="cp949")
    take_in = convert_size(df_db_take_in)
    take_in.to_csv(os.path.join(data_dir, "temp_take_in.csv"), encoding="cp949", index=False)
    
    df_db_take_out = pd.read_csv(os.path.join(input_dir, "shi_take_out.csv"), encoding="cp949")
    take_out = convert_size(df_db_take_out)
    take_out.to_csv(os.path.join(data_dir, "temp_take_out.csv"), encoding="cp949", index=False)
    
    # 지번 파일 정보 Dictionary로 매칭
    area_dict = pd.read_csv(os.path.join(src_dir, "area_dict.csv"), encoding="cp949")
    area_dict = area_dict.loc[(area_dict["구분"] == "사외"), ["부문", "대지번"]]
    mapping_dict = area_dict.set_index("대지번").to_dict()["부문"]
    mapping_dict["OBI3"] = "오비3"
    mapping_dict["사내"] = "사내"
    inv_mapping_dict = {v: k for k, v in mapping_dict.items()}
    
    # 컬럼명 변경 및 대지번끼리 묶어서 label 변경
    whole_block = df_db_block[["PROJ_NO", "BLK_NO", "LTH", "BTH", "HGT", "NET_WGT", "WKA_CD"]]
    whole_block.loc[~whole_block["WKA_CD"].isin(list(mapping_dict.keys())), "WKA_CD"] = "사내"
    whole_block["WKA_CD"] = whole_block["WKA_CD"].map(mapping_dict)
    whole_block.rename(columns={"PROJ_NO" : "vessel_id", "BLK_NO" : "block", "LTH" : "length", "BTH" : "width", "HGT" : "height", "NET_WGT" : "weight", "WKA_CD" : "location"}, inplace=True)
    whole_block = convert_size(whole_block) # 캐리어 사이즈 반영해주는 함수
    whole_block.to_csv(os.path.join(data_dir, "temp_whole_block.csv"), encoding="cp949", index=False)
    
    if isExp:
        # 반출 블록 랜덤 선정
        n_rand_take_out = np.random.randint(30, 60)
        blocks_at_in = whole_block.loc[(whole_block["location"] == "사내")].copy()
        blocks_at_in = blocks_at_in.iloc[:n_rand_take_out]
        blocks_at_in.to_csv(os.path.join(data_dir, "shi_take_out.csv"), encoding="cp949", index=False)
        
        # 반입 블록 랜덤 선정
        n_rand_take_in = np.random.randint(30, 60)
        blocks_at_out = whole_block.loc[(whole_block["location"] != "사내")].copy()
        blocks_at_out = blocks_at_out.iloc[:n_rand_take_in]
        blocks_at_out.to_csv(os.path.join(data_dir, "shi_take_in.csv"), encoding="cp949", index=False)
    
        # 임의로 빈 슬롯 생성
        generate_empty_slots(mapping_dict) 
    
    # 사용자가 남아있는 슬롯에 대해 기록을 할 때, count로 기록된 수만큼 저장하는 함수
    empty_slots = pd.read_csv(os.path.join(input_dir, "DB_empty_slots.csv"), encoding="cp949")
    yard_out = whole_block.loc[(whole_block["location"] != "사내")].copy()
    
    # for loc in empty_slots["location"].unique():
        # print(loc)
    for loc in whole_block["location"].unique():
        if loc == "사내": continue
        location_slots = whole_block.loc[whole_block["location"] == loc, ["vessel_id", "block", "length", "width", "height", "weight"]].copy()
        location_slots.dropna(inplace=True)
        if empty_slots.loc[empty_slots["location"] == loc].empty:
            yard_slots = location_slots
        else:
            empty_location_slot_extended = []
            # 비어있는 블록은 count 개수 만큼 반복해서 추가해주고 extended에 저장
            empty_location_slot = empty_slots.loc[empty_slots["location"] == loc].copy()
            for idx, row in empty_location_slot.iterrows():
                temp_slots = np.zeros((row["count"], 6))
                temp_slots[:, :2] = np.nan
                temp_slots[:, 2:5] = np.repeat(row[["length", "width", "height"]].values.reshape(-1, 1), row["count"], axis=1).T
                temp_slots[:, 5] = 1000
                empty_location_slot_extended.append(temp_slots)
        
            # 강화학습 알고리즘과 호환되도록 컬럼 명 변경
            empty_location_slot_extended = pd.DataFrame(np.concatenate(empty_location_slot_extended, axis=0), columns=["vessel_id", "block", "width", "length", "height", "weight"])
            yard_slots = pd.concat([location_slots, empty_location_slot_extended])
        
        yard_slots.to_csv(os.path.join(data_dir, f"{loc}.csv"), encoding="cp949", index=False)
        
    # for loc in empty_slots["location"].unique():
    #     # 사용되는 슬롯은 블록의 제원에 크기만큼 사용하고 있으므로 블록 제원 정보 로딩
    #     location_slots = whole_block.loc[whole_block["location"] == loc, ["vessel_id", "block", "length", "width", "height", "weight"]].copy()
    #     location_slots.dropna(inplace=True)
        
    #     empty_location_slot_extended = []
    #     # 비어있는 블록은 count 개수 만큼 반복해서 추가해주고 extended에 저장
    #     empty_location_slot = empty_slots.loc[empty_slots["location"] == loc].copy()
    #     for idx, row in empty_location_slot.iterrows():
    #         temp_slots = np.zeros((row["count"], 6))
    #         temp_slots[:, :2] = np.nan
    #         temp_slots[:, 2:5] = np.repeat(row[["length", "width", "height"]].values.reshape(-1, 1), row["count"], axis=1).T
    #         temp_slots[:, 5] = 1000
    #         empty_location_slot_extended.append(temp_slots)
        
    #     # 강화학습 알고리즘과 호환되도록 컬럼 명 변경
    #     empty_location_slot_extended = pd.DataFrame(np.concatenate(empty_location_slot_extended, axis=0), columns=["vessel_id", "block", "width", "length", "height", "weight"])
        
    #     yard_slots = pd.concat([location_slots, empty_location_slot_extended])
    #     yard_slots.to_csv(os.path.join(data_dir, f"{loc}.csv"), encoding="cp949", index=False)
    