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
from glob import glob

import matplotlib.pyplot as plt

from typing import List, Dict, Optional, Tuple
import time
import warnings
from preproc_data import preprocess
warnings.filterwarnings("ignore")

torch.autograd.set_detect_anomaly(True)


class Args:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.save_dir = None
        self.load_dir = None
        self.isExp = None
        
        self.barge_batch_size = None
        self.n_yard = None
        self.n_max_block = None
        self.n_take_in = None
        self.n_take_out = None
        
        self.batch_size = None
        self.hid_dim = None
        self.decay = None
        self.pad_len = None
        self.n_epoch = None
        self.exp_num = None


def make_input_from_loaded(blocks_to_be_out, blocks_to_be_in, env):
    encoder_inputs, yard_slots, remaining_areas = env.get_state(blocks_to_be_out.copy(), blocks_to_be_in.copy())
    return encoder_inputs, remaining_areas, yard_slots

class Yard:
    def __init__(self, name : str, area_slots : Dict, blocks : pd.DataFrame = None):
        self.name = name
        self.block_count_limit = deepcopy(area_slots)
        self.available_blocks_count = area_slots
        self.blocks_by_size : Dict[str, pd.DataFrame] = {}
        self.blocks : pd.DataFrame = blocks
    
    def move_block_to(self, block_ids : List[int], yard :"Yard"):
        pass
        # yard.blocks[] = self.blocks[name]
            
    def __getitem__(self, item):
        if self.blocks is None:
            raise Exception("Yard does not have blocks")
        # 옮기는 데는 사용하지 말고, 확인 용도로 사용
        return self.blocks.iloc[item]
        
        
class YardInside(Yard):
    def __init__(self, name: str, area_slots : Dict, blocks : pd.DataFrame = None):
        super().__init__(name, area_slots, blocks)
        # TODO: 현재 사내는 고려를 크게 안 하는 중
            
class YardOutside(Yard):
    def __init__(self, args : Args, name: str, area_slots : pd.DataFrame, blocks : pd.DataFrame = None):
        super().__init__(name, area_slots, blocks)
        if args.isExp:
            self.orig_slots = area_slots.copy()
            self.orig_slots["id"] = range(len(area_slots))
            
            self.area_slots = area_slots[["vessel_id", "block", "length", "width_max","height", "location"]].rename(columns={"width_max": "width"})
            self.area_slots["id"] = range(len(area_slots))
            
            self.blocks["location"] = self.name
            self._update_slot()
            
            self.first_step_infos = {
                "area_slots": deepcopy(self.area_slots), 
                "blocks": deepcopy(self.blocks)
            }
            
        else :
            self.area_slots = area_slots
        self.columns = self.area_slots.columns
        self.used_area = np.sum(self.area_slots["length"] * self.area_slots["width"])
        
    def _reset(self):
        self.area_slots = deepcopy(self.first_step_infos["area_slots"])
        self.blocks = deepcopy(self.first_step_infos["blocks"])
        
    def _load_env(self):
        self.first_step_infos = {
            "area_slots": deepcopy(self.area_slots), 
            "blocks": deepcopy(self.blocks)
        }
    
    def move_to_yard_out(self, blocks: pd.DataFrame):
        self.blocks = pd.concat([self.blocks, blocks]).reset_index(drop=True)
        self.blocks["location"] = self.name
        for idx, block in blocks.iterrows():
            count = self.area_slots.loc[self.area_slots["vessel_id"].isnull(), ["length", "width", "height", "location"]].groupby(["length", "width"], as_index=False).value_counts().values
            fitting_slot = np.argmin(np.abs((block["length"] * block["width"]) - (count[:, 0] * count[:, 1])))
            length, width = count[fitting_slot, 0], count[fitting_slot, 1]
            slot_idx = self.area_slots[(self.area_slots["vessel_id"].isnull()) & (self.area_slots["length"] == length) & (self.area_slots["width"] == width)].index[0]
            self.area_slots.loc[self.area_slots.index[slot_idx], "block"] = block["block"]
            self.area_slots.loc[self.area_slots.index[slot_idx], "vessel_id"] = block["vessel_id"]
        
    def move_to_yard_in(self, blocks: pd.DataFrame):
        for _, block in blocks.iterrows():
            vessel_id, block_id = block["vessel_id"] ,block["block"]
            self.blocks = self.blocks.loc[~((self.blocks["vessel_id"] == vessel_id) & (self.blocks["block"] == block_id)), :]
            self.area_slots.loc[~((self.area_slots["vessel_id"] == vessel_id) & (self.area_slots["block"] == block_id)), ["vessel_id", "block"]] = np.nan
        
    def _check_information(self):
        print("Area slots: \n", self.area_slots)
    
    def _update_slot(self):
        # ! 현 성과 공유에서는 제외
        # 슬롯의 크기와 들어가 있는 블록의 크기를 비교해서 남는 공간을 새로운 슬롯으로 변경
        # 남는 공간은 (remaining_width * length)와 (width * remaining_length)큰 공간을 기준으로 선정
        df_merged = pd.merge(self.area_slots, self.blocks , on=["vessel_id", "block"], suffixes=("_slots","_block"), how="outer")
        # display(df_merged)
        df_merged[["remaining_width", "remaining_length"]] = df_merged[["width_slots", "length_slots"]].values - df_merged[["width_block", "length_block"]].values
        df_merged["isFull"] = np.any(df_merged[["remaining_width", "remaining_length"]].values > 11, axis=1) # 캐리어로 인해 11은 확보가 되어야 함
        df_merged["area_length"] = df_merged["remaining_width"] * df_merged["length_block"]
        df_merged["area_width"] = df_merged["width_slots"] * df_merged["remaining_length"]
        viable_condition = (df_merged["isFull"] == True) & (~df_merged["vessel_id"].isnull())
        
        new_slots = df_merged.loc[viable_condition].copy()
        condition_new = (new_slots["area_length"] > new_slots["area_width"])
        new_slots.loc[condition_new, "width_slots"] = new_slots.loc[condition_new, "remaining_width"]
        new_slots.loc[~condition_new, "length_slots"] = new_slots.loc[~condition_new, "remaining_length"]
        new_slots[["vessel_id", "block", "weight"]] = np.nan
        
        changeWidth = df_merged["area_length"] > df_merged["area_width"]
        df_merged.loc[viable_condition & changeWidth, "width_slots"] = df_merged.loc[viable_condition & changeWidth, "width_block"]
        df_merged.loc[viable_condition & ~changeWidth, "length_slots"] = df_merged.loc[viable_condition & ~changeWidth, "length_block"]
        # length 기준으로 한 너비가 충분히 크면, length 방향으로 정렬하고 width 방향을 공간으로 둠
        
        df_merged = pd.concat([df_merged, new_slots], axis=0)[["id","vessel_id", "block", "width_slots", "length_slots", "height_slots", "weight"]].reset_index(drop=True)
        df_merged = df_merged.rename(columns={"width_slots": "width", "length_slots": "length", "height_slots": "height"})
        # else:
        #     df_merged = pd.concat([df_merged, new_slots], axis=0)[["id","vessel_id", "block", "width_slots", "length_slots", "height_slots", "weight_block"]].reset_index(drop=True)
        #     df_merged = df_merged.rename(columns={"width_slots": "width", "length_slots": "length", "height_slots": "height"})
        
        # length 기준으로 더 긴 값이 오도록 하고 있으므로 값 바꿔주기
        max_vals = np.max(df_merged[["length", "width"]].values, axis=1)
        min_vals = np.min(df_merged[["length", "width"]].values, axis=1)
        df_merged["length"] = max_vals
        df_merged["width"] = min_vals
        
        self.area_slots = df_merged
        
        # 같은 아이디로 나눠졌던 공간이 다 비어 있으면 하나로 다시 합쳐주기
        for id, idxs in self.area_slots.groupby("id").groups.items():
            if np.all(self.area_slots.iloc[idxs]["id"].isnull()):
                self.area_slots = self.area_slots.loc[self.area_slots["id"] != id] 
                self.area_slots.loc[len(self.area_slots.index)] = self.orig_slots.loc[self.orig_slots["id"] == id]["vessel_id", "block", "length", "width_max","height"].values        
        
    def calc_remaining_area(self):
        taken_area = np.sum(self.blocks["length"] * self.blocks["width"])
        possible_area = np.sum(self.orig_slots["length"] * self.orig_slots["width_max"])
        # TODO: 원래 하나의 칸이였던 경우를 블록이 비어서 없어지는 경우를 고려하기 위해서 여유가 되면 id별로 다시 합쳐주기 
        return possible_area, taken_area
    
    def calc_remaining_area_by_slot(self):
        remaining_area = []
        for slot_info in self.area_slots.loc[~(self.area_slots["vessel_id"].isnull()),["vessel_id", "block","length", "width"]].values:
            taken_space = self.blocks.loc[(self.blocks["vessel_id"] == slot_info[0]) & (self.blocks["block"] == slot_info[1]), ["length", "width"]].values[0]
            remaining_area.append(((slot_info[2] * slot_info[3]) - (taken_space[0] * taken_space[1])) ** 2)
        # TODO: 원래 하나의 칸이였던 경우를 블록이 비어서 없어지는 경우를 고려하기 위해서 여유가 되면 id별로 다시 합쳐주기 
        return np.sum(remaining_area)

        

class PGAgent(nn.Module):
    def __init__(self, args: Args) -> None:
        super(PGAgent, self).__init__()
        self.alpha = args.alpha
        self.beta = args.beta
        self.max_trip = args.max_trip
        self.barge_max_row = args.barge_max_row
        self.barge_par_width = args.barge_par_width
        self.barge_max_length = args.barge_max_length
        self.hid_dim = args.hid_dim
        self.batch_size = args.batch_size
        self.pad_len = args.pad_len
        self.n_yard = args.n_yard
        self.n_max_block = args.n_max_block
        self.n_trip_infos_list = args.n_trip_infos_list
        self.priori_conditions = args.priori_conditions 
        
        self.n_take_in = args.n_take_in
        self.n_take_out = args.n_take_out
        
        # self.encoder = nn.LSTM(input_size=5*(args.pad_len+1), hidden_size=args.hid_dim, batch_first=True) # ! 제거된 기능
        self.to_decoder = nn.LSTMCell(input_size=5*(args.pad_len+1), hidden_size=args.hid_dim)
        self.to_fc_block = nn.Linear(args.hid_dim, self.n_max_block*self.n_yard) # 400 = 40(n_block) * 10(n_yard)
        
        self.ti_decoder = nn.LSTMCell(input_size=5*(args.pad_len+1), hidden_size=args.hid_dim)
        self.ti_fc_block = nn.Linear(args.hid_dim, self.n_max_block)
        # self.emb_yard = nn.Embedding(self.n_yard, 5*(args.pad_len+1)) # ! 미구현
        
        self.device = args.device
        self.batch_limit = args.barge_batch_size
        # TODO: 제원 정보 임베딩으로 변환하는 방법 적용 고려해보기
        
        self.soft_max = nn.Softmax(dim=-1)
    
    def init_env_info(self, infos):
        # 위에서 Init을 안 하는 이유는 이건 Env.reset과 함께 다시 초기화되어야 하는 값이기 때문
        take_in, take_out, yard_slots = infos
        
        input0 = torch.zeros(self.batch_size, 5*(self.pad_len+1)).to(self.device)
        h0, c0 = torch.zeros((self.batch_size, self.hid_dim), device=self.device), torch.zeros((self.batch_size, self.hid_dim), device=self.device)
        
        self.n_trip_infos_tensor = torch.FloatTensor(self.n_trip_infos_list).to(self.device)
        self.n_trip_infos_tensor = self.n_trip_infos_tensor.repeat(self.batch_size, 1)
        yard_slots = np.expand_dims(yard_slots, axis=0).repeat(self.batch_size, axis=0)
        self.yard_slots = torch.FloatTensor(yard_slots).to(self.device)
        
        self.priori_index = []
        if self.priori_conditions is not None:
            for row in self.priori_conditions:
                block_idx, yard_idx = row
                for i in range(self.n_yard):
                    if i == yard_idx: continue
                    self.priori_index.append(block_idx * self.n_yard + i)
        
        # _______________________________ Take out _______________________________
        self.block_size_info_to = torch.FloatTensor(np.expand_dims(take_out[["length", "width", "height", "weight", "location"]].values, axis=0).repeat(self.batch_size, axis=0)).to(self.device)
        self.encoder_mask_whole_to = self.generate_mask(take_out, yard_slots)
        self.block_mask_to = torch.zeros((self.batch_size, self.n_take_out, 1), dtype=bool) # 제외 시키고 싶은 경우를 True
        
        self.barge_count_to = torch.zeros((self.batch_size, self.n_yard), dtype=torch.int64, device=args.device)
        self.barge_slot_to = -1 * torch.ones((self.batch_size, self.n_yard, self.max_trip, self.barge_max_row, 2), dtype=torch.int64, device=args.device) # 블록 인덱스 저장
        self.block_lengths_on_barge_to = torch.zeros((self.batch_size, self.n_yard, self.max_trip, self.barge_max_row), dtype=torch.float32, device=args.device) # 블록 길이 저장
        self.block_width_on_barge_to = torch.zeros((self.batch_size, self.n_yard, self.max_trip, self.barge_max_row), dtype=torch.float32, device=args.device) # 블록 너비 저장
        
        
        self.probs_history_to = []
        self.rewards_history_to = []
        self.action_history_to = []
        self.reward_parts = torch.zeros((self.batch_size, self.n_take_out), dtype=float).to(self.device)
        self.reward_spaces_to = torch.zeros((self.batch_size, self.n_take_out), dtype=float).to(self.device)
        
        # _______________________________ Take In _______________________________
        self.block_size_info_ti = torch.FloatTensor(np.expand_dims(take_in[["length", "width", "height", "weight", "location"]].values, axis=0).repeat(self.batch_size, axis=0)).to(self.device)
        self.block_mask_ti = torch.zeros((self.batch_size, self.n_take_in), dtype=bool) # 제외 시키고 싶은 경우를 True
        
        self.barge_count_ti = torch.zeros((self.batch_size, self.n_yard), dtype=torch.int64, device=args.device)
        self.barge_slot_ti = -1 * torch.ones((self.batch_size, self.n_yard, self.max_trip, self.barge_max_row, 2), dtype=torch.int64, device=args.device) # 블록 인덱스 저장
        self.block_lengths_on_barge_ti = torch.zeros((self.batch_size, self.n_yard, self.max_trip, self.barge_max_row), dtype=torch.float32, device=args.device) # 블록 길이 저장
        self.block_width_on_barge_ti = torch.zeros((self.batch_size, self.n_yard, self.max_trip, self.barge_max_row), dtype=torch.float32, device=args.device) # 블록 너비 저장
        
        self.probs_history_ti = []
        self.rewards_history_ti = []
        self.action_history_ti = []
        
        return input0, h0, c0
        
    def generate_mask(self, take_out: pd.DataFrame, yard_slots: np.ndarray):
        encoder_mask = np.zeros((self.batch_size, take_out.shape[0], self.n_yard, self.pad_len), dtype=bool)
        
        for b_idx, block_info in enumerate(take_out[["length", "width", "height"]].values):
            mask = np.all(block_info <= yard_slots[:, :, :, :3], axis=-1)# 블록 크기가 슬롯의 모든 면보다 작으면 True, 블록이 슬롯보다 작은게 하나라도 있다면 True
            encoder_mask[:, b_idx] = mask # 
        
        # 불가능한 경우에 값들을 변경해야 하므로 위에서 구한 가능한 Case 들을 False가 되도록 Not 적용
        return torch.BoolTensor(encoder_mask).to(self.device)
    
    
    def allocated_to_barge(self, block_selection, block_size, yard_selection, barge_infos):
        barge_count ,barge_slot, block_lengths_on_barge, block_width_on_barge = barge_infos
        
        cur_barge_num = barge_count[range(self.batch_size), yard_selection] # [B]
    
        cur_barge_total_length = torch.sum(block_lengths_on_barge[range(self.batch_size), yard_selection, cur_barge_num], dim=-1) # [B]: 배치별로 선택된 야드의 현재 바지선에 배정된 블록들의 길이
        barge_parallel_space = block_width_on_barge[range(self.batch_size), yard_selection, cur_barge_num] <= self.barge_par_width # [B R]: 배치별로 선택된 야드의 현재 바지선에 중에서 병렬이 가능한 슬롯에 대한 진위 여부
        is_barge_slot_parallel_space = block_width_on_barge[range(self.batch_size), yard_selection, cur_barge_num] <= self.barge_par_width # [B R]: 배치별로 선택된 야드의 현재 바지선에 중에서 병렬이 가능한 슬롯에 대한 진위 여부
        does_barge_have_parallel_space = torch.any(is_barge_slot_parallel_space, dim=-1) # [B] : 해당 배치에 공간이 충분한 slot이 있는 바지가 있는지 확인 여부
        is_possible = cur_barge_total_length <= self.barge_max_length # [B]: 배치별로 선택된 야드의 현재 바지선에 배정된 블록들의 길이, 가능한 경우는 True로 불가능한 경우는 False로 
    
        block_length_b = block_size[:, 0, 0] # Unsqueeze in [1]
        block_width_b = block_size[:, 0, 1] # [B]
    
    
        is_parallel_block = block_width_b <= self.barge_par_width
        cur_block_lengths_on_barge = block_lengths_on_barge[range(self.batch_size), yard_selection, cur_barge_num]
        
        is_parallel_block = block_width_b <= self.barge_par_width # [B]
        
        "Looping through the parallel block by batch, if batch have parallel space then allocate it by the rule"
        
        # print(does_have_existing_on_par_slot.shape)
        
        # ! block: parallel, slot: have_space is decided here
        is_parallel_and_allocable = is_parallel_block &  does_barge_have_parallel_space # [B]
        par_b_idxs = is_parallel_and_allocable.nonzero().reshape(-1)
        par_y_idxs = yard_selection[is_parallel_and_allocable]
        par_s_idxs = cur_barge_num[is_parallel_and_allocable]
        
        slots_with_existing_block_but_more_space = block_width_on_barge[(par_b_idxs, par_y_idxs, par_s_idxs)].clone()
        slots_with_existing_block_but_more_space[torch.logical_not(is_barge_slot_parallel_space[is_parallel_and_allocable])] = -1 # [B R]: 병렬 배치 제외
        slots_with_existing_block_but_more_space[slots_with_existing_block_but_more_space == 0] = -1 # 비어있는 경우 제외 
        
        ###### existing_block: is decided here
        batch_with_existing_block_but_more_space = torch.any(slots_with_existing_block_but_more_space != -1, dim=-1)
        target_par_b_idxs = par_b_idxs[batch_with_existing_block_but_more_space]
        target_par_y_idxs = par_y_idxs[batch_with_existing_block_but_more_space]
        target_par_s_idxs = par_s_idxs[batch_with_existing_block_but_more_space]
        
        parallel_block_length_on_barge_batch = block_lengths_on_barge[(par_b_idxs, par_y_idxs, par_s_idxs)].clone() # [B R 1]
        parallel_block_length_on_barge_batch[torch.logical_not(is_barge_slot_parallel_space[is_parallel_and_allocable])] = -1 # [B R]
        
        maxs = torch.max(parallel_block_length_on_barge_batch, dim=-1)
        max_idxs, max_val = maxs.indices, maxs.values
        
        # * block: parallel, slot: have_space, length: Check if it is smaller than existing, existing_block: True
        directly_allocable_slot = max_val[batch_with_existing_block_but_more_space] >= block_length_b[target_par_b_idxs]
        dir_alloc_b_idxs = target_par_b_idxs[directly_allocable_slot]
        dir_alloc_y_idxs = target_par_y_idxs[directly_allocable_slot]
        dir_alloc_s_idxs = target_par_s_idxs[directly_allocable_slot]
        dir_alloc_slot_idxs = max_idxs[batch_with_existing_block_but_more_space][directly_allocable_slot]
        
        barge_slot[dir_alloc_b_idxs, dir_alloc_y_idxs, dir_alloc_s_idxs, dir_alloc_slot_idxs, 1] = block_selection[dir_alloc_b_idxs]
        block_width_on_barge[dir_alloc_b_idxs, dir_alloc_y_idxs, dir_alloc_s_idxs, dir_alloc_slot_idxs] += block_width_b[dir_alloc_b_idxs]
        # print(block_width_on_barge[0])
        
        # ! length: If it is not smaller than existing, it need reconsideration
        need_reconsider_slot = torch.logical_not(directly_allocable_slot)
        recons_b_idxs = target_par_b_idxs[need_reconsider_slot]
        recons_y_idxs = target_par_y_idxs[need_reconsider_slot]
        recons_s_idxs = target_par_s_idxs[need_reconsider_slot]
        recons_slot_idxs = max_idxs[batch_with_existing_block_but_more_space][need_reconsider_slot]
        recons_max_val = max_val[batch_with_existing_block_but_more_space][need_reconsider_slot]
        
        # * block: parallel, slot: have_space, length: Adding block doesn't matter, existing_block: True
        recons_total_block_length = torch.sum(block_lengths_on_barge[recons_b_idxs, recons_y_idxs, recons_s_idxs], dim=-1)
        recons_allocable = recons_total_block_length - recons_max_val + block_length_b[recons_b_idxs] <= self.barge_max_length
        
        recons_allocable_b_idxs = recons_b_idxs[recons_allocable]
        recons_allocable_y_idxs = recons_y_idxs[recons_allocable]
        recons_allocable_s_idxs = recons_s_idxs[recons_allocable]
        recons_allocable_slot_idxs = recons_slot_idxs[recons_allocable]
        
        barge_slot[recons_allocable_b_idxs, recons_allocable_y_idxs, recons_allocable_s_idxs, recons_allocable_slot_idxs, 1] = block_selection[recons_allocable_b_idxs]
        block_lengths_on_barge[recons_allocable_b_idxs, recons_allocable_y_idxs, recons_allocable_s_idxs, recons_allocable_slot_idxs] = block_length_b[recons_allocable_b_idxs]
        block_width_on_barge[recons_allocable_b_idxs, recons_allocable_y_idxs, recons_allocable_s_idxs, recons_allocable_slot_idxs] += block_width_b[recons_allocable_b_idxs]
        
        # * block: parallel, slot: have_space, length: Adding block does matter, so move to next barge, existing_block: True
        recons_non_allocable_b_idxs = recons_b_idxs[torch.logical_not(recons_allocable)]
        recons_non_allocable_y_idxs = recons_y_idxs[torch.logical_not(recons_allocable)]
        recons_non_allocable_s_idxs = recons_s_idxs[torch.logical_not(recons_allocable)] + 1
        
        barge_count[(recons_non_allocable_b_idxs, recons_non_allocable_y_idxs)] += 1
        barge_slot[recons_non_allocable_b_idxs, recons_non_allocable_y_idxs, recons_non_allocable_s_idxs, 0, 0] = block_selection[recons_non_allocable_b_idxs]
        block_lengths_on_barge[recons_non_allocable_b_idxs, recons_non_allocable_y_idxs, recons_non_allocable_s_idxs, 0] = block_length_b[recons_non_allocable_b_idxs]
        block_width_on_barge[recons_non_allocable_b_idxs, recons_non_allocable_y_idxs, recons_non_allocable_s_idxs, 0] += block_width_b[recons_non_allocable_b_idxs]
        
        # * block: parallel, slot: have_space, length: fine, existing_block: False --> same as plain
        parallel_block_but_not_existing = torch.logical_not(batch_with_existing_block_but_more_space)
        par_plain_b_idxs = par_b_idxs[parallel_block_but_not_existing]
        par_plain_y_idxs = par_y_idxs[parallel_block_but_not_existing]
        par_plain_s_idxs = par_s_idxs[parallel_block_but_not_existing]
        
        par_plain_block_length_on_barge_batch = block_lengths_on_barge[(par_plain_b_idxs, par_plain_y_idxs, par_plain_s_idxs)] # [B R 1]
        par_plain_barge_slot = barge_slot[(par_plain_b_idxs, par_plain_y_idxs, par_plain_s_idxs)]  # [B R 2]
        
        par_plain_total_length_on_barge = torch.sum(par_plain_block_length_on_barge_batch, dim=-1) # [B]
        does_barge_have_par_plain_space = par_plain_total_length_on_barge + block_length_b[par_plain_b_idxs] <= self.barge_max_length 
        does_barge_have_par_plain_slots = torch.any(par_plain_barge_slot[:, :, 0] == -1, dim=-1) # 하나라도(any) slot이 비어있으면(==-1) True가 반환되겠지
        
        allocate_directly_mask = torch.logical_and(does_barge_have_par_plain_space, does_barge_have_par_plain_slots)
        slot_idxs = torch.max(par_plain_barge_slot[allocate_directly_mask, :, 0] == -1, dim=-1).indices
        
        par_plain_cur_b_idxs = par_plain_b_idxs[allocate_directly_mask]
        par_plain_cur_y_idxs = par_plain_y_idxs[allocate_directly_mask]
        par_plain_cur_s_idxs = par_plain_s_idxs[allocate_directly_mask]
        
        barge_slot[par_plain_cur_b_idxs, par_plain_cur_y_idxs, par_plain_cur_s_idxs, slot_idxs, 0] = block_selection[par_plain_cur_b_idxs]
        block_lengths_on_barge[par_plain_cur_b_idxs, par_plain_cur_y_idxs, par_plain_cur_s_idxs, slot_idxs] = block_length_b[par_plain_cur_b_idxs]
        block_width_on_barge[par_plain_cur_b_idxs, par_plain_cur_y_idxs, par_plain_cur_s_idxs, slot_idxs] += block_width_b[par_plain_cur_b_idxs]
        
        next_barge_mask = torch.logical_not(allocate_directly_mask)
        b_idxs = par_plain_b_idxs[next_barge_mask]
        y_idxs = par_plain_y_idxs[next_barge_mask]
        s_idxs = par_plain_s_idxs[next_barge_mask]
        barge_count[(b_idxs, y_idxs)] += 1
        barge_slot[(b_idxs, y_idxs, s_idxs+1, 0, 0)] = block_selection[b_idxs]
        block_lengths_on_barge[(b_idxs, y_idxs, s_idxs+1, 0)] = block_length_b[b_idxs]
        block_width_on_barge[(b_idxs, y_idxs, s_idxs+1, 0)] = block_width_b[b_idxs]
        
        # ! plain blocks
        
        plain_allocable = torch.logical_not(is_parallel_and_allocable) # [B]
        plain_b_idxs = plain_allocable.nonzero().reshape(-1)
        plain_y_idxs = yard_selection[plain_allocable]
        plain_s_idxs = cur_barge_num[plain_allocable]
        # plain_block_length_on_barge_batch = torch.concat([block_lengths_on_barge[b_idx, y_idx, s_idx] for b_idx, y_idx, s_idx in zip(plain_b_idxs, plain_y_idxs, plain_s_idxs)]).unsqueeze(-1) # [B R 1]
        plain_block_length_on_barge_batch = block_lengths_on_barge[(plain_b_idxs, plain_y_idxs, plain_s_idxs)] # [B R 1]
        plain_barge_slot = barge_slot[(plain_b_idxs, plain_y_idxs, plain_s_idxs)]  # [B R 2]
        
        # 슬롯이 꽉 찼는지 확인
        plain_total_length_on_barge = torch.sum(plain_block_length_on_barge_batch, dim=-1) # [B]
        does_barge_have_plain_space = plain_total_length_on_barge + block_length_b[plain_b_idxs] <= self.barge_max_length 
        does_barge_have_plain_slots = torch.any(plain_barge_slot[:, :, 0] == -1, dim=-1) # 하나라도(any) slot이 비어있으면(==-1) True가 반환되겠지
        
        allocate_directly_mask = torch.logical_and(does_barge_have_plain_space, does_barge_have_plain_slots)
        slot_idxs = torch.max(plain_barge_slot[allocate_directly_mask, :, 0] == -1, dim=-1).indices
        b_idxs = plain_b_idxs[allocate_directly_mask]
        y_idxs = plain_y_idxs[allocate_directly_mask]
        s_idxs = plain_s_idxs[allocate_directly_mask]
        barge_slot[(b_idxs, y_idxs, s_idxs, slot_idxs, torch.zeros_like(slot_idxs))] = block_selection[b_idxs]
        block_lengths_on_barge[(b_idxs, y_idxs, s_idxs, slot_idxs)] = block_length_b[b_idxs]
        block_width_on_barge[(b_idxs, y_idxs, s_idxs, slot_idxs)] = block_width_b[b_idxs]
        
        next_barge_mask = torch.logical_not(allocate_directly_mask)
        b_idxs = plain_b_idxs[next_barge_mask]
        y_idxs = plain_y_idxs[next_barge_mask]
        s_idxs = plain_s_idxs[next_barge_mask]
        barge_count[(b_idxs, y_idxs)] += 1
        barge_slot[(b_idxs, y_idxs, s_idxs+1, 0, 0)] = block_selection[b_idxs]
        block_lengths_on_barge[(b_idxs, y_idxs, s_idxs+1, 0)] = block_length_b[b_idxs]
        block_width_on_barge[(b_idxs, y_idxs, s_idxs+1, 0)] = block_width_b[b_idxs]
        
        return barge_count ,barge_slot, block_lengths_on_barge, block_width_on_barge
    
    
    def forward(self, infos: List[pd.DataFrame], encoder_inputs : torch.Tensor, remaining_area = None):
        """
        Feature information
        take_out, take_in, feat_vec: ["length", "width", "height", "weight", "location"]
        slot_info: [length  width  height  location  count]
        """
        input0, h0, c0 = self.init_env_info(infos)
        
        for idx in range(self.n_take_in):
            h0, c0 = self.ti_decoder(input0, (h0, c0))
            out = self.ti_fc_block(h0)
            
            out[:, self.n_take_in:] = -20000
            out[:, :self.n_take_in][self.block_mask_ti] = -20000
            
            probs = self.soft_max(out)
            m = Categorical(probs)
            action = m.sample()
            
            block_size = self.block_size_info_ti[range(self.batch_size), action]
            
            yard = block_size[:, -1] -1
            yard = yard.type(torch.LongTensor).to(self.device)
            slot_info = torch.stack([self.yard_slots[batch_idx, yard_idx] for batch_idx, yard_idx in zip(range(self.batch_size), yard)], dim=0)
            
            input0 = torch.cat([block_size.unsqueeze(1), slot_info], dim=1).reshape(self.batch_size, -1).to(self.device)
            
            self.barge_count_ti, self.barge_slot_ti, self.block_lengths_on_barge_ti, self.block_width_on_barge_ti = \
                self.allocated_to_barge(block_selection=action, yard_selection=yard, block_size=block_size.unsqueeze(1), 
                                        barge_infos=(self.barge_count_ti, self.barge_slot_ti, self.block_lengths_on_barge_ti, self.block_width_on_barge_ti))
            
            barge_with_blocks = torch.any(torch.any(self.barge_slot_ti != -1, dim=-1), dim=-1)
            barge_num_by_yard = torch.sum(barge_with_blocks, dim=-1) # [B Y]
            total_barge_num = torch.sum(barge_num_by_yard, dim=-1) # [B]
            
            mask_wrong_barge_selection = torch.any(barge_num_by_yard > self.n_trip_infos_tensor, dim=-1)
            penalty_for_wrong_barge_selection = (block_size[:, 0] * block_size[:, 1]).reshape(-1) * (torch.where(mask_wrong_barge_selection, -1, 0))
            
            self.probs_history_ti.append(m.log_prob(action))
            self.action_history_ti.append(action)
            self.rewards_history_ti.append(self.beta * total_barge_num + penalty_for_wrong_barge_selection)
            
            # Gradient Error 때문에 매번 새로운 객체 형성 후 대체
            new_block_mask = self.block_mask_ti.clone()
            new_block_mask[range(self.batch_size), action] = True
            self.block_mask_ti = new_block_mask
            
        self.probs_history_ti = torch.stack(self.probs_history_ti).transpose(1, 0).to(self.device)
        self.rewards_history_ti = torch.stack(self.rewards_history_ti).transpose(1, 0).to(self.device)
        self.action_history_ti = torch.stack(self.action_history_ti).transpose(1, 0).to(self.device)
        
        for idx in range(self.n_take_out):
            
            # ________________________________________________
            encoder_mask = ~torch.any(self.encoder_mask_whole_to, dim=-1)
            h0, c0 = self.to_decoder(input0, (h0, c0))
            out = self.to_fc_block(h0)
            # ________________________________________________
            out[:, self.n_yard*self.n_take_out:] = -20000 # 블록 개수보다 많을 필요는 없으니 마스킹
            # out[:, :self.n_yard*self.n_take_out][block_mask.repeat(1, 1, self.n_yard).reshape(batch_size, -1)] = -10000 # 선택된 블록이 다시 선택되지 않도록 마스킹
            out[:, :self.n_yard*self.n_take_out][encoder_mask.reshape(self.batch_size, -1)] = -10000 # 보내고자 하는 블록이 적치장에 안 맞으면 마스킹, 선택이 안 됐는데 적치장에 보낼 수는 있도록
            out[:, :self.n_yard*self.n_take_out][self.block_mask_to.repeat(1, 1, self.n_yard).reshape(self.batch_size, -1)] = -15000 # 선택된 블록은 선택 안 되도록
            out[:, self.priori_index] = -15000
            
            feasible_mask = torch.ones_like(out, dtype=torch.bool)
            feasible_mask[:, self.n_yard*self.n_take_out:] = False
            feasible_mask[:, :self.n_yard*self.n_take_out][self.block_mask_to.repeat(1, 1, self.n_yard).reshape(self.batch_size, -1)] = False
            feasible_mask[:, :self.n_yard*self.n_take_out][encoder_mask.reshape(self.batch_size, -1)] = False
            
            probs = self.soft_max(out)
            m = Categorical(probs)
            action = m.sample() # [B]
            
            # ________________________________________________
            # 다음 스텝의 마스킹을 위해서 값들 불러오기
            block_selection, yard_selection = torch.div(action, self.n_yard, rounding_mode='trunc').type(torch.int64), (action % self.n_yard)
            isFeasible = feasible_mask[range(self.batch_size), action]
            
            block_size = self.block_size_info_to[range(self.batch_size), block_selection].unsqueeze(1) # [B, 1, 5]
            slot_info = torch.clone(self.yard_slots[range(self.batch_size), yard_selection]) # [B, pad_len, 5]
                        
            input0 = torch.cat([block_size, slot_info], dim=1).reshape(self.batch_size, -1).to(self.device)
            
            slot_info[slot_info[:, :, 0] == 0] = 100 # 가능한 슬롯이 없는 경우에 값 최대화
            slot_info[torch.any(block_size[:, :, 0:3] > slot_info[:, :, 0:3], dim=-1)] = 100 # 크기가 큰 경우는 선택이 안 되도록 최대화 
            yard_offset = torch.argmax((block_size[:, :, 0] * block_size[:, :, 1]) - (slot_info[:, :, 0] * slot_info[:, :, 1]), dim=-1) # [B]
            
            self.yard_slots[range(self.batch_size), yard_selection, yard_offset, -1] -= 1 # 선택 됐으면 감소
            
            self.yard_slots[self.yard_slots[:, :, :, -1] == 0] = 0 # 해당 크기의 슬롯이 남아있지 않으면 선택이 안 되도록 제거 
            slot_size = slot_info[range(self.batch_size), yard_offset] # 선택된 슬롯의 크기 불러오기
            left_space_in_slot = (slot_size[:, 0] * slot_size[:, 1] - (block_size[:, :, 0] * block_size[:, :, 1]).reshape(-1)) ** 2 * (torch.where(isFeasible, 1, -1)) # 불가능한 선택이면 Reward 최소화
            
            self.encoder_mask_whole_to[range(self.batch_size), block_selection] = False # False가 제외 시킬 대상
            self.encoder_mask_whole_to = self.encoder_mask_whole_to.permute(0, 2, 3, 1)
            self.encoder_mask_whole_to[(self.yard_slots[:, :, :, -1] == 0).unsqueeze(-1).repeat(1, 1, 1, self.n_take_out).to(self.device)] = False
            self.encoder_mask_whole_to = self.encoder_mask_whole_to.permute(0, 3, 1, 2)
            
            # ______________________      Reward      __________________________
            remaining_space_of_slots = self.yard_slots[:, :, :, 0] * self.yard_slots[:, :, :, 1]
            reward_space = left_space_in_slot + torch.sum(torch.sum(remaining_space_of_slots, dim=-1), dim=-1)
            
            self.barge_count_to, self.barge_slot_to, self.block_lengths_on_barge_to, self.block_width_on_barge_to = \
                self.allocated_to_barge(block_selection=block_selection, yard_selection=yard_selection, block_size=block_size, 
                                        barge_infos=(self.barge_count_to, self.barge_slot_to, self.block_lengths_on_barge_to, self.block_width_on_barge_to))
            
            # ______________________      Penalty      __________________________
            barge_with_blocks = torch.any(torch.any(self.barge_slot_to != -1, dim=-1), dim=-1)
            barge_num_by_yard = torch.sum(barge_with_blocks, dim=-1) # [B Y]
            total_barge_num = torch.sum(barge_num_by_yard, dim=-1) # [B]
            
            mask_wrong_barge_selection = torch.any(barge_num_by_yard > self.n_trip_infos_tensor, dim=-1)
            penalty_for_wrong_barge_selection = (block_size[:, :, 0] * block_size[:, :, 1]).reshape(-1) * (torch.where(mask_wrong_barge_selection, -1, 0))
            
            barge_taking_in = torch.any(torch.any(self.barge_slot_ti != -1, dim=-1), dim=-1).sum(-1)
            barge_num_by_yard = torch.sum(barge_with_blocks, dim=-1) # [B Y]
            penalty_for_excessive_barge = barge_num_by_yard - barge_taking_in
            penalty_for_excessive_barge[penalty_for_excessive_barge > 0] = 0
            penalty_for_excessive_barge = penalty_for_excessive_barge.sum(-1)
            
            # ______________________      Objective      __________________________
            self.reward_spaces_to[:, idx] = reward_space
            self.probs_history_to.append(m.log_prob(action))
            self.action_history_to.append(torch.hstack([torch.vstack((block_selection, yard_selection)).T, slot_size]))
            obj = self.alpha * torch.sum(self.reward_spaces_to, dim=-1)  + self.beta * total_barge_num + penalty_for_wrong_barge_selection + penalty_for_excessive_barge
            self.rewards_history_to.append(obj)
            
            # Gradient Error 때문에 매번 새로운 객체 형성 후 대체
            new_block_mask = self.block_mask_to.clone()
            new_block_mask[range(self.batch_size), block_selection] = True
            self.block_mask_to = new_block_mask
        
        self.probs_history_to = torch.stack(self.probs_history_to).transpose(1, 0).to(self.device)
        self.rewards_history_to = torch.stack(self.rewards_history_to).transpose(1, 0).to(self.device)
        self.action_history_to = torch.stack(self.action_history_to).transpose(1, 0).to(self.device)
        
        return {
            "take_out_probs" : self.probs_history_to,
            "take_out_rewards" : self.rewards_history_to,
            "take_out_actions" : self.action_history_to,
            "take_out_barges" : self.barge_slot_to,
            "take_in_probs" : self.probs_history_ti,
            "take_in_rewards" : self.rewards_history_ti,
            "take_in_actions" : self.action_history_ti,
            "take_in_barges" : self.barge_slot_ti
        }
    
        

class RLEnv():
    def __init__(self, args : Args, n_take_out=None) -> None:
        self.args = args
        self.yard_in = YardInside(name="사내", area_slots=None, blocks=None)
        # Set by random for now
        self.target_columns = ["length", "width", "height", "weight", "location"]
        self.yards_out : Dict[str, "YardOutside"]= {}
        self.first_step_infos = {}
        # self.load_env()
        self.batch_limit = (65, 20)
        self.probs = []
        self.rewards = []
        # self.min_result = (float("inf"), None, None) # Value of reward and combination of actions
        self.max_result = dict().fromkeys(["take_out_probs", "take_out_rewards", "take_out_actions", "take_out_barges", "take_in_probs", "take_in_rewards", "take_in_actions", "take_in_barges"]) # Value of reward and combination of actions
        
    def _init_env(self):
        def _init_yard_out(info):
            area_slots_ = []
            blocks_ = []
            
            for _, slot_info in info.iterrows():
                rand_n_block = np.random.randint(min(int(slot_info["count"]*0.7), (int(slot_info["count"]*0.9)-1)), int(slot_info["count"]*0.9))
                mask = ((self.whole_block["width"] >= slot_info["width_min"]) & (self.whole_block["width"] < slot_info["width_max(<)"]) 
                    & (self.whole_block["length"] < slot_info["length_max"]) & (self.whole_block["height"] < slot_info["height_max"])
                    & (self.whole_block["location"].isnull().values))
            
                if np.sum(mask) < rand_n_block:
                    rand_blocks = pd.DataFrame(None, columns=["vessel_id", "block"])
                else:
                    rand_blocks = self.whole_block.loc[mask, :].sample(rand_n_block).copy()
                self.whole_block.loc[rand_blocks.index, "location"] = slot_info["name"]
                blocks = rand_blocks.reset_index(drop=True)
            
                df_temp = np.repeat([[slot_info["length_max"], slot_info["width_min"], slot_info["width_max(<)"], slot_info["height_max"]]], slot_info["count"], axis=0)
                df_temp = pd.DataFrame(df_temp, columns=["length", "width_min", "width_max", "height"])
                df_temp.loc[df_temp.index[:len(blocks)], "vessel_id"] = blocks["vessel_id"].values
                df_temp.loc[df_temp.index[:len(blocks)], "block"] = blocks["block"].values
                # display(df_temp)
                area_slots_.append(df_temp)
                blocks_.append(blocks)
        
            area_slots = pd.concat(area_slots_).reset_index(drop=True)
            blocks = pd.concat(blocks_).reset_index(drop=True)
        
            return blocks, area_slots
        
        # ! 사외 적치장 정보 추가
        self.yards_out_slot_sizes = pd.read_csv(os.path.join(src_dir, "Yard_capacity.csv"))
        self.yards_out_slot_sizes.fillna(float("inf"), inplace=True)
        
        self.yards_out : Dict[str, "YardOutside"]= {}
        for name in self.yards_out_slot_sizes["name"].unique():
            info = self.yards_out_slot_sizes.loc[self.yards_out_slot_sizes["name"] == name]
            blocks, area_slots = _init_yard_out(info)
            area_slots["location"] = name
            self.yards_out[name] = YardOutside(name=name, area_slots= area_slots, blocks=blocks.reset_index(drop=True))
            
        self.whole_block.loc[self.whole_block["location"].isnull(), "location"] = "사내"
        self.labels_encoder = dict(zip(self.yards_out_slot_sizes["name"].unique(), range(1, len(self.yards_out_slot_sizes["name"].unique())+1)))
        self.labels_encoder["사내"] = 0
        self.labels_encoder_inv = dict(zip(range(1, len(self.yards_out_slot_sizes["name"].unique())+1), self.yards_out_slot_sizes["name"].unique()))
        self.labels_encoder_inv[0] = "사내"
        
    def get_state(self, possible_take_out: pd.DataFrame, take_in: pd.DataFrame):
        
        # Encoder에 Input으로 들어갈수록 
        possible_take_out["location"] = possible_take_out["location"].map(self.labels_encoder)
        encoder_inputs =[]
        yard_slots = []
        
        # print(self.yards_out.keys())
        for name, yard in self.yards_out.items():
            # print(name, end=" ")
            state_part_ = yard.area_slots.copy()
            state_part_["location"] = self.labels_encoder[name]
            state_part_.loc[state_part_["height"] == float("inf"), "height"] = 1000
            count = state_part_.loc[state_part_["vessel_id"].isnull(), ["length", "width", "height", "location"]].groupby(["length", "width"], as_index=False).value_counts()
            # if count.shape[0] == 0:
                # continue
            yard_t = torch.FloatTensor(count.values)
            count_ = np.zeros((self.args.pad_len, 5))
            count_[:count.shape[0], :] = count
            
            # if name =="오비3":
                # print(count_.shape, count_)
            yard_slots.append(count_)
            for _, block in possible_take_out.iterrows():
                block_t = torch.FloatTensor([block[["length", "width", "height", "location","weight"]].values])
                pad_t = torch.zeros((self.args.pad_len-len(count),5))
                input = torch.concat([block_t, yard_t, pad_t], dim=0)
                encoder_inputs.append(input)
                
        encoder_inputs = torch.stack(encoder_inputs)
        encoder_inputs = encoder_inputs.reshape(encoder_inputs.shape[0], -1)
        yard_slots = np.array(yard_slots)
        
        return encoder_inputs, yard_slots, None
            
    def step(self, infos: List[pd.DataFrame], inputs: torch.Tensor, remaining_areas: np.array):
        """
        infos: infos about block to be taken out, infos about block to be taken in, infos about slots in the yard 
        inputs: combination of block infos and yard slots
        """
        
        # TODO: Implement the block moving to the yard
        take_in, take_out, yard_slots = infos
        
        take_in["location"] = take_in["location"].map(self.labels_encoder).astype(float)
        take_out["location"] = take_out["location"].map(self.labels_encoder).astype(float)
        
        kwargs  = self.RLAgent((take_in.reset_index(drop=True), take_out.reset_index(drop=True), yard_slots), inputs, remaining_areas)
        self.result = kwargs
        
        max_idx = torch.argmax(kwargs["take_out_rewards"][:, -1], dim=-1).detach().cpu().numpy()
        max_reward = kwargs["take_out_rewards"][max_idx, -1].detach().cpu().numpy()
        if self.max_result["take_out_rewards"] is None:
            self.max_result = {k:v[max_idx].detach().cpu().numpy() for k, v in kwargs.items()}
        elif max_reward > self.max_result["take_out_rewards"][-1]:
            self.max_result = {k:v[max_idx].detach().cpu().numpy() for k, v in kwargs.items()}
        
        """
        Objective: maximize a*여유공간 - b*배치 수
        """
        return torch.mean(kwargs["take_out_rewards"][:, -1]).detach().cpu().numpy()
    
    def reset(self):
        self.whole_block = self.first_step_infos["whole_block"]
        for yard in self.yards_out.values():
            yard._reset()
            
    def get_result(self):
        # return self.min_result
        return self.max_result
    
    def update_policy(self):
        args = self.args
        
        take_out_probs = self.result["take_out_probs"]
        take_out_rewards = self.result["take_out_rewards"]
        
        take_out_loss = torch.zeros((len(take_out_probs)))
        take_out_returns = torch.zeros_like(take_out_rewards)
        for idx in range(args.n_take_out -1, -1, -1):
            # R = np.log1p(r) + 0.99 * R
            if idx == args.n_take_out-1:
                # take_out_returns[:, 0] = self.rewards[:, idx] # 처음에는 그냥 리워드 값
                take_out_returns[:, 0] = take_out_rewards[:, idx] # 처음에는 그냥 리워드 값
            else:
                take_out_returns[:, args.n_take_out-1-idx] = (take_out_rewards[:, idx] + args.decay * take_out_returns[:, args.n_take_out-idx-2])
                
        # norm_returns = torch.nn.functional.normalize(take_out_returns, dim=-1)
        mean = torch.mean(take_out_returns.reshape(-1))
        std = torch.std(take_out_returns.reshape(-1))
        norm_returns = (take_out_returns - mean) / std
        take_out_loss = torch.mul(take_out_probs * (-1), norm_returns)
        take_out_loss = torch.mean(take_out_loss)
        
        
        take_in_probs = self.result["take_in_probs"]
        take_in_rewards = self.result["take_in_rewards"]
        
        take_in_loss = torch.zeros((len(take_in_probs)))
        take_in_returns = torch.zeros_like(take_in_rewards)
        for idx in range(args.n_take_in -1, -1, -1):
            # R = np.log1p(r) + 0.99 * R
            if idx == args.n_take_in-1:
                # take_in_returns[:, 0] = self.rewards[:, idx] # 처음에는 그냥 리워드 값
                take_in_returns[:, 0] = take_in_rewards[:, idx] # 처음에는 그냥 리워드 값
            else:
                take_in_returns[:, args.n_take_in-1-idx] = (take_in_rewards[:, idx] + self.args.decay * take_in_returns[:, args.n_take_in-idx-2])
                
        norm_returns = (take_in_returns - mean) / std
        take_in_loss = torch.mul(take_in_probs * (-1), norm_returns)
        take_in_loss = torch.mean(take_in_loss)
        
        
        total_loss = take_out_loss + take_in_loss
        total_loss = take_out_loss
        total_loss.backward()
        
        self.optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.RLAgent.parameters(), 20)
        self.optimizer.step()
        
        return total_loss.item()
    
    
    def save_env(self, save_dir, exp_num, infos):
        whole_block = deepcopy(self.first_step_infos["whole_block"])
        whole_block.to_csv(os.path.join(save_dir, f"{exp_num}_whole_block.csv"), index=False)
        
        for name, yard in self.yards_out.items():
            yard.area_slots.to_csv(os.path.join(save_dir, f"{exp_num}_{name}.csv"), index=False)
        
        take_in, take_out, yard_slots = infos
        take_in.to_csv(os.path.join(save_dir, f"{exp_num}_shi_take_in.csv"), index=False)
        take_out.to_csv(os.path.join(save_dir, f"{exp_num}_shi_take_out.csv"), index=False)
        pd.DataFrame(yard_slots).to_csv(os.path.join(save_dir, f"{exp_num}_yard_slots.csv"), index=False)
        
        self.RLAgent = PGAgent(n_yard_out=len(self.yards_out), n_max_block=150, n_take_out=len(take_out))
        self.RLAgent.to(self.args.device)
        self.optimizer = torch.optim.Adam(self.RLAgent.parameters(), lr=5e-3)
        
    def load_env(self):
        data_dir = self.args.data_dir
        self.whole_block = pd.read_csv(os.path.join(data_dir, f"temp_whole_block.csv"), encoding="cp949")
        self.first_step_infos["whole_block"] = deepcopy(self.whole_block)
        
        yard_out_names = [os.path.basename(name).split(".")[0] for name in glob(os.path.join(data_dir, "*.csv")) if "temp" not in name]
        
        for name in yard_out_names:
            area_slots = pd.read_csv(os.path.join(data_dir, f"{name}.csv"), encoding="cp949")
            
            self.yards_out[name] = YardOutside(args=self.args, name=name, area_slots= area_slots, blocks=self.whole_block.loc[self.whole_block["location"] == name].copy().reset_index(drop=True))
            self.yards_out[name]._load_env()
            
            
        self.labels_encoder = dict(zip(yard_out_names, range(1, len(yard_out_names)+1))) # ! 빈공간이 없는 적치장(=고려하지 않아도 되는 적치장)이 있어도 이름을 인덱싱으로 고려하기 때문에 후에 인덱싱 문제 야기
        # print(self.labels_encoder)
        # quit()
        self.labels_encoder["사내"] = 0
        self.labels_encoder_inv = dict(zip(self.labels_encoder.values(), self.labels_encoder.keys()))
        
        # print(self.args.priori_conditions["destination"])
        # print(self.labels_encoder)
        self.args.priori_conditions["destination"] = self.args.priori_conditions["destination"].map(self.labels_encoder)
        self.args.priori_conditions = self.args.priori_conditions.values
        self.args.priori_conditions[:, 1] -= 1
        print(self.args.priori_conditions)
        
        n_trip_infos_enc = {self.labels_encoder[k]: v for k, v in self.args.n_trip_infos.items() if k != "사내"}
        self.args.n_trip_infos_list = [n_trip_infos_enc[k] for k in sorted(n_trip_infos_enc.keys())]
        
        self.RLAgent = PGAgent(self.args)
        self.RLAgent.to(self.args.device)
        self.optimizer = torch.optim.Adam(self.RLAgent.parameters(), lr=1e-3)
        
        
    
def match_batch_num(result, result_barge):
    var1, var2, var3, var4 = result_barge.shape
    for y_idx in range(var1): # 야드 인덱스, 굳이 result1에는 안 쓰임. 이미 destination으로 매칭되어 있기 때문
        for t_idx in range(var2): # 항차 인덱스, 몇 번째 항차인지
            for s_idx in range(var3): # 
                for p_idx in range(var4):
                    block_num = result_barge[y_idx, t_idx, s_idx, p_idx]
                    if block_num == -1 : continue
                    # print(env.labels_encoder_inv[y_idx], f"{t_idx} 번쨰 바지/ {s_idx} 번째 슬롯/ 병렬 슬롯: {p_idx}", result.iloc[int(block_num)])
                    # * 블록 번호랑, index랑 매칭
                    result.loc[block_num.item(), ["바지", "슬롯", "병렬"]] = [t_idx, s_idx, p_idx==1]
    

def save_result(result, obj_loss_history, args, infos, labels_encoder):
    take_in, take_out = infos
    plt.plot(obj_loss_history)
    plt.savefig(os.path.join(args.save_dir, "loss.png"))
    plt.close()
    
    take_out_blocks = result["take_out_actions"]
    result1 = take_out.iloc[take_out_blocks[:, 0]]
    result1 = result1[[col for col in result1.columns if col != "location"]]
    result1["location"] = "사내"
    result1["destination"] = take_out_blocks[:, 1] + 1
    result1["destination"] = result1["destination"].map(labels_encoder)
    result1["slot_length"] = take_out_blocks[:, 2]
    result1["slot_width"] = take_out_blocks[:, 3]
    result1["slot_height"] = take_out_blocks[:, 4]
    result1["slot_space"] = result1["slot_length"] * result1["slot_width"]
    result1["block_space"] = result1["length"] * result1["width"]
    result1["empty_space"] = result1["slot_space"] - result1["block_space"]
    match_batch_num(result1, result["take_out_barges"])
    
    take_in_blocks = result["take_in_actions"]
    result2 = take_in.iloc[take_in_blocks]
    result2["block_space"] = result2["length"] * result2["width"]
    result2["destination"] = "사내"
    match_batch_num(result2, result["take_in_barges"])
    
    result = pd.concat([result1, result2])
    # result["Batch"] = result["Batch"].map(dict(zip(result["Batch"].unique(), range(len(result["Batch"].unique())))))
    # result["Batch"] = result["Batch"].astype(int)
    # result.sort_values(by="Batch",ascending=True, inplace=True)
    result.rename(columns={
        "location": "출발지",
        "destination": "목적지", 
        "slot_length": "슬롯길이", 
        "slot_width": "슬롯너비", 
        "slot_height": "슬롯높이", 
        # "Batch": "배치번호", 
        "slot_space": "슬롯면적",
        "block_space": "블록면적", 
        "length": "블록길이", 
        "width": "블록너비", 
        "height": "블록높이", 
        "empty_space": "유휴면적"}, inplace=True)
    # result = result[["vessel_id", "block", "바지", "슬롯", "병렬", "출발지", "목적지", "배치번호", "슬롯면적", "블록면적", "유휴면적", "슬롯너비", "슬롯길이", "슬롯높이", "블록길이", "블록너비", "블록높이"]]
    result = result[["vessel_id", "block", "출발지", "목적지", "바지", "슬롯", "병렬", "슬롯면적", "블록면적", "유휴면적", "슬롯너비", "슬롯길이", "슬롯높이", "블록길이", "블록너비", "블록높이"]]
    
    # result.loc["reward", "location"] = result_obj
    # result.loc["time", "location"] = np.round((time2 - time1) / 60, 3)
    result.to_csv(os.path.join(args.save_dir, f"actions.csv"), index=False, encoding="cp949")

def run(args : Args):
    
    n_trip_infos = pd.read_csv(os.path.join(args.input_dir, "shi_max_trip_count.csv"))
    args.n_trip_infos = dict(zip(n_trip_infos["name"].values, n_trip_infos["count"].values))
    priori_blocks = pd.read_csv(os.path.join(args.input_dir, "shi_priori.csv"))
    
    take_in = pd.read_csv(os.path.join(args.data_dir, f"temp_take_in.csv"), encoding="cp949")
    take_out = pd.read_csv(os.path.join(args.data_dir, f"temp_take_out.csv"), encoding="cp949")
    
    args.n_take_in = len(take_in)
    args.n_take_out = len(take_out)
    
    take_out["key"] = take_out["vessel_id"]+"_"+take_out["block"]
    take_out["index"] = range(len(take_out))
    priori_blocks["key"] = priori_blocks["vessel_id"]+"_"+priori_blocks["block"]
    temp = pd.merge(take_out, priori_blocks, on="key", how="inner")
    args.priori_conditions = temp[["index", "destination"]]
    
    with open(os.path.join(args.save_dir, "info.txt"), "w") as f:
        f.write(f"Gamma to 0.1 \n \
            batch_size: {args.batch_size}\n \
            hid_dim: {args.hid_dim}\n \
            gamma: {args.decay}\n \
            pad_len: {args.pad_len}\n \
            n_epoch: {args.n_epoch} \
        ")
    
    
    with open(os.path.join(args.save_dir, "result.txt"), "w") as f:
        f.write("exp_num,objective,time\n")
        env = RLEnv(args)
        env.load_env()
        
        encoder_inputs, remaining_areas, yard_slots = make_input_from_loaded(take_out, take_in, env) # == env.get_state()와 같다
        
        time1 = time.time()
        obj_loss_history = []
        loop = tqdm(range(args.n_epoch))
        for epo in loop:
            env.reset()
            for day in range(1):
                objective = env.step((take_in.copy(), take_out.copy(), deepcopy(yard_slots)), deepcopy(encoder_inputs.to(args.device)), deepcopy(remaining_areas))
            loss = env.update_policy()
            loop.set_description(f"Loss: {loss:.3f} / Average Reward: {objective:.3f}")
            # loss2 = env.update_policy2(len(take_in))
            # loop.set_description(f"{exp_num} / {loss:.3f} + {loss2:.3f} / {objective}")
            obj_loss_history.append(objective)
            # break
        
        time2 = time.time()
        kwargs = env.get_result()
        save_result(obj_loss_history=obj_loss_history, result=kwargs, args=args, infos=(take_in, take_out), labels_encoder=env.labels_encoder_inv)
    
        print(f"Took {(time2 - time1):.3f} seconds")
        f.write(f"{kwargs['take_out_rewards']},{time2-time1}\n")
        
        
    
    
if __name__ == "__main__":
    args = Args()
    
    # par_dir = os.path.abspath(os.path.join(__file__, * [os.pardir] * 1))
    par_dir = r"D:\Dropbox\Projects_Mine\삼성중공업\code\SHI_Project"
    src_dir = os.path.join(par_dir, "Src")
    result_dir = os.path.join(par_dir, "Result")
    input_dir = os.path.join(par_dir, "Input")
    data_dir = os.path.join(par_dir, "Temp")
    
    # isExp는 실험을 하는 상황을 가정하므로 반출될 블록과 반입될 블록을 직접 기입하는 게 아니라 랜덤으로 블러오는 상황을 가정한 겁니다. 
    # 랜덤이 주어진 입력 값을 사용하고 싶으시다면 isExp 인자를 False로 바꿔서 넣어주면 됩니다.
    # preprocess(par_dir=par_dir, src_dir=src_dir, data_dir=data_dir, isExp=True)
    preprocess(par_dir=par_dir, isExp=False)
    
    current_dateTime = datetime.now()
    # args.save_dir = os.path.join(result_dir, f"{current_dateTime.year}_{current_dateTime.month}_{current_dateTime.day}_{current_dateTime.hour}_{current_dateTime.minute}_exp")
    args.save_dir = os.path.join(result_dir, "test")
    os.makedirs(args.save_dir, exist_ok=True)
    args.data_dir = data_dir
    os.makedirs(args.data_dir, exist_ok=True)
    args.input_dir = input_dir
    
    # args.exp_num = None
    args.n_epoch = 10 # 빠르게 확인해보고 싶은 경우 200까지 돌려도 무난합니다만, Epoch 크기가 커질수록 성능이 향상됩니다
    args.batch_size = 512 # 딥러닝 모델에서 동시에 실험해보는 경우의 수입니다. GPU가 있을 경우 비디오 RAM에 공간을 차지하게 됩니다.
    args.hid_dim = 256 # 모델의 크기
    args.decay = 0.9
    args.pad_len = 50 # 야드 슬롯의 Case를 표현하는 값입니다. 크기가 클수록 다양한 Slot들의 경우를 고려해줄 수 있지만 속도는 느려집니다.
    args.barge_batch_size = (65, 20) # 바지선을 표현하는 Batch의 크기 제한입니다. 우선은 너비로 고려했기 때문에 본 값이 곱해진 1300으로 고려가 될 거 같습니다.
    args.n_yard = len(glob(os.path.join(data_dir, "*.csv"))) - len(glob(os.path.join(data_dir, "temp_*.csv"))) # 사외 적치장의 수 입니다.
    args.n_max_block = 100 # 최대 고려 가능한 블록 수: 수가 많아질 수록 느려집니다. 진행했던 실험은 60으로 두고 진행했습니다.
    args.max_trip = 4
    args.barge_max_row = 5
    args.barge_par_width = 10.5
    args.barge_max_length = 70
    args.alpha = 1e-2 # 유휴 공간 목적식 파라미터
    args.beta = -1 # 바지선 개수 목적식 파라미터
    args.max_barge_block_alloc = 10 # 한 바지선이 최대로 수용할 수 있는 블록 수를 임의로 정의한 것입니다.
    
    run(args)
    # env, result1 = run(args) # ! For the test

# %%

# result_val, result_actions, result_obj, result_barge = env.get_result()
# result_barge.detach().cpu().numpy()
# print(result_barge.shape)
# print(result1)
# # %%
# display(result1)
# temp_match = result1.index.to_list()
# print(temp_match)
# temp_result = result1.copy()
# var1, var2, var3, var4 = result_barge.shape
# for y_idx in range(var1):
#     for t_idx in range(var2):
#         for s_idx in range(var3):
#             for p_idx in range(var4):
#                 block_num = result_barge[y_idx, t_idx, s_idx, p_idx]
#                 if block_num == -1 : continue
#                 # print(env.labels_encoder_inv[y_idx], f"{t_idx} 번쨰 바지/ {s_idx} 번째 슬롯/ 병렬 슬롯: {p_idx}", result1.iloc[int(block_num)])
#                 # * 블록 번호랑, index랑 매칭
#                 temp_result.loc[block_num.item(), ["바지", "슬롯", "병렬"]] = [t_idx, s_idx, p_idx==1]
#                 print(env.labels_encoder_inv[y_idx], f"{t_idx} 번쨰 바지/ {s_idx} 번째 슬롯/ 병렬 슬롯: {p_idx}/ 블록번호: {block_num.item()}")
#                 # print(temp_result.iloc[temp_match.index(block_num.item())].values)
# display(temp_result)

# import numpy as np

# temp = np.arange(6).reshape(3,2)

# for row in temp:
#     elem_l, elem_r = row
#     print(elem_l)
import torch

temp_mask = torch.ones((3,4,2), dtype=torch.bool)
temp_mask[:, [0, 1]] = False
print(temp_mask)
# %%
