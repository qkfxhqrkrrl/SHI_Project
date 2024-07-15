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
warnings.filterwarnings("ignore")

torch.autograd.set_detect_anomaly(True)

class Args:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.save_dir = None
        self.input_dir = None
        self.isExp = None
        
        self.barge_batch_size = None
        self.n_yard = None
        self.n_max_block = None
        self.n_take_in = None
        self.n_take_out = None
        
        self.batch_size = None
        self.lr = None
        self.std_eps = None
        self.hid_dim = None
        self.emb_dim = None
        self.decay = None
        self.clipping_size = None
        self.pad_len = None
        self.n_epoch = None
        self.exp_num = None

        
# ! ________________________  PREPROCESS _____________________________

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
        if True:
            self.area_slots = area_slots
            self.blocks = blocks
        else :
            # TODO: 아직 슬롯이 업데이트 되는 상황은 가정 X
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
        self.emb_dim = args.emb_dim
        self.batch_size = args.batch_size
        self.pad_len = args.pad_len
        self.n_yard = args.n_yard
        self.n_max_block = args.n_max_block
        self.n_trip_infos_list = args.n_trip_infos_list
        self.priori_conditions = args.priori_conditions 
        
        self.n_take_in = args.n_take_in
        self.n_take_out = args.n_take_out
        

        self.to_decoder = nn.LSTMCell(input_size=(self.emb_dim+1)*self.emb_dim, hidden_size=args.hid_dim)
        self.to_fc_block = nn.Linear(args.hid_dim, self.n_max_block*self.n_yard) # 400 = 40(n_block) * 10(n_yard)
        
        self.ti_decoder = nn.LSTMCell(input_size=(self.emb_dim+1)*self.emb_dim, hidden_size=args.hid_dim)
        self.ti_fc_block = nn.Linear(args.hid_dim, self.n_max_block*self.n_yard)

        self.yard_emb = nn.Embedding(self.n_yard, self.emb_dim)
        self.size_emb = nn.Linear(2, self.emb_dim)
        self.length_emb = nn.Linear(1, self.emb_dim)
        self.width_emb = nn.Linear(1, self.emb_dim)
        
        self.device = args.device
        # TODO: 제원 정보 임베딩으로 변환하는 방법 적용 고려해보기
        
        self.soft_max = nn.Softmax(dim=-1)
    
    def init_env_info(self, infos):
        # 위에서 Init을 안 하는 이유는 이건 Env.reset과 함께 다시 초기화되어야 하는 값이기 때문
        take_in, take_out, yard_slots = infos
        
        input0 = torch.zeros(self.batch_size, self.emb_dim*(self.emb_dim+1)).to(self.device)
        # input0 = torch.zeros(self.batch_size, 5*(self.pad_len+1)).to(self.device)
        h0, c0 = torch.zeros((self.batch_size, self.hid_dim), device=self.device), torch.zeros((self.batch_size, self.hid_dim), device=self.device)
        
        if self.n_trip_infos_list is not None:
            self.n_trip_infos_tensor = torch.FloatTensor(self.n_trip_infos_list).to(self.device).repeat(self.batch_size, 1)
        yard_slots = np.expand_dims(yard_slots, axis=0).repeat(self.batch_size, axis=0)
        self.yard_slots = torch.FloatTensor(yard_slots).to(self.device)
        
        
        # _______________________________ Take out _______________________________
        self.block_size_info_to = torch.FloatTensor(np.expand_dims(take_out[["length", "width", "height", "weight", "location"]].values, axis=0).repeat(self.batch_size, axis=0)).to(self.device)
        self.encoder_mask_whole_to = self.generate_mask(take_out, yard_slots) # [B n_take_out n_yard n_pad]
        self.block_mask_to = torch.zeros((self.batch_size, self.n_take_out, 1), dtype=bool) # 제외 시키고 싶은 경우를 True
        self.feasible_mask = torch.ones((self.batch_size, self.n_max_block*self.n_yard), dtype=torch.bool, device=args.device)
        
        self.barge_count_to = torch.zeros((self.batch_size, self.n_yard), dtype=torch.int64, device=args.device)
        self.barge_slot_to = -1 * torch.ones((self.batch_size, self.n_yard, self.max_trip, self.barge_max_row, 2), dtype=torch.int64, device=args.device) # 블록 인덱스 저장
        self.block_lengths_on_barge_to = torch.zeros((self.batch_size, self.n_yard, self.max_trip, self.barge_max_row), dtype=torch.float32, device=args.device) # 블록 길이 저장
        self.block_width_on_barge_to = torch.zeros((self.batch_size, self.n_yard, self.max_trip, self.barge_max_row), dtype=torch.float32, device=args.device) # 블록 너비 저장
        self.mask_wrong_barge_selection = torch.zeros(self.batch_size, self.n_yard, self.n_max_block, dtype=torch.bool)
        
        self.probs_history_to = []
        self.rewards_history_to = []
        self.action_history_to = []
        self.reward_parts = torch.zeros((self.batch_size, self.n_take_out), dtype=float).to(self.device)
        self.left_spaces_after_block_to = torch.zeros((self.batch_size, self.n_take_out), dtype=float).to(self.device)
        
        self.priori_index = []
        if self.priori_conditions is not None:
            for row in self.priori_conditions:
                block_idx, yard_idx = row
                self.priori_index.append(block_idx * self.n_yard + yard_idx)
        
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
    
    
    def allocate_to_barge(self, block_selection, block_size, yard_selection, barge_infos):
        # TODO: 바지선에 병렬 제약을 10이 아니라 합이 23 이하가 되도록
        barge_count ,barge_slot, block_lengths_on_barge, block_width_on_barge = barge_infos
        
        cur_barge_num = barge_count[range(self.batch_size), yard_selection] # [B]
    
        block_length_b = block_size[:, 0, 0] # Unsqueeze in [1]
        block_width_b = block_size[:, 0, 1] # [B]
        
        # ! block: parallel, slot: have_space is decided here
        does_it_have_existing = block_width_on_barge[range(self.batch_size), yard_selection, cur_barge_num] > 0
        is_allocable_mask = block_width_on_barge[range(self.batch_size), yard_selection, cur_barge_num] + block_width_b.unsqueeze(-1) <= self.barge_par_width #  # [B R]: 배치별로 선택된 야드의 현재 바지선에 중에서 병렬이 가능한 슬롯에 대한 진위 여부
        already_have_parallel = barge_slot[range(self.batch_size), yard_selection, cur_barge_num, :, 1] == -1  # [B R]: 병렬 열이 비어있는 경우
        is_allocable_and_have_existing_mask = torch.logical_and(torch.logical_and(is_allocable_mask, does_it_have_existing), already_have_parallel) 
        is_allocable_and_have_existing = torch.any(is_allocable_and_have_existing_mask, dim=-1)
        par_b_idxs = is_allocable_and_have_existing.nonzero().reshape(-1)
        par_y_idxs = yard_selection[is_allocable_and_have_existing]
        par_s_idxs = cur_barge_num[is_allocable_and_have_existing]
        
        parallel_block_length_on_barge_batch = block_lengths_on_barge[(par_b_idxs, par_y_idxs, par_s_idxs)].clone() # [B R 1]
        parallel_block_length_on_barge_batch[torch.logical_not(is_allocable_and_have_existing_mask[par_b_idxs])] = -1 # [B R]: 병렬 가능한 Slot에 한해서 길이가 가장 긴 경우의 인덱스를 구하고 싶은 거니까 나머진 -1로
        
        maxs = torch.max(parallel_block_length_on_barge_batch, dim=-1)
        max_idxs, max_val = maxs.indices, maxs.values
        
        # * block: parallel, slot: have_space, length: Check if it is smaller than existing, existing_block: True
        directly_allocable_slot = max_val >= block_length_b[par_b_idxs]
        dir_alloc_b_idxs = par_b_idxs[directly_allocable_slot]
        dir_alloc_y_idxs = par_y_idxs[directly_allocable_slot]
        dir_alloc_s_idxs = par_s_idxs[directly_allocable_slot]
        dir_alloc_slot_idxs = max_idxs[directly_allocable_slot]
        
        barge_slot[dir_alloc_b_idxs, dir_alloc_y_idxs, dir_alloc_s_idxs, dir_alloc_slot_idxs, 1] = block_selection[dir_alloc_b_idxs]
        block_width_on_barge[dir_alloc_b_idxs, dir_alloc_y_idxs, dir_alloc_s_idxs, dir_alloc_slot_idxs] += block_width_b[dir_alloc_b_idxs]
        # print(block_width_on_barge[0])
        
        # ! length: If it is not smaller than existing, it need reconsideration
        need_reconsider_slot = torch.logical_not(directly_allocable_slot)
        recons_b_idxs = par_b_idxs[need_reconsider_slot]
        recons_y_idxs = par_y_idxs[need_reconsider_slot]
        recons_s_idxs = par_s_idxs[need_reconsider_slot]
        recons_slot_idxs = max_idxs[need_reconsider_slot]
        recons_max_val = max_val[need_reconsider_slot]
        
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
        
        # ! plain blocks
        
        plain_allocable = torch.logical_not(is_allocable_and_have_existing) # [B]
        plain_b_idxs = plain_allocable.nonzero().reshape(-1)
        plain_y_idxs = yard_selection[plain_allocable]
        plain_s_idxs = cur_barge_num[plain_allocable]
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
    
    
    
    def forward(self, infos: List[pd.DataFrame], encoder_inputs : torch.Tensor = None, remaining_area = None):
        """
        Feature information
        take_out, take_in, feat_vec: ["length", "width", "height", "weight", "location"]
        slot_info: [length  width  height  location  count]
        """
        input0, h0, c0 = self.init_env_info(infos)
        next_inputs = (torch.clone(input0), torch.clone(h0), torch.clone(c0))
        
        for idx in range(self.n_take_in):
            h0, c0 = self.ti_decoder(input0, (h0, c0))
            out = self.ti_fc_block(h0)
            
            out[:, self.n_take_in:] = -20000
            out[:, :self.n_take_in][self.block_mask_ti] = -20000
            
            probs = self.soft_max(out)
            m = Categorical(probs)
            action = m.sample()
            
            block_size = self.block_size_info_ti[range(self.batch_size), action]
            # print(block_size)
            
            yard = block_size[:, -1] -1
            # print(torch.unique(yard))
            yard = yard.type(torch.LongTensor).to(self.device)
            slot_info = self.yard_slots[range(self.batch_size), yard] 
            
            embedded_block = self.size_emb(block_size[:, [0,1]]).unsqueeze(1)
                        
            embedded_slot_pre = self.size_emb(self.yard_slots[:, :, :, [0,1]])
        
            count_of_slots = self.yard_slots[:, :, :, -1].clone().unsqueeze(-1)
            embedded_slot = torch.sum(embedded_slot_pre * count_of_slots, dim=2).transpose(1,2)
        
            yard_emb = self.yard_emb(torch.arange(self.n_yard).to(self.device)).unsqueeze(0).repeat(self.batch_size, 1, 1)
            embedded_yard = torch.matmul(embedded_slot, yard_emb)
        
            input0 = torch.cat([embedded_block, embedded_yard], dim=1).reshape(self.batch_size, -1)
            
            self.barge_count_ti, self.barge_slot_ti, self.block_lengths_on_barge_ti, self.block_width_on_barge_ti = \
                self.allocate_to_barge(block_selection=action, yard_selection=yard, block_size=block_size.unsqueeze(1), 
                                        barge_infos=(self.barge_count_ti, self.barge_slot_ti, self.block_lengths_on_barge_ti, self.block_width_on_barge_ti))
            
            barge_with_blocks = torch.any(torch.any(self.barge_slot_ti != -1, dim=-1), dim=-1)
            barge_num_by_yard = torch.sum(barge_with_blocks, dim=-1) # [B Y]
            total_barge_num = torch.sum(barge_num_by_yard, dim=-1) # [B]
            
            if self.n_trip_infos_list is not None:
                mask_wrong_barge_selection = torch.any(barge_num_by_yard > self.n_trip_infos_tensor, dim=-1)
                penalty_for_wrong_barge_selection = (block_size[:, 0] * block_size[:, 1]).reshape(-1) * (torch.where(mask_wrong_barge_selection, -1, 0))
            
            obj = self.beta * total_barge_num
            if self.n_trip_infos_list is not None:
                obj = obj + penalty_for_wrong_barge_selection
            
            self.probs_history_ti.append(m.log_prob(action))
            self.action_history_ti.append(action)
            self.rewards_history_ti.append(obj)
            
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
            
            if self.n_trip_infos_list is not None:
                barge_with_blocks = torch.any(torch.any(self.barge_slot_to != -1, dim=-1), dim=-1) 
                barge_num_by_yard = torch.sum(barge_with_blocks, dim=-1) # [B Y]
                wrong_barge_selection = barge_num_by_yard > self.n_trip_infos_tensor # [B Y]
                self.mask_wrong_barge_selection[wrong_barge_selection] = True
            
            h0, c0 = self.to_decoder(input0, (h0, c0))
            out = self.to_fc_block(h0)
            # ________________________________________________
            
            out[:, :self.n_yard*self.n_take_out][encoder_mask.reshape(self.batch_size, -1)] = -10000 # *(기존) 보내고자 하는 블록이 적치장에 안 맞으면 마스킹, 제약 조건을 어겨도 적치장에 보낼 수는 있도록
            if len(self.priori_index)  != 0:
                out[:, self.priori_index] = -15000 # !(추가) 특정 블록을 해당 지역에는 보내면 안 되므로 값을 제한
            if self.n_trip_infos_list is not None:
                out[self.mask_wrong_barge_selection.transpose(1,2).reshape(self.batch_size, -1)] = -30000 # !(추가) 최대 항차 수가 넘은 경우 제한 
            out[:, :self.n_yard*self.n_take_out][self.block_mask_to.repeat(1, 1, self.n_yard).reshape(self.batch_size, -1)] = -30000 # *(기존) 선택된 블록은 선택 안 되도록
            out[:, self.n_yard*self.n_take_out:] = -30000 # *(기존) 고려하고자 하는 외의 Padding 부분은 마스킹
            
            self.feasible_mask[:, self.n_yard*self.n_take_out:] = False
            self.feasible_mask[:, :self.n_yard*self.n_take_out][self.block_mask_to.repeat(1, 1, self.n_yard).reshape(self.batch_size, -1)] = False
            self.feasible_mask[:, :self.n_yard*self.n_take_out][encoder_mask.reshape(self.batch_size, -1)] = False
            if len(self.priori_index)  != 0:
                self.feasible_mask[:, self.priori_index] = False # 특정 블록을 보내야 하므로 배정된 야드 외에 값들은 제한
            
            probs = self.soft_max(out)
            m = Categorical(probs)
            action = m.sample() # [B]
            
            # ________________________________________________
            block_selection, yard_selection = torch.div(action, self.n_yard, rounding_mode='trunc').type(torch.int64), (action % self.n_yard)
            isFeasible = self.feasible_mask[range(self.batch_size), action]
            
            block_size = self.block_size_info_to[range(self.batch_size), block_selection].unsqueeze(1) # [B, 1, 5]
            slot_info = torch.clone(self.yard_slots[range(self.batch_size), yard_selection]) # [B, pad_len, 5]
            
            ####
            embedded_block = self.size_emb(block_size[:, 0, [0,1]]).unsqueeze(1)
            embedded_slot_pre = self.size_emb(self.yard_slots[:, :, :, [0,1]])
        
            count_of_slots = self.yard_slots[:, :, :, -1].clone().unsqueeze(-1)
            embedded_slot = torch.sum(embedded_slot_pre * count_of_slots, dim=2).transpose(1,2)
        
            yard_emb = self.yard_emb(torch.arange(self.n_yard).to(self.device)).unsqueeze(0).repeat(self.batch_size, 1, 1)
            embedded_yard = torch.matmul(embedded_slot, yard_emb)
        
            input0 = torch.cat([embedded_block, embedded_yard], dim=1).reshape(self.batch_size, -1)
            
            self.barge_count_to, self.barge_slot_to, self.block_lengths_on_barge_to, self.block_width_on_barge_to = \
                self.allocate_to_barge(block_selection=block_selection, yard_selection=yard_selection, block_size=block_size, 
                                        barge_infos=(self.barge_count_to, self.barge_slot_to, self.block_lengths_on_barge_to, self.block_width_on_barge_to))
                
            # ______________________      Masking      __________________________
            
            #### Maximize
            slot_info[slot_info[:, :, 0] == 0] = 100 # 가능한 슬롯이 없는 경우에 값 최대화
            slot_info[torch.any(block_size[:, :, 0:3] > slot_info[:, :, 0:3], dim=-1)] = 100 # 크기가 큰 경우는 선택이 안 되도록 최대화 
            yard_offset = torch.argmax((block_size[:, :, 0] * block_size[:, :, 1]) - (slot_info[:, :, 0] * slot_info[:, :, 1]), dim=-1) # [B]
            
            self.yard_slots[range(self.batch_size), yard_selection, yard_offset, -1] = self.yard_slots[range(self.batch_size), yard_selection, yard_offset, -1] - 1 # 선택 됐으면 감소
            
            self.yard_slots[self.yard_slots[:, :, :, -1] == 0] = 0 # 해당 크기의 슬롯이 남아있지 않으면 선택이 안 되도록 제거 
            slot_size = slot_info[range(self.batch_size), yard_offset] # 선택된 슬롯의 크기 불러오기
            left_space_after_block = (slot_size[:, 0] * slot_size[:, 1] - (block_size[:, :, 0] * block_size[:, :, 1]).reshape(-1)) ** 2 * (torch.where(isFeasible, 1.0, self.alpha * -1e-2)) # 불가능한 선택이면 Reward 최소화
            
            whole_slots = self.yard_slots[:, :, :, :3].unsqueeze(1).repeat(1, self.n_take_out, 1, 1, 1)
            whole_block_size = self.block_size_info_to[:, :,:3].unsqueeze(-2).unsqueeze(-2)
            self.encoder_mask_whole_to = torch.all(whole_block_size <= whole_slots, dim=-1) # 크기가 맞는 slot 이 없으면 제외
            self.encoder_mask_whole_to = self.encoder_mask_whole_to.permute(0, 2, 3, 1)
            self.encoder_mask_whole_to[(self.yard_slots[:, :, :, -1] == 0).unsqueeze(-1).repeat(1, 1, 1, self.n_take_out).to(self.device)] = False # 남아있는 slot이 없는 적치장 제외
            self.encoder_mask_whole_to = self.encoder_mask_whole_to.permute(0, 3, 1, 2)
            
            
            # ______________________      Reward      __________________________
            remaining_space_of_slots = (self.yard_slots[:, :, :, 0] * self.yard_slots[:, :, :, 1]) ** 2
            
            # ______________________      Penalty      __________________________
            barge_with_blocks = torch.any(torch.any(self.barge_slot_to != -1, dim=-1), dim=-1)
            barge_num_by_yard = torch.sum(barge_with_blocks, dim=-1) # [B Y]
            total_barge_num = torch.sum(barge_num_by_yard, dim=-1) # [B]
            
            if self.n_trip_infos_list is not None:
                mask_wrong_barge_selection = torch.any(barge_num_by_yard >= self.n_trip_infos_tensor, dim=-1)
                penalty_for_wrong_barge_selection = (block_size[:, :, 0] * block_size[:, :, 1]).reshape(-1) * (torch.where(mask_wrong_barge_selection, -1, 0))
            
            barge_taking_in = torch.any(torch.any(self.barge_slot_ti != -1, dim=-1), dim=-1).sum(-1)
            barge_num_by_yard = torch.sum(barge_with_blocks, dim=-1) # [B Y]
            penalty_for_excessive_barge = barge_num_by_yard - barge_taking_in
            penalty_for_excessive_barge[penalty_for_excessive_barge > 0] = 0
            penalty_for_excessive_barge = penalty_for_excessive_barge.sum(-1)
            
            # ______________________      Objective      __________________________
            self.left_spaces_after_block_to[:, idx] = left_space_after_block
            space_reward = torch.sum(self.left_spaces_after_block_to, dim=-1) + torch.sum(torch.sum(remaining_space_of_slots, dim=-1), dim=-1)
            self.probs_history_to.append(m.log_prob(action))
            self.action_history_to.append(torch.hstack([torch.vstack((block_selection, yard_selection)).T, slot_size]))
            # obj = self.alpha * space_reward + self.beta * total_barge_num
            obj = self.alpha * space_reward + self.beta * total_barge_num + penalty_for_excessive_barge
            if self.n_trip_infos_list is not None:
                obj = obj + penalty_for_wrong_barge_selection 
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
        self.probs = []
        self.rewards = []
        self.max_result = dict().fromkeys(["take_out_probs", "take_out_rewards", "take_out_actions", "take_out_barges", "take_in_probs", "take_in_rewards", "take_in_actions", "take_in_barges"]) # Value of reward and combination of actions
        
        # TODO: Input 파일에 맞게 로딩되도록
        self.labels_encoder = {
            "사내" : 0,
            "덕곡1" : 1,
            "덕곡2" : 2,
            "봉암8" : 3,
            "봉암9" : 4,
            "HSG성동" : 5,
            "오비" : 6,
            "신한내" : 7,
            "한내" : 8,
            "오비3" : 9,
            "OFD" : 10 # 임시로 설정해둔 값입니다.
        } # 해당 방식에 맞지 않으면 에러가 발생합니다
        args.n_yard = len(self.labels_encoder) - 1
        # 만약 적치장 전체에 대한 정보를 얻을 수 있다면 해당 부분을 직접 입력하는 형태가 아니라 파일을 불러오는 방식으로 변경하면 될거 같습니다.
        # 모두 Batch 형태로 변환하여 병렬 연산으로 수행하기 위해섭니다. 
        self.labels_encoder_inv = dict(zip(self.labels_encoder.values(), self.labels_encoder.keys()))
        
        
    def get_state(self, possible_take_out: pd.DataFrame, take_in: pd.DataFrame):
        
        # Encoder에 Input으로 들어갈수록 
        possible_take_out["location"] = possible_take_out["location"].map(self.labels_encoder)
        encoder_inputs =[]
        yard_slots = []
        
        for name in self.labels_encoder.keys():
            if name == "사내": continue
            yard = self.yards_out[name]
            if yard.area_slots is None: 
                count_ = np.zeros((self.args.pad_len, 5))
            else:
                state_part_ = yard.area_slots.copy()
                state_part_["name"] = self.labels_encoder[name]
                state_part_.loc[state_part_["height"] == float("inf"), "height"] = 1000
                count = state_part_[["length", "width", "height", "location"]].groupby(["length", "width"], as_index=False).value_counts()
                # if count.shape[0] == 0:                 # continue
                yard_t = torch.FloatTensor(count.values)
                count_ = np.zeros((self.args.pad_len, 5))
                count_[:count.shape[0], :] = count
            
            yard_slots.append(count_)
        yard_slots = np.array(yard_slots)
        
        return encoder_inputs, yard_slots, None
            
    def step(self, infos: List[pd.DataFrame], inputs: torch.Tensor, remaining_areas: np.array):
        
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
        
        return torch.mean(kwargs["take_out_rewards"][:, -1]).detach().cpu().numpy()
    
    def reset(self):
        for yard in self.yards_out.values():
            yard._reset()
            
    def get_result(self):
        return self.max_result
    
    def update_policy(self):
        args = self.args
        
        self.optimizer.zero_grad()
        
        take_out_probs = self.result["take_out_probs"]
        take_out_rewards = self.result["take_out_rewards"].type(torch.float32)
        
        take_out_loss = torch.zeros((len(take_out_probs)))
        take_out_returns = torch.zeros_like(take_out_rewards)
        for idx in range(args.n_take_out -1, -1, -1):
            if idx == args.n_take_out-1:
                take_out_returns[:, 0] = take_out_rewards[:, idx] # 처음에는 그냥 리워드 값
            else:
                take_out_returns[:, args.n_take_out-1-idx] = (take_out_rewards[:, idx] + args.decay * take_out_returns[:, args.n_take_out-idx-2])
                
        norm_returns = (take_out_returns - take_out_returns.mean(dim=-1).unsqueeze(-1)) / (take_out_returns.std(dim=-1).unsqueeze(-1) + args.std_eps)
        take_out_loss = torch.mul(-take_out_probs, norm_returns).sum(-1) # Maximize
        
        
        take_in_probs = self.result["take_in_probs"]
        take_in_rewards = self.result["take_in_rewards"].type(torch.float32)
        
        take_in_loss = torch.zeros((len(take_in_probs)))
        take_in_returns = torch.zeros_like(take_in_rewards)
        for idx in range(args.n_take_in -1, -1, -1):
            if idx == args.n_take_in-1:
                take_in_returns[:, 0] = take_in_rewards[:, idx] # 처음에는 그냥 리워드 값
            else:
                take_in_returns[:, args.n_take_in-1-idx] = (take_in_rewards[:, idx] + self.args.decay * take_in_returns[:, args.n_take_in-idx-2])
                
        norm_returns = (take_in_returns - take_in_returns.mean(dim=-1).unsqueeze(-1)) / (take_in_returns.std(dim=-1).unsqueeze(-1) + args.std_eps)
        take_in_loss = torch.mul(-take_in_probs, norm_returns).sum(-1) # Maximize
        
        total_loss = torch.mean(take_out_loss)
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.RLAgent.parameters(), self.args.clipping_size)
        self.optimizer.step()
        
        return total_loss.item()
    
        
    def load_env(self):
        # TODO: 최적화와 동기화
        
        empty_slots = pd.read_csv(os.path.join(self.args.input_dir, "shi_slot_infos.csv"), encoding="cp949")
        del empty_slots["Unnamed: 0"]
        for name in self.labels_encoder.keys():
            if name == "사내": continue
            area_slots = empty_slots.loc[empty_slots["name"] == name].copy().reset_index(drop=True)
            
            empty_location_slot_extended = []
            for idx, row in area_slots.iterrows():
                temp_slots = np.zeros((row["count"], 6))
                temp_slots[:, :2] = np.nan
                temp_slots[:, 2:5] = np.repeat(row[["length", "width", "height"]].values.reshape(-1, 1), row["count"], axis=1).T
                temp_slots[:, 5] = 1000
                temp_slots = pd.DataFrame(temp_slots, columns=["vessel_id", "block", "length", "width", "height", "location"])
                empty_location_slot_extended.append(temp_slots)
            area_slots = pd.concat(empty_location_slot_extended) if len(empty_location_slot_extended) > 0 else None
            
            self.yards_out[name] = YardOutside(args=self.args, name=name, area_slots= area_slots)
            self.yards_out[name]._load_env()
            
        if self.args.priori_conditions is not None:
            self.args.priori_conditions["name"] = self.args.priori_conditions["name"].map(self.labels_encoder)
            self.args.priori_conditions = self.args.priori_conditions.values
            self.args.priori_conditions[:, 1] -= 1
            
        if self.args.n_trip_infos is not None:
            self.args.n_trip_infos_list = []
            for key in self.labels_encoder.keys():
                if key == "사내" : continue
                if key not in self.args.n_trip_infos.keys():
                    self.args.n_trip_infos_list.append(0)
                else:
                    self.args.n_trip_infos_list.append(self.args.n_trip_infos[key])
                    
        self.RLAgent = PGAgent(self.args)
        self.RLAgent.to(self.args.device)
        self.optimizer = torch.optim.Adam(self.RLAgent.parameters(), lr=self.args.lr)
        
    
def match_batch_num(result, result_barge):
    var1, var2, var3, var4 = result_barge.shape
    for y_idx in range(var1): # 야드 인덱스, 굳이 result1에는 안 쓰임. 이미 destination으로 매칭되어 있기 때문
        for t_idx in range(var2): # 항차 인덱스, 몇 번째 항차인지
            for s_idx in range(var3): # 
                for p_idx in range(var4):
                    block_num = result_barge[y_idx, t_idx, s_idx, p_idx]
                    if block_num == -1 : 
                        continue
                    # * 블록 번호랑, index랑 매칭
                    # print(block_num.item())
                    if p_idx == 0:
                        is_parallel = (result_barge[y_idx, t_idx, s_idx, p_idx+1] != -1)
                    elif p_idx == 1:
                        is_parallel = True
                    else:
                        is_parallel = False
                        
                    result.loc[block_num.item(), ["바지", "슬롯", "병렬"]] = [t_idx, s_idx, is_parallel]
    # print("\n\n")
    

def save_result(result, args, infos, labels_encoder, obj_loss_history=None, dl_loss_history=None):
    take_in, take_out = infos
    if obj_loss_history is not None:
        plt.plot(obj_loss_history)
        plt.savefig(os.path.join(args.save_dir, "loss.png"))
        plt.close()
        pd.Series(obj_loss_history).to_csv(os.path.join(args.save_dir, "loss.csv"))
    if dl_loss_history is not None:
        plt.plot(dl_loss_history)
        plt.savefig(os.path.join(args.save_dir, "dl_loss.png"))
        plt.close()
        pd.Series(dl_loss_history).to_csv(os.path.join(args.save_dir, "dl_loss.csv"))
    
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
    
    result = result[["vessel_id", "block", "출발지", "목적지", "바지", "슬롯", "병렬", "슬롯면적", "블록면적", "유휴면적", "슬롯길이", "슬롯너비", "슬롯높이", "블록길이", "블록너비", "블록높이"]]    
    
    blocks_at_yards = []
    for label in labels_encoder.values():
        if label == "사내": continue
        take_in_temp = result.loc[(result["출발지"] == label)]
        take_out_temp = result.loc[(result["목적지"] == label)]
        if take_in_temp.empty and take_out_temp.empty: 
            # print("Empty", label)
            continue
        blocks_at_yard = pd.concat([take_in_temp, take_out_temp], axis=0)
        if take_out_temp.shape[0] == 0:
            blocks_at_yard = blocks_at_yard.sort_values(by = ["바지", "출발지"]).reset_index(drop=True)
        else:
            blocks_at_yard = blocks_at_yard.sort_values(by = ["바지", "목적지"]).reset_index(drop=True)
        blocks_at_yards.append(blocks_at_yard)
    
    result = pd.concat(blocks_at_yards, axis=0)
    
    if obj_loss_history is not None:
        result.to_csv(os.path.join(args.save_dir, f"actions.csv"), index=False, encoding="cp949")
    return result

def train(args : Args):
        
    block_plan = pd.read_csv(os.path.join(args.input_dir, "shi_block_plan_preproc.csv"), encoding="cp949")
    # block_plan = pd.read_csv(os.path.join(args.input_dir, "shi_block_plan_preproc.csv"))

    take_in = block_plan.loc[block_plan["location"] != "사내"].copy().reset_index(drop=True)
    take_out = block_plan.loc[block_plan["location"] == "사내"].copy().reset_index(drop=True)
    
    # impossible_blocks = pd.read_csv(os.path.join(args.load_dir, f"Exp_{n}", "Ban_list.txt")) # 수리 모델에서 불가능한 경우를 선택해주게 될 경우 달라지게 될 내용입니다
    impossible_blocks = None # 일단 불가능한 블록이 없을 경우 if 문으로 해당 경우를 모두 고려할 수 있도록 해놨습니다.
    if impossible_blocks is not None and not impossible_blocks.empty:
        take_out = take_out.loc[~take_out.index.isin(impossible_blocks["Index"]-1)].reset_index(drop=True)
        
    priori_blocks = take_out.loc[~take_out["not_in"].isnull()]
    if not priori_blocks.empty:
        args.priori_conditions = []
        for idx, row in priori_blocks.iterrows():
            args.priori_conditions.extend([(idx, elem.replace(" ", "")) for elem in row["not_in"].split(",")])
    else:
        args.priori_conditions = None
        
    args.priori_conditions = pd.DataFrame(args.priori_conditions, columns=["idx", "name"])
    
    n_trip_infos_path = os.path.join(args.input_dir, "shi_times.csv")
    if os.path.exists(n_trip_infos_path):
        n_trip_infos = pd.read_csv(n_trip_infos_path, encoding="cp949")
        args.n_trip_infos = dict(zip(n_trip_infos["name"].values, n_trip_infos["number"].values))
    else:
        args.n_trip_infos = None
            
    args.n_take_in = len(take_in)
    args.n_take_out = len(take_out)
    
    # with open(os.path.join(args.save_dir, "info.txt"), "w") as f:
    #     f.write(f"Gamma to 0.1 \n \
    #         batch_size: {args.batch_size}\n \
    #         hid_dim: {args.hid_dim}\n \
    #         gamma: {args.decay}\n \
    #         pad_len: {args.pad_len}\n \
    #         n_epoch: {args.n_epoch} \
    #     ")
    with open(os.path.join(args.save_dir, "info.txt"), "w") as f:
        f.write(args.__dict__.__str__())
    
    with open(os.path.join(args.save_dir, "result.txt"), "w") as f:
        f.write("exp_num,objective,time\n")
        env = RLEnv(args)
        env.load_env()
        
        encoder_inputs, remaining_areas, yard_slots = make_input_from_loaded(take_out, take_in, env) # == env.get_state()와 같다
        
        time1 = time.time()
        obj_loss_history = []
        dl_loss_history = []
        loop = tqdm(range(args.n_epoch), leave=False)
        for epo in loop:
            # print(f"Epoch: {epo}")
            env.reset()
            for day in range(1):
                # objective = env.step((take_in.copy(), take_out.copy(), deepcopy(yard_slots)), deepcopy(encoder_inputs.to(args.device)), deepcopy(remaining_areas))
                objective = env.step((take_in.copy(), take_out.copy(), deepcopy(yard_slots)), deepcopy(encoder_inputs), deepcopy(remaining_areas))
            loss = env.update_policy()
            loop.set_description(f"Loss: {loss:.3f} / Average Reward: {objective:.3f}")
            # loss2 = env.update_policy2(len(take_in))
            # loop.set_description(f"{exp_num} / {loss:.3f} + {loss2:.3f} / {objective}")
            obj_loss_history.append(objective)
            dl_loss_history.append(loss)
            # break
        
        time2 = time.time()
        kwargs = env.get_result()
        result_df = save_result(obj_loss_history=obj_loss_history, dl_loss_history=dl_loss_history, result=kwargs, args=args, infos=(take_in, take_out), labels_encoder=env.labels_encoder_inv)

        print(f"Took {(time2 - time1):.3f} seconds")
        f.write(f"{kwargs['take_out_rewards']},{time2-time1}\n")
        
    
    
if __name__ == "__main__":
    args = Args()
    
    par_dir = os.path.abspath(os.path.join(__file__, * [os.pardir] * 1))
    # par_dir = r"D:\Dropbox\Projects_Mine\삼성중공업\code\SHI_Project"
    args.src_dir = os.path.join(par_dir, "Src")
    args.input_dir = os.path.join(par_dir, "Input")
    current_dateTime = datetime.now()
    args.save_dir = os.path.join(par_dir, "Result", f"{current_dateTime.year}_{current_dateTime.month}_{current_dateTime.day}_{current_dateTime.hour}_{current_dateTime.minute}")
    # args.save_dir = os.path.join(par_dir, "Result", f"Test")
    os.makedirs(args.save_dir, exist_ok=True)
    
    args.lr = 5e-3
    args.std_eps = 1e-3
    args.clipping_size = 10 # RNN 모델의 그래디언트 폭주 문제를 막기 위해 설정하는 파라미터입니다. 크기가 너무 작으면 업데이트가 충분히 이루어지지 않아 설정 해둔 값이 제일 좋은 걸로 확인했습니다
    args.n_epoch = 10000 # 빠르게 확인해보고 싶은 경우 200까지 돌려도 무난합니다만, Epoch 크기가 커질수록 성능이 향상됩니다
    args.batch_size = 1 # 딥러닝 모델에서 동시에 실험해보는 경우의 수입니다. 크기가 너무 커져도 분산의 크기 증가로 성능 하락을 초래 할 수 있습니다.
    # args.n_epoch = 200 # 빠르게 확인해보고 싶은 경우 200까지 돌려도 무난합니다만, Epoch 크기가 커질수록 성능이 향상됩니다
    # args.batch_size = 200 # 딥러닝 모델에서 동시에 실험해보는 경우의 수입니다. 크기가 너무 커져도 분산의 크기 증가로 성능 하락을 초래 할 수 있습니다.
    args.hid_dim = 128 # 모델의 크기
    args.emb_dim = 16
    args.decay = 0.99
    args.pad_len = 30 # 야드 슬롯의 Case를 표현하는 값입니다. 크기가 클수록 다양한 Slot들의 경우를 고려해줄 수 있지만 속도는 느려집니다.
    args.n_max_block = 100 # 최대 고려 가능한 블록 수: 수가 많아질 수록 느려집니다. 진행했던 실험은 60으로 두고 진행했습니다.
    args.max_trip = 5 # 바지선 최대 항차 수 입니다.
    args.barge_max_row = 5 # 한 바지선이 최대 수용할 수 있는 블록의 수입니다.
    args.barge_par_width = 21
    args.barge_max_length = 70
    args.alpha = 1e-3 # 목적식 파라미터 - 유휴 공간
    args.beta = -1 # 목적식 파라미터 - 바지선 개수 
    
    train(args)

# %%
