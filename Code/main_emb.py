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


class Args:
    def __init__(self) -> None:
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
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
        self.emb_dim = None
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
        # print(self.area_slots)
        self.used_area = np.sum(self.area_slots["length"] * self.area_slots["width"])
        # print(self.blocks)
        
    def _reset(self):
        self.area_slots = deepcopy(self.first_step_infos["area_slots"])
        self.blocks = deepcopy(self.first_step_infos["blocks"])
        
    def _load_evn(self):
        self.first_step_infos = {
            "area_slots": deepcopy(self.area_slots), 
            "blocks": deepcopy(self.blocks)
        }
    
    def move_to_yard_out(self, blocks: pd.DataFrame):
        self.blocks = pd.concat([self.blocks, blocks]).reset_index(drop=True)
        self.blocks["location"] = self.name
        print(self.area_slots)
        for idx, block in blocks.iterrows():
            count = self.area_slots.loc[self.area_slots["vessel_id"].isnull(), ["length", "width", "height", "location"]].groupby(["length", "width"], as_index=False).value_counts().values
            fitting_slot = np.argmin(np.abs((block["length"] * block["width"]) - (count[:, 0] * count[:, 1])))
            length, width = count[fitting_slot, 0], count[fitting_slot, 1]
            slot_idx = self.area_slots[(self.area_slots["vessel_id"].isnull()) & (self.area_slots["length"] == length) & (self.area_slots["width"] == width)].index[0]
            self.area_slots.loc[self.area_slots.index[slot_idx], "block"] = block["block"]
            self.area_slots.loc[self.area_slots.index[slot_idx], "vessel_id"] = block["vessel_id"]
        # print(self.area_slots)
        
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
        # print(self.blocks)
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
            # print(slot_info)
            taken_space = self.blocks.loc[(self.blocks["vessel_id"] == slot_info[0]) & (self.blocks["block"] == slot_info[1]), ["length", "width"]].values[0]
            # print(taken_space)
            remaining_area.append(((slot_info[2] * slot_info[3]) - (taken_space[0] * taken_space[1])) ** 2)
        # TODO: 원래 하나의 칸이였던 경우를 블록이 비어서 없어지는 경우를 고려하기 위해서 여유가 되면 id별로 다시 합쳐주기 
        return np.sum(remaining_area)

        

class PGAgent_to(nn.Module):
    def __init__(self, args: Args) -> None:
        super(PGAgent_to, self).__init__()
        self.args = args
        self.pad_len = args.pad_len
        self.n_yard = args.n_yard
        self.n_max_block = args.n_max_block
        self.n_take_out = args.n_take_out
        self.encoder = nn.LSTM(input_size=args.emb_dim * 2, hidden_size=args.hid_dim, batch_first=True)
        self.to_decoder = nn.LSTMCell(input_size=args.emb_dim*(args.emb_dim+1), hidden_size=args.hid_dim)
        # self.to_fc_block = nn.Linear(args.hid_dim*2, self.n_max_block*self.n_yard) # 400 = 40(n_block) * 10(n_yard)
        self.to_fc_block = nn.Linear(args.hid_dim, self.n_max_block*self.n_yard) # 400 = 40(n_block) * 10(n_yard)
        self.count = np.zeros((args.batch_size, self.n_take_out))
        self.to_remaining_space = np.full((args.batch_size, self.n_take_out), self.args.barge_batch_size[0] * self.args.barge_batch_size[1], dtype=float)
        
        self.emb_yard = nn.Embedding(self.n_yard, 5*(args.pad_len+1))
        self.yard_emb = nn.Embedding(self.n_yard, self.args.emb_dim)
        self.size_emb = nn.Embedding(40*40, self.args.emb_dim)
        
        self.device = args.device
        self.batch_limit = args.barge_batch_size
        # print(self.remaining_space.shape, batch_limit[0] * batch_limit[1])
        # TODO: 제원 정보 임베딩으로 변환하는 방법 적용 고려해보기
        
        self.soft_max = nn.Softmax(dim=-1)
        
    def generate_mask(self, take_out: pd.DataFrame, yard_slots: np.ndarray):
        encoder_mask = np.zeros((self.args.batch_size, take_out.shape[0], self.n_yard, self.pad_len), dtype=bool)
        
        for b_idx, block_info in enumerate(take_out[["length", "width", "height"]].values):
            mask = np.all(block_info <= yard_slots[:, :, :, :3], axis=-1)# 블록 크기가 슬롯의 모든 면보다 작으면 True, 블록이 슬롯보다 작은게 하나라도 있다면 True
            encoder_mask[:, b_idx] = mask # 
        
        # 불가능한 경우에 값들을 변경해야 하므로 위에서 구한 가능한 Case 들을 False가 되도록 Not 적용
        return torch.BoolTensor(encoder_mask).to(self.device)
    
    def calculate_batch_nums(self, block_size :np.array, yard_idx):
        """
        1안) 넓이로 무식하게
        2안) 왼쪽에 쭉 정렬, 오른쪽에 쭉 정렬하는 방식으로 채우자
        3안) 모든 경우의 수 다 해보고 가능한 케이스
        """
        batch_size = self.args.batch_size
        
        mask_l = np.zeros_like(self.count, dtype=bool) # 선택이 된 값들
        mask_l[range(batch_size), yard_idx] = True
        mask_r = np.zeros_like(self.count, dtype=bool) # count가 0인 경우
        mask_r[self.count == 0] = True
        mask = np.logical_and(mask_l, mask_r) # 선택된 값 중 count가 0인 경우는 1로 변경
        self.count[mask] = 1 
        # print(self.remaining_space.shape)
        # print(self.remaining_space[range(batch_size), yard_idx].shape, (block_size[:, :, 0] * block_size[:, :, 1]).shape)
        self.to_remaining_space[range(batch_size), yard_idx] -= (block_size[:, :, 0] * block_size[:, :, 1]).reshape(-1)
        
        mask_l = np.zeros_like(self.count, dtype=bool) # 선택이 된 값들
        mask_l[range(batch_size), yard_idx] = True
        mask_r = np.zeros_like(self.count, dtype=bool) # 남는 공간이 -1인 경우
        mask_r[self.to_remaining_space < 0] = True
        mask = np.logical_and(mask_l, mask_r) # 선택이 된 값들 중 블록의 허용 범위가 넘어버린 경우는 카운트를 늘리고 크기는 초기화
        block_mask = np.any(mask, axis=-1) # 값을 새로 생성할 때는 이미 마스킹이 된 상태여야 하므로 따로 개수를 확인해줘야 함
        # print(block_size[block_mask, :, 0] * block_size[block_mask, :, 1])
        self.count[mask] += 1
        self.to_remaining_space[mask] = (np.full((np.sum(block_mask), 1), fill_value=self.batch_limit[0] * self.batch_limit[1], dtype=float) \
            - block_size[block_mask, :, 0] * block_size[block_mask, :, 1]).reshape(-1) # 한도를 넘으면 값 초기화
        
        return self.count.sum(axis=-1)
    
    def generate_encoder_input(self, take_out : pd.DataFrame):
        yard_emb = self.yard_emb(torch.arange(self.n_yard).to(self.device)).to(self.device)
        emb_idx = torch.LongTensor(25 * np.ceil(take_out["length"]).astype(int) + np.ceil(take_out["width"]).astype(int)).to(self.device)
        # print(emb_idx)
        take_out_emb = self.size_emb(emb_idx)
        
        encoder_inputs = []
        for y_emb in yard_emb:
            for t_emb in take_out_emb:
                encoder_inputs.append(torch.cat([y_emb, t_emb], -1).reshape(-1))
        
        encoder_inputs = torch.stack(encoder_inputs)
        
        return encoder_inputs
    
    def embed_input(self, block_emb_info : np.array, slot_info):
        """
        block_emb_info: [BatchSize, N_remaining, 5]
        slot_info: [BatchSize, N_yard, pad_len, 5]
        """
        emb_idx = torch.LongTensor(np.ceil(block_emb_info[:, :, 0]).astype(int) * np.ceil(block_emb_info[:, :, 1]).astype(int)).to(self.device)
        embedded_block = self.size_emb(emb_idx)
        embedded_block = torch.sum(embedded_block, dim=1).unsqueeze(1)
        
        slot_idx = torch.LongTensor(np.ceil(slot_info[:, :, :, 0]).astype(int) * np.ceil(slot_info[:, :, :, 1]).astype(int)).to(self.device)
        # print("Slot idx shape: ", slot_idx.shape)
        embedded_slot = self.size_emb(slot_idx)
        # print("Embed shape: ", embedded_slot.shape)
        
        count_of_slots = torch.FloatTensor(slot_info[:, :, :, -1]).to(self.device).unsqueeze(-1)
        embedded_slot = torch.sum(embedded_slot * count_of_slots, dim=2).transpose(1, 2) # 제원 정보 임베딩 후 개수 만큼 곱해주고 다 합해주기
        # print(embedded_slot.shape)
        
        yard_idx = torch.LongTensor(slot_info[:, :, :, -2]).to(self.device)
        yard_emb = self.yard_emb(torch.arange(self.n_yard).to(self.device)).unsqueeze(0).repeat(self.args.batch_size, 1, 1)
        
        embedded_yard = torch.matmul(embedded_slot, yard_emb)
        # print("Embed shape: ", embedded_block.shape, embedded_yard.shape)
        
        input0 = torch.cat([embedded_block, embedded_yard], dim=1)
        return input0.reshape(self.args.batch_size, -1)
        
    
    def forward(self, infos: List[pd.DataFrame], encoder_inputs =None, remaining_area = None):
        batch_size = self.args.batch_size
        pad_len = self.args.pad_len
        """
        Feature information
        take_out, take_in, feat_vec: ["length", "width", "height", "weight", "location"]
        slot_info: [length  width  height  location  count]
        """
        take_in, take_out, yard_slots = infos
        # print(take_in.shape, take_out.shape, yard_slots.shape)
        yard_slots = np.expand_dims(yard_slots, axis=0).repeat(batch_size, axis=0)
        encoder_mask_whole = self.generate_mask(take_out, yard_slots)
        encoder_inputs = self.generate_encoder_input(take_out)
        # print("Encoder input shape: ", encoder_inputs.shape)
        
        # 돌기 전에 초기화
        hids, (h0, c0) = self.encoder(encoder_inputs)
        self.count = np.zeros((batch_size, self.n_take_out))
        self.to_remaining_space = np.full((batch_size, self.n_take_out), self.batch_limit[0] * self.batch_limit[1], dtype=float)
        
        input0 = torch.zeros(batch_size, self.args.emb_dim*(self.args.emb_dim+1)).to(self.device)
        h0, c0 = torch.zeros((batch_size, self.args.hid_dim)), torch.zeros((batch_size, self.args.hid_dim))
        
        block_mask = torch.zeros((batch_size, self.n_take_out, 1), dtype=bool) # 제외 시키고 싶은 경우를 True
        # print(block_mask.repeat(1, 1, self.n_yard).shape)
        
        probs_history = []
        rewards_history = []
        reward_parts = np.zeros((batch_size, self.n_take_out), dtype=float)
        reward_wholes = np.zeros((batch_size, self.n_take_out), dtype=float)
        action_history = []
        
        for idx in range(self.n_take_out):
            
            # ________________________________________________
            encoder_mask = ~torch.any(encoder_mask_whole, dim=-1)
            # print("Inputs shape: ", input0.shape)
            h0, c0 = self.to_decoder(input0, (h0, c0))
            out = self.to_fc_block(h0)
            # ________________________________________________
            out[:, self.n_yard*self.n_take_out:] = -20000 # 블록 개수보다 많을 필요는 없으니 마스킹
            out[:, :self.n_yard*self.n_take_out][block_mask.repeat(1, 1, self.n_yard).reshape(batch_size, -1)] = -10000 # 선택된 블록이 다시 선택되지 않도록 마스킹
            out[:, :self.n_yard*self.n_take_out][encoder_mask.reshape(batch_size, -1)] = -10000 # 보내고자 하는 블록이 적치장에 안 맞으면 마스킹
            
            feasible_mask = torch.ones_like(out, dtype=torch.bool)
            # print(feasible_mask.shape)
            feasible_mask[:, self.n_yard*self.n_take_out:] = False
            feasible_mask[:, :self.n_yard*self.n_take_out][block_mask.repeat(1, 1, self.n_yard).reshape(batch_size, -1)] = False
            feasible_mask[:, :self.n_yard*self.n_take_out][encoder_mask.reshape(batch_size, -1)] = False
            
            probs = self.soft_max(out)
            m = Categorical(probs)
            action = m.sample()
            
            # ________________________________________________
            # 다음 스텝의 마스킹을 위해서 값들 불러오기
            block_selection, yard_selection = torch.div(action, self.n_yard, rounding_mode='trunc').detach().cpu().numpy(), (action % self.n_yard).detach().cpu().numpy()
            isFeasible = feasible_mask[range(batch_size), action].detach().cpu().numpy()
            block_size_info = np.expand_dims(take_out[["length", "width", "height", "weight", "location"]].values, axis=0).repeat(batch_size, axis=0)
            
            block_size = np.expand_dims(block_size_info[range(batch_size), block_selection], axis=1)
            slot_info = yard_slots[range(batch_size), yard_selection].copy()
                        
            block_emb_info = block_size_info[~block_mask.detach().cpu().numpy().repeat(5, axis=-1)].reshape(batch_size, -1, 5)
            input0 = self.embed_input(block_emb_info, yard_slots)
            
            slot_info[slot_info[:, :, 0] == 0] = 10000 # 슬롯이 비어있는 경우에 값 최대화
            slot_info[np.any(block_size[:, :, 0:3] > slot_info[:, :, 0:3], axis=-1)] = 10000 # 크기가 큰 경우는 선택이 안 되도록 최대화
            yard_offset = np.argmax((block_size[:, :, 0] * block_size[:, :, 1]) - (slot_info[:, :, 0] * slot_info[:, :, 1]), axis=-1)
            
            # print(encoder_mask[0].reshape(self.n_take_out, self.n_yard))
            yard_slots[range(batch_size), yard_selection, yard_offset, -1] -= 1
            yard_slots[yard_slots[:, :, :, -1] == 0] = 0 # 해당 크기의 슬롯이 남아있지 않으면 선택이 안 되도록 제거 
            slot_size = slot_info[range(batch_size), yard_offset] # 선택된 슬롯의 크기 불러오기
            reward_part = (slot_size[:, 0] * slot_size[:, 1] - (block_size[:, :, 0] * block_size[:, :, 1]).reshape(-1)) ** 2 * (np.where(isFeasible, 1, -1)) # 불가능한 선택이면 Reward 최소화
            # reward_part = (slot_size[:, 0] * slot_size[:, 1] - (block_size[:, :, 0] * block_size[:, :, 1]).reshape(-1)) ** 2 * isFeasible # 불가능한 선택이면 Reward 최소화
            
            encoder_mask_whole[range(batch_size), block_selection] = False # False가 제외 시킬 대상
            encoder_mask_whole = encoder_mask_whole.permute(0, 2, 3, 1)
            encoder_mask_whole[torch.BoolTensor(yard_slots[:, :, :, -1] == 0).unsqueeze(-1).repeat(1, 1, 1, self.n_take_out)] = False
            encoder_mask_whole = encoder_mask_whole.permute(0, 3, 1, 2)
            
            
            # ________________________________________________
            # Reward를 저장
            current_remaining = yard_slots[:, :, :, 0] * yard_slots[:, :, :, 1]
            reward_whole = reward_part + np.sum(np.sum(current_remaining, axis=-1), axis=-1)
            batch_num = self.calculate_batch_nums(block_size, yard_selection)
            
            # store current step
            reward_parts[:, idx] = reward_part
            reward_wholes[:, idx] = reward_whole
            probs_history.append(m.log_prob(action))
            action_history.append(np.hstack([np.vstack((block_selection, yard_selection)).T, slot_size]))
            rewards_history.append(np.sum(reward_wholes, axis=-1) * 1e-2 - batch_num)
            block_mask[range(batch_size), block_selection] = True
            # print((yard_slots[:, :, :, -1] == 0).shape, encoder_mask_whole[:, :].shape)
        
        probs_history = torch.stack(probs_history).transpose(1, 0).to(self.device)
        rewards_history = torch.FloatTensor(rewards_history).transpose(1, 0).to(self.device)
        action_history = torch.LongTensor(action_history).transpose(1, 0).to(self.device)
        
        return probs_history, rewards_history, action_history, np.sum(reward_wholes, axis=-1)
    
    
class PGAgent_ti(nn.Module):
    def __init__(self, args: Args) -> None:
        super(PGAgent_ti, self).__init__()
        self.args = args
        self.n_yard = args.n_yard
        self.n_max_block = args.n_max_block
        self.n_take_in = args.n_take_in
        self.hid_dim = args.hid_dim // 4
        self.decoder = nn.LSTMCell(input_size=5*(args.pad_len+1), hidden_size=self.hid_dim)
        self.fc_block = nn.Linear(self.hid_dim, self.n_max_block) # 400 = 40(=n_block) * 10(=n_yard)
        
        self.device = args.device
        self.batch_limit = args.barge_batch_size
        
        self.soft_max = nn.Softmax(dim=-1)
        
    
    def calculate_batch_nums(self, block_size :np.array, yard_idx):
        """
        1안) 넓이로 무식하게
        2안) 왼쪽에 쭉 정렬, 오른쪽에 쭉 정렬하는 방식으로 채우자
        3안) 모든 경우의 수 다 해보고 가능한 케이스
        """
        batch_size = self.args.batch_size
        mask_l = np.zeros_like(self.count, dtype=bool) # 선택이 된 값들
        mask_l[range(batch_size), yard_idx] = True
        mask_r = np.zeros_like(self.count, dtype=bool) # count가 0인 경우
        mask_r[self.count == 0] = True
        mask = np.logical_and(mask_l, mask_r) # 선택된 값 중 count가 0인 경우는 1로 변경
        self.count[mask] = 1 
        self.remaining_space[range(batch_size), yard_idx] -= (block_size[:, :, 0] * block_size[:, :, 1]).reshape(-1)
        
        mask_l = np.zeros_like(self.count, dtype=bool) # 선택이 된 값들
        mask_l[range(batch_size), yard_idx] = True
        mask_r = np.zeros_like(self.count, dtype=bool) # 남는 공간이 -1인 경우
        mask_r[self.remaining_space < 0] = True
        mask = np.logical_and(mask_l, mask_r) # 선택이 된 값들 중 블록의 허용 범위가 넘어버린 경우는 카운트를 늘리고 크기는 초기화
        block_mask = np.any(mask, axis=-1) # 값을 새로 생성할 때는 이미 마스킹이 된 상태여야 하므로 따로 개수를 확인해줘야 함
        self.count[mask] += 1
        self.remaining_space[mask] = (np.full((np.sum(block_mask), 1), fill_value=self.batch_limit[0] * self.batch_limit[1], dtype=float) \
            - block_size[block_mask, :, 0] * block_size[block_mask, :, 1]).reshape(-1) # 한도를 넘으면 값 초기화
        
        return self.count.sum(axis=-1)
        
    
    def forward(self, infos: List[pd.DataFrame]):
        batch_size = self.args.batch_size
        pad_len = self.args.pad_len
        """
        Feature information
        take_out, take_in, feat_vec: ["length", "width", "height", "weight", "location"]
        slot_info: [length  width  height  location  count]
        """
        take_in, _, yard_slots = infos
        yard_slots = np.expand_dims(yard_slots, axis=0).repeat(batch_size, axis=0)
        
        # 돌기 전에 초기화
        self.count = np.zeros((batch_size, self.n_take_in))
        self.remaining_space = np.full((batch_size, self.n_take_in), self.batch_limit[0] * self.batch_limit[1], dtype=float)
        
        
        block_mask = torch.zeros((batch_size, self.n_take_in), dtype=bool) # 제외 시키고 싶은 경우를 True
        
        probs_history = []
        rewards_history = []
        action_history = []
        
        input0 = torch.zeros(batch_size, 5*(pad_len+1)).to(self.device)
        h0, c0 = torch.zeros((batch_size, self.hid_dim), dtype=torch.float32).to(self.device), torch.zeros((batch_size, self.hid_dim), dtype=torch.float32).to(self.device)
        
        for idx in range(self.n_take_in):
            out, c0 = self.decoder(input0, (h0, c0))
            out = self.fc_block(out)
            
            # Masking
            out[:, self.n_take_in:] = -10000 # 블록 개수보다 많을 필요는 없으니 마스킹
            out[:, :self.n_take_in][block_mask] = -10000 # 선택된 블록이 다시 선택되지 않도록 마스킹
            
            probs = self.soft_max(out)
            m = Categorical(probs)
            action = m.sample()
            block = action.detach().cpu().numpy()
            
            # 다음 스텝의 마스킹을 위해서 값들 불러오기
            block_size = np.expand_dims(take_in[["length", "width", "height", "weight", "location"]].values, axis=0).repeat(batch_size, axis=0)
            block_size = np.expand_dims(block_size[range(batch_size), block], axis=1)
            
            yard = block_size[:, :, -1].reshape(-1).astype(int) -1
            slot_info = yard_slots[range(batch_size), yard]
            
            input0 = torch.cat([torch.FloatTensor(block_size), torch.FloatTensor(slot_info)], dim=1).reshape(batch_size, -1).to(self.device)
            
            # slot 중에 제일 크기가 비슷한 경우(yard_offset)을 고르고 다시 선택되지 않도록 남은 수 -1 적용
            batch_num = self.calculate_batch_nums(block_size, yard)
            
            rewards_history.append(batch_num)
            probs_history.append(m.log_prob(action))
            action_history.append(block)
        
        probs_history = torch.stack(probs_history).transpose(1, 0).to(self.device)
        rewards_history = torch.FloatTensor(rewards_history).transpose(1, 0).to(self.device)
        action_history = torch.LongTensor(action_history).transpose(1, 0).to(self.device)
        
        return probs_history, rewards_history, action_history
        

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
        # self.min_result = (float("inf"), None, None) # Value of reward and combination of actions
        
        
    def get_state(self, possible_take_out: pd.DataFrame, take_in: pd.DataFrame):
        
        # Encoder에 Input으로 들어갈수록 
        possible_take_out["location"] = possible_take_out["location"].map(self.labels_encoder)
        # encoder_inputs =[]
        yard_slots = []
        
        for name, yard in self.yards_out.items():
            state_part_ = yard.area_slots.copy()
            state_part_["location"] = self.labels_encoder[name]
            state_part_.loc[state_part_["height"] == float("inf"), "height"] = 1000
            count = state_part_.loc[state_part_["vessel_id"].isnull(), ["length", "width", "height", "location"]].groupby(["length", "width"], as_index=False).value_counts()
            yard_t = torch.FloatTensor(count.values)
            count_ = np.zeros((self.args.pad_len, 5))
            count_[:count.shape[0], :] = count
            yard_slots.append(count_)
            for _, block in possible_take_out.iterrows():
                block_t = torch.FloatTensor([block[["length", "width", "height", "location","weight"]].values])
                pad_t = torch.zeros((self.args.pad_len-len(count),5))
                input = torch.concat([block_t, yard_t, pad_t], dim=0)
                # encoder_inputs.append(input)
                
        # encoder_inputs = torch.stack(encoder_inputs)
        # encoder_inputs = encoder_inputs.reshape(encoder_inputs.shape[0], -1)
        yard_slots = np.array(yard_slots)
        
        # return encoder_inputs, yard_slots, None
        return None, yard_slots, None
            
    def step(self, infos: List[pd.DataFrame], inputs: torch.Tensor, remaining_areas: np.array):
        """
        infos: infos about block to be taken out, infos about block to be taken in, infos about slots in the yard 
        inputs: combination of block infos and yard slots
        """
        
        # TODO: Implement the block moving to the yard
        take_in, take_out, yard_slots = infos
        
        take_in["location"] = take_in["location"].map(self.labels_encoder).astype(float)
        take_out["location"] = take_out["location"].map(self.labels_encoder).astype(float)
        
        probs, rewards, actions, obj  = self.Agent_to((take_in.reset_index(drop=True), take_out.reset_index(drop=True), yard_slots), inputs, remaining_areas)
        self.probs, self.rewards = probs, rewards
        
        probs2, rewards2, actions2  = self.Agent_ti((take_in.reset_index(drop=True), take_out.reset_index(drop=True), yard_slots))
        self.probs2, self.rewards2 = probs2, rewards2
        actions[:, :, 1] += 1 # 0은 사내라서 사외를 표현하려면 1씩 더해줘야 함
            
        max_reward, max_actions, max_obj = torch.max(rewards[:, -1]).detach().cpu().numpy(), actions[torch.argmax(rewards[:, -1])].detach().cpu().numpy(), obj[torch.argmax(rewards[:, -1])]
        if max_reward > self.max_result[0]:
            self.max_result = (max_reward, max_actions, max_obj)
        
        min_reward2, min_actions2 = torch.max(rewards2[:, -1]).detach().cpu().numpy(), actions2[torch.argmin(rewards2[:, -1])].detach().cpu().numpy()
        if min_reward2 < self.min_result2[0]:
            self.min_result2 = (min_reward2, min_actions2)
        """
        Objective: maximize a*여유공간 - b*배치 수
        """
        return torch.mean(rewards[:, -1]).detach().cpu().numpy()
    
    def reset(self):
        self.whole_block = self.first_step_infos["whole_block"]
        for yard in self.yards_out.values():
            yard._reset()
            
    def get_result(self):
        # return self.min_result
        return self.max_result
    
        
    def update_policy(self):
        policy_loss = torch.zeros((len(self.probs)))
        returns = torch.zeros_like(self.rewards)
        # print(returns.shape)
        for idx in range(args.n_take_out -1, -1, -1):
            # R = np.log1p(r) + 0.99 * R
            if idx == args.n_take_out-1:
                # returns[:, 0] = self.rewards[:, idx] # 처음에는 그냥 리워드 값
                returns[:, 0] = (-1) * self.rewards[:, idx] # 처음에는 그냥 리워드 값
            else:
                returns[:, args.n_take_out-1-idx] = (self.rewards[:, idx] + self.args.decay * returns[:, args.n_take_out-idx-2]) * (-1)
                
        # norm_returns = torch.nn.functional.normalize(returns, dim=-1)
        mean = torch.mean(returns.reshape(-1))
        std = torch.std(returns.reshape(-1))
        norm_returns = (returns - mean) / std
        policy_loss = torch.mul(self.probs, norm_returns)
        self.optimizer.zero_grad()
        policy_loss = torch.mean(policy_loss)
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Agent_to.parameters(), 20)
        self.optimizer.step()
        
        # ______________________________________
        policy_loss = torch.zeros((len(self.probs2)))
        returns = torch.zeros_like(self.rewards2)
        
        for idx in range(args.n_take_in-1, -1, -1):
            # R = np.log1p(r) + 0.99 * R
            if idx == args.n_take_in-1:
                returns[:, 0] = (-1) * self.rewards2[:, idx] # 처음에는 그냥 리워드 값
            else:
                returns[:, args.n_take_in-1-idx] = (self.rewards2[:, idx] + self.args.decay * returns[:, args.n_take_in-idx-2]) * (-1)
                
                
        mean = torch.mean(returns.reshape(-1))
        std = torch.std(returns.reshape(-1))
        norm_returns = (returns - mean) / std
        policy_loss = torch.mul(self.probs2, norm_returns)
        self.optimizer2.zero_grad()
        policy_loss = torch.mean(policy_loss)
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Agent_to.parameters(), 20)
        self.optimizer2.step()
        
        self.saved_log_probs = []
        self.rewards2 = []
        
        return policy_loss.item()
    
    
    # def save_env(self, save_dir, exp_num, infos):
    #     whole_block = deepcopy(self.first_step_infos["whole_block"])
    #     whole_block.to_csv(os.path.join(save_dir, f"{exp_num}_whole_block.csv"), index=False)
        
    #     for name, yard in self.yards_out.items():
    #         yard.area_slots.to_csv(os.path.join(save_dir, f"{exp_num}_{name}.csv"), index=False)
        
    #     take_in, take_out, yard_slots = infos
    #     take_in.to_csv(os.path.join(save_dir, f"{exp_num}_shi_take_in.csv"), index=False)
    #     take_out.to_csv(os.path.join(save_dir, f"{exp_num}_shi_take_out.csv"), index=False)
    #     pd.DataFrame(yard_slots).to_csv(os.path.join(save_dir, f"{exp_num}_yard_slots.csv"), index=False)
        
    #     self.Agent_to = PGAgent_to(n_yard_out=len(self.yards_out), n_max_block=150, n_take_out=len(take_out))
    #     self.Agent_to.to(self.args.device)
    #     self.optimizer = torch.optim.Adam(self.Agent_to.parameters(), lr=5e-3)
        
    def load_env(self):
        self.probs = []
        self.rewards = []
        self.probs2 = []
        self.rewards2 = []
        self.max_result = (float("-inf"), None, None) 
        self.min_result2 = (float("inf"), None, None)
        
        load_dir = self.args.load_dir
        exp_num = self.args.exp_num
        self.whole_block = pd.read_csv(os.path.join(load_dir, f"whole_block.csv"), encoding="cp949")
        self.first_step_infos["whole_block"] = deepcopy(self.whole_block)
        
        yard_out_names = [os.path.basename(name).split(".")[0] for name in glob(os.path.join(load_dir, "*.csv"))]
        yard_out_names = [name for name in yard_out_names if name not in ["whole_block", "shi_take_in", "shi_take_out", "DB_empty_slots", "DB_whole_block"]]
        
        # print("Yard outside names: ", yard_out_names)
        # quit()
        
        for name in yard_out_names:
            area_slots = pd.read_csv(os.path.join(load_dir, f"{name}.csv"), encoding="cp949")
            
            self.yards_out[name] = YardOutside(args=self.args, name=name, area_slots= area_slots, blocks=self.whole_block.loc[self.whole_block["location"] == name].copy().reset_index(drop=True))
            self.yards_out[name]._load_evn()
            # print(yard.blocks)
        
        self.Agent_to = PGAgent_to(self.args)
        self.Agent_to.to(self.args.device)
        self.optimizer = torch.optim.Adam(self.Agent_to.parameters(), lr=1e-3)
        
        self.Agent_ti = PGAgent_ti(args=self.args)
        self.Agent_ti.to(self.args.device)
        self.optimizer2 = torch.optim.Adam(self.Agent_ti.parameters(), lr=1e-3)
        
        self.labels_encoder = dict(zip(yard_out_names, range(1, len(yard_out_names)+1)))
        self.labels_encoder["사내"] = 0
        self.labels_encoder_inv = dict(zip(self.labels_encoder.values(), self.labels_encoder.keys()))
        
    def load_model(self):
        
        self.Agent_to = PGAgent_to(self.args)
        self.Agent_to.to(self.args.device)
        self.Agent_to.load_state_dict(torch.load(os.path.join(self.args.save_dir, "to_model.pt")))
        # print(self.Agent_to.n_take_out)
        self.optimizer = torch.optim.Adam(self.Agent_to.parameters(), lr=1e-3)
        
        self.Agent_ti = PGAgent_ti(args=self.args)
        self.Agent_ti.to(self.args.device)
        self.Agent_ti.load_state_dict(torch.load(os.path.join(self.args.save_dir, "ti_model.pt")))
        self.optimizer2 = torch.optim.Adam(self.Agent_ti.parameters(), lr=1e-3)
    
def match_batch_num(env, result):
    
    remaining_spaces = np.full(len(env.yards_out), 1300)
    batch_count_by_yard = np.zeros(len(env.yards_out))
    batch_idxs = []
    for _, row in result.iterrows():
        block_space = row["block_space"]
        destination = env.labels_encoder[row["destination"]] -1
        remaining_spaces[destination] -= block_space
        if batch_count_by_yard[destination] == 0 :
            batch_count_by_yard[destination] += 1
            batch_idxs.append("{}_{}".format(destination, batch_count_by_yard[destination]))
        else:
            if remaining_spaces[destination] <= 0:
                remaining_spaces[destination] = 1300 - block_space
                batch_count_by_yard[destination] += 1
                batch_idxs.append("{}_{}".format(destination, batch_count_by_yard[destination]))
            else:
                batch_idxs.append("{}_{}".format(destination, batch_count_by_yard[destination]))
    return batch_idxs
    
    
def train(args : Args):
    
    
    with open(os.path.join(args.save_dir, "info.txt"), "w") as f:
        f.write(f"Gamma to 0.1 \n \
            batch_size: {args.batch_size}\n \
            hid_dim: {args.hid_dim}\n \
            gamma: {args.decay}\n \
            pad_len: {args.pad_len}\n \
            n_epoch: {args.n_epoch} \
        ")
    
    
    with open(os.path.join(args.save_dir, "result.txt"), "w") as f:
        env = RLEnv(args)
        for exp_num in range(args.n_exp):
            args.exp_num = exp_num
            preprocess(par_dir=par_dir, src_dir=args.src_dir, data_dir=args.data_dir, isExp=True)
            take_in = pd.read_csv(os.path.join(args.load_dir, f"shi_take_in.csv"), encoding="cp949")
            take_out = pd.read_csv(os.path.join(args.load_dir, f"shi_take_out.csv"), encoding="cp949")
            args.n_take_in = len(take_in)
            args.n_take_out = len(take_out)
            print(f"n_take_in :{args.n_take_in}, n_take_out : {args.n_take_out}")
        
            f.write("exp_num,objective,time\n")
            env.load_env()
            if exp_num != 0:
                env.load_model()
            
            encoder_inputs, remaining_areas, yard_slots = make_input_from_loaded(take_out, take_in, env)
            
            time1 = time.time()
            obj_loss_history = []
            loop = tqdm(range(args.n_epoch))
            for epo in loop:
                env.reset()
                for day in range(1):
                    # print(f"\nDay: {day} / {n_take_out}, {n_take_in} ")
                    # objective = env.step((take_in.copy(), take_out.copy(), deepcopy(yard_slots)), deepcopy(encoder_inputs.to(args.device)), deepcopy(remaining_areas))
                    objective = env.step((take_in.copy(), take_out.copy(), deepcopy(yard_slots)), None, deepcopy(remaining_areas))
                loss = env.update_policy()
                loop.set_description(f"{args.exp_num} / {loss:.3f} / {objective:.3f}")
                # loss2 = env.update_policy2(len(take_in))
                # loop.set_description(f"{exp_num} / {loss:.3f} + {loss2:.3f} / {objective}")
                obj_loss_history.append(objective)
                # break
                
            time2 = time.time()
            plt.plot(obj_loss_history)
            plt.savefig(os.path.join(args.save_dir, "{}_loss.png".format(args.exp_num)))
            plt.close()
            
            result_val, result_actions, result_obj = env.get_result()
            result1 = take_out.iloc[result_actions[:, 0]]
            result1 = result1[[col for col in result1.columns if col != "location"]]
            result1["destination"] = result_actions[:, 1]
            result1["destination"] = result1["destination"].map(env.labels_encoder_inv)
            result1["slot_length"] = result_actions[:, 2]
            result1["slot_width"] = result_actions[:, 3]
            result1["slot_height"] = result_actions[:, 4]
            result1["slot_space"] = result1["slot_length"] * result1["slot_width"]
            result1["block_space"] = result1["length"] * result1["width"]
            result1["empty_space"] = result1["slot_space"] - result1["block_space"]
            result1["Batch"] = match_batch_num(env, result1)
                
            
            result_actions2 = env.min_result2[1]
            result2 = take_in.iloc[result_actions2]
            result2["block_space"] = result2["length"] * result2["width"]
            result2.rename(columns={"location": "destination"}, inplace=True)
            result2["Batch"] = match_batch_num(env, result2)
            result2["destination"] = "사내"
            
            result = pd.concat([result1, result2])
            result["Batch"] = result["Batch"].map(dict(zip(result["Batch"].unique(), range(len(result["Batch"].unique())))))
            result["Batch"] = result["Batch"].astype(int)
            result.sort_values(by="Batch",ascending=True, inplace=True)
            result.rename(columns={
                "destination": "목적지", 
                "slot_length": "슬롯길이", 
                "slot_width": "슬롯너비", 
                "slot_height": "슬롯높이", 
                "Batch": "배치번호", 
                "slot_space": "슬롯면적",
                "block_space": "블록면적", 
                "length": "블록길이", 
                "width": "블록너비", 
                "height": "블록높이", 
                "empty_space": "유휴면적"}, inplace=True)
            result = result[["vessel_id", "block", "목적지", "배치번호", "슬롯면적", "블록면적", "유휴면적", "슬롯너비", "슬롯길이", "슬롯높이", "블록길이", "블록너비", "블록높이"]]
            
            # result.loc["reward", "location"] = result_obj
            # result.loc["time", "location"] = np.round((time2 - time1) / 60, 3)
            print(f"Took {(time2 - time1):.3f} seconds")
            result.to_csv(os.path.join(args.save_dir, f"actions_{args.exp_num}.csv"), index=False, encoding="cp949")
            
            f.write(f"{result_obj},{time2-time1}\n")
            # break
        
            torch.save(env.Agent_to.state_dict() , os.path.join(args.save_dir, "to_model.pt"))
            torch.save(env.Agent_ti.state_dict() , os.path.join(args.save_dir, "ti_model.pt"))

def inference():
    pass
    
if __name__ == "__main__":
    args = Args()
    
    par_dir = os.path.abspath(os.path.join(__file__, * [os.pardir] * 1))
    args.src_dir = os.path.join(par_dir, "Src")
    args.result_dir = os.path.join(par_dir, "Result")
    args.data_dir = os.path.join(par_dir, "Data")
    
    # isExp는 실험을 하는 상황을 가정하므로 반출될 블록과 반입될 블록을 직접 기입하는 게 아니라 랜덤으로 블러오는 상황을 가정한 겁니다. 
    # 랜덤이 주어진 입력 값을 사용하고 싶으시다면 isExp 인자를 False로 바꿔서 넣어주면 됩니다.
    # preprocess(par_dir=par_dir, src_dir=src_dir, data_dir=data_dir, isExp=False)
    
    current_dateTime = datetime.now()
    args.save_dir = os.path.join(args.result_dir, f"{current_dateTime.year}_{current_dateTime.month}_{current_dateTime.day}_{current_dateTime.hour}_{current_dateTime.minute}_inf")
    # args.save_dir = os.path.join(result_dir, "test")
    os.makedirs(args.save_dir, exist_ok=True)
    
    args.load_dir = args.data_dir
    os.makedirs(args.load_dir, exist_ok=True)
    
    # args.exp_num = None
    args.batch_size = 512 # 딥러닝 모델에서 동시에 실험해보는 경우의 수입니다. GPU가 있을 경우 비디오 RAM에 공간을 차지하게 됩니다.
    args.hid_dim = 256
    args.emb_dim = 16
    args.decay = 0.9
    args.pad_len = 50 # 야드 슬롯의 Case를 표현하는 값입니다. 크기가 클수록 다양한 Slot들의 경우를 고려해줄 수 있지만 속도는 느려집니다.
    args.n_epoch = 200 # 빠르게 확인해보고 싶은 경우 200까지 돌려도 무난합니다만, Epoch 크기가 커질수록 성능이 향상됩니다
    args.barge_batch_size = (65, 20) # 바지선을 표현하는 Batch의 크기 제한입니다. 우선은 너비로 고려했기 때문에 본 값이 곱해진 1300으로 고려가 될 거 같습니다.
    args.n_exp = 100
    
    db_yard_out = pd.read_csv(os.path.join(args.data_dir, "DB_empty_slots.csv"), encoding="cp949")
    args.n_yard = len(db_yard_out["location"].unique()) # 사외 적치장의 수 입니다.
    args.n_max_block = 100 # 최대 고려 가능한 블록 수: 수가 많아질 수록 느려집니다. 진행했던 실험은 60으로 두고 진행했습니다.
    
    # n_epoch = 100
    train(args)



#%%
# import torch

# temp_reward = torch.arange(50).reshape(5, 10)
# returns = temp_reward.cumsum(dim=1).flip(dims=[1])
# decay = torch.tensor([0.5**step for step in range(temp_reward.shape[1])], dtype=torch.float32)
# print(decay[0].reshape(-1 ,1).repeat(5, 1).shape)
# print(returns[:, 0] * decay[0].reshape(-1 ,1).repeat(5, 1))
# print(torch.sum(temp_reward[:, 0:] * decay[0]))
# temp_reward = torch.concat([torch.sum(temp_reward[:, i:] * decay[i]) for i in range(temp_reward.shape[1])], dim=1)
# %%
