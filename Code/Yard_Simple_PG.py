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

par_dir = os.path.abspath(os.path.join(__file__, * [os.pardir] * 2))
src_dir = os.path.join(par_dir, "Src")
result_dir = os.path.join(par_dir, "Result")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

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

def sample_dataset(n_blocks):
    sizes = pd.read_csv(os.path.join(src_dir, "block_size.csv")) 
    sizes = sizes.loc[sizes["weight"] > 0.0, :]
    sizes.dropna(inplace=True)
    sampled_blocks = sizes.sample(n_blocks)
    result = convert_size(sampled_blocks)
    
    return result.reset_index(drop=True)


class Yard:
    def __init__(self, name : str, area_slots : Dict, blocks : pd.DataFrame = None):
        self.name = name
        self.block_count_limit = deepcopy(area_slots)
        self.blocks_count_by_size : pd.DataFrame = pd.DataFrame(None, index=list(range(10, 31, 5)), columns=list(range(10, 31, 5)))
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
    def __init__(self, name: str, area_slots : pd.DataFrame, blocks : pd.DataFrame = None):
        super().__init__(name, area_slots, blocks)
        self.orig_slots = area_slots.copy()
        # print(self.orig_slots.columns)
        # print(self.orig_slots)
        self.orig_slots["id"] = range(len(area_slots))
        self.area_slots = area_slots[["vessel_id", "block", "length", "width_max","height", "location"]].rename(columns={"width_max": "width"})
        self.area_slots["id"] = range(len(area_slots))
        self.columns = self.area_slots.columns
        self.used_area = np.sum(self.area_slots["length"] * self.area_slots["width"])
        self.blocks["location"] = self.name
        # print(self.blocks)
        # self._update_slot()
        self.first_step_infos = {
            "area_slots": deepcopy(self.area_slots), 
            "blocks": deepcopy(self.blocks)
        }
        
    def _reset(self):
        # if self.name == "덕곡1":
        #     print("Before reset: \n", self.area_slots)
        self.area_slots = deepcopy(self.first_step_infos["area_slots"])
        
        # if self.name == "덕곡1":
        #     print("After reset: \n", self.area_slots)
        self.blocks = deepcopy(self.first_step_infos["blocks"])
        

    def move_to_yard_out(self, blocks: pd.DataFrame):
        # print(blocks.shape[0])
        self.blocks = pd.concat([self.blocks, blocks]).reset_index(drop=True)
        self.blocks["location"] = self.name
        # print(self.area_slots)
        for idx, block in blocks.iterrows():
            count = self.area_slots.loc[self.area_slots["vessel_id"].isnull(), ["length", "width", "height", "location"]].groupby(["length", "width"], as_index=False).value_counts().values
            fitting_slot = np.argmin(np.abs((block["length"] * block["width"]) - (count[:, 0] * count[:, 1])))
            # try:
            #     fitting_slot = np.argmin(np.abs((block["length"] * block["width"]) - (count[:, 0] * count[:, 1])))
            # except:
            #     print(self.name)
            #     print(self.area_slots)
            #     print(count)
            #     print(block)
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
        
        # display(self.area_slots)
    
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

class Args:
    def __init__(self) -> None:
        pass

class DQNAgent(nn.Module):
    def __init__(self, n_yard_out=10, n_max_block=30, hidden_dim = 128, device = device) -> None:
        super(DQNAgent, self).__init__()
        self.n_yard = n_yard_out
        self.n_max_block = n_max_block
        self.to_encoder = nn.LSTM(input_size=25, hidden_size=hidden_dim)
        self.to_decoder = nn.LSTMCell(input_size=25, hidden_size=hidden_dim)
        self.to_fc_block = nn.Linear(hidden_dim*2, n_max_block*n_yard_out) # 400 = 40(n_block) * 10(n_yard)
        self.ti_encoder = nn.LSTMCell(input_size=25, hidden_size=hidden_dim)
        self.ti_fc_block = nn.Linear(512, n_max_block)
        self.device = device
        
        self.soft_max = nn.Softmax(dim=-1)
        
    def generate_mask(self, take_out: pd.DataFrame, yard_slots: np.ndarray):
        encoder_mask = np.zeros((take_out.shape[0], self.n_yard), dtype=bool)
        
        for b_idx, block_info in enumerate(take_out[["length", "width", "height"]].values):
            mask = np.all(block_info <= yard_slots[:, :3], axis=-1) # 블록 크기가 슬롯보다 작으면 True
            for y_idx in range(self.n_yard):
                mask_part = mask[y_idx*4: (y_idx+1)*4]
                encoder_mask[b_idx, y_idx] = np.any(mask_part) # 블록이 슬롯보다 작은게 하나라도 있다면 True
        
        return torch.BoolTensor(~encoder_mask).to(self.device)
        
    
    def forward(self, infos: List[pd.DataFrame], encoder_inputs : torch.Tensor, remaining_area: np.array):
        """
        take_out, take_in, feat_vec: ["length", "width", "height", "weight", "location"]
        slot_info: [length  width  height  location  count]
        """
        # TODO: Memory에 unsqueeze 돼서 들어가도록 수정
        take_in, take_out, yard_slots = infos
        
        # print(take_out)
        
        n_take_out = len(take_out)
        # print(encoder_inputs)
        
        hids, (h0, c0) = self.to_encoder(encoder_inputs)
        
        input0 = torch.zeros(1, 25).to(self.device)
        # print(h0.device, c0.device, input0.device)
        block_mask = torch.zeros((n_take_out, 1), dtype=bool) # 제외 시키고 싶은 경우를 True
        
        probs_history = []
        rewards_history = []
        action_history = []
        
        for idx in range(n_take_out):
            encoder_mask = self.generate_mask(take_out, yard_slots)
            h0, c0 = self.to_decoder(input0, (h0, c0))
            att_score = self.soft_max(torch.mm(h0, hids.transpose(1, 0)))
            att_vals = att_score.transpose(1, 0) * hids
            encoder_context = torch.sum(att_vals, dim=0)
            out = torch.cat([h0, encoder_context.unsqueeze(0)], dim=1)
            out = self.to_fc_block(out).squeeze(0)
            out[self.n_yard*n_take_out:] = -10000 # 블록 개수보다 많을 필요는 없으니 마스킹
            # print(out[:10*n_take_out].shape)
            out[:self.n_yard*n_take_out][block_mask.repeat(1, self.n_yard).reshape(-1)] = -10000 # 선택된 블록이 다시 선택되지 않도록 마스킹
            out[:self.n_yard*n_take_out][encoder_mask.reshape(-1)] = -10000 # 보내고자 하는 블록이 적치장에 안 맞으면 마스킹
            
            probs = self.soft_max(out)
            # print(probs)
            m = Categorical(probs)
            action = m.sample()
            
            # for masking next step
            block, yard = torch.div(action, self.n_yard, rounding_mode='trunc'), action % self.n_yard
            block, yard = block.item(), yard.item()
            
            # try:
            #     block_size = take_out.iloc[block][["length", "width", "height", "weight", "location"]].values
            # except:
            #     print(probs)
            #     print(n_take_out)
            #     print(action, block, yard)
            block_size = take_out.iloc[block][["length", "width", "height", "weight", "location"]].values

                
            # print(block_size.dtypes)
            # print(torch.from_numpy(block_size))
            slot_info = deepcopy(yard_slots[yard*4:(yard+1)*4])
            input0 = torch.cat([torch.FloatTensor(block_size.astype(np.float32)), torch.FloatTensor(slot_info).reshape(-1)], dim=0).unsqueeze(0).to(self.device)
            
            # print("\n\n")
            
            # print(yard)
            # print(yard_slots)
            slot_info[slot_info[:, 0] == 0] = 10000
            # print(np.abs((block_size[0] * block_size[1]) - (slot_info[:, 0] * slot_info[:, 1])))
            yard_offset = np.argmin(np.abs((block_size[0] * block_size[1]) - (slot_info[:, 0] * slot_info[:, 1])))
            yard_idx = yard*4 + yard_offset
            # print(yard_idx)
            yard_slots[yard_idx, -1] -= 1
            yard_slots[yard_slots[:, -1] == 0] = 0
            slot_size = slot_info[yard_offset]
            reward_part = (slot_size[0] * slot_size[1] - block_size[0] * block_size[1]) ** 2
            remaining_area[yard] += reward_part
             
            
            probs_history.append(m.log_prob(action))
            action_history.append((block, yard))
            rewards_history.append(np.sum(remaining_area))
        
        
        action_history = np.array(action_history)
        # print(action_history)
        
        return probs_history, rewards_history, action_history
        

class RLEnv():
    def __init__(self) -> None:
        self.yard_in = YardInside(name="사내", area_slots=None, blocks=None)
        # Set by random for now
        self.whole_block : pd.DataFrame = sample_dataset(600)
        self.whole_block["location"] = None
        self.target_columns = ["length", "width", "height", "weight", "location"]
        self._init_env()
        self.batch_limit = (65, 20)
        self.DQNet = DQNAgent(n_yard_out=len(self.yards_out), n_max_block=30)
        self.DQNet.to(device)
        self.optimizer = torch.optim.Adam(self.DQNet.parameters(), lr=5e-3)
        self.probs = []
        self.rewards = []
        self.first_step_infos = [deepcopy(self.whole_block)]
        
        
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
                # print(slot_info["name"])
                # print(rand_blocks)
                self.whole_block.loc[rand_blocks.index, "location"] = slot_info["name"]
                blocks = rand_blocks.reset_index(drop=True)
            
                df_temp = np.repeat([[slot_info["length_max"], slot_info["width_min"], slot_info["width_max(<)"], slot_info["height_max"]]], slot_info["count"], axis=0)
                df_temp = pd.DataFrame(df_temp, columns=["length", "width_min", "width_max", "height"])
                df_temp.loc[df_temp.index[:len(blocks)], "vessel_id"] = blocks["vessel_id"].values
                df_temp.loc[df_temp.index[:len(blocks)], "block"] = blocks["block"].values
                # display(df_temp)
                area_slots_.append(df_temp)
                blocks_.append(blocks)
        
            # print(area_slots_)
            area_slots = pd.concat(area_slots_).reset_index(drop=True)
            blocks = pd.concat(blocks_).reset_index(drop=True)
            # print(area_slots)
            # print(blocks)
        
            return blocks, area_slots
        
        # ! 사외 적치장 정보 추가
        self.yards_out_slot_sizes = pd.read_csv(os.path.join(src_dir, "Yard_capacity.csv"))
        # print(self.yards_out_slot_sizes)
        self.yards_out_slot_sizes.fillna(float("inf"), inplace=True)
        
        self.yards_out : Dict[str, "YardOutside"]= {}
        for name in self.yards_out_slot_sizes["name"].unique():
            # print(name)
            info = self.yards_out_slot_sizes.loc[self.yards_out_slot_sizes["name"] == name]
            blocks, area_slots = _init_yard_out(info)
            area_slots["location"] = name
            self.yards_out[name] = YardOutside(name=name, area_slots= area_slots, blocks=blocks.reset_index(drop=True))
            
        self.whole_block.loc[self.whole_block["location"].isnull(), "location"] = "사내"
        # print(self.whole_block["location"].value_counts())
        self.labels_encoder = dict(zip(self.yards_out_slot_sizes["name"].unique(), range(1, len(self.yards_out_slot_sizes["name"].unique())+1)))
        self.labels_encoder["사내"] = 0
        
    def get_state(self, possible_take_out: pd.DataFrame, take_in: pd.DataFrame):
        
        possible_take_out["location"] = possible_take_out["location"].map(self.labels_encoder)
        encoder_inputs =[]
        yard_slots = []
        remaining_areas = []
        
        for name, yard in self.yards_out.items():
            # remaining_areas.append(yard.calc_remaining_area())
            remaining_areas.append(yard.calc_remaining_area_by_slot())
            state_part_ = yard.area_slots.copy()
            # print(state_part_)
            state_part_["location"] = self.labels_encoder[name]
            state_part_.loc[state_part_["height"] == float("inf"), "height"] = 1000
            count = state_part_.loc[state_part_["vessel_id"].isnull(), ["length", "width", "height", "location"]].groupby(["length", "width"], as_index=False).value_counts()
            yard_t = torch.FloatTensor(count.values)
            count_ = np.zeros((4, 5))
            count_[:count.shape[0], :] = count
            yard_slots.append(count_)
            for _, block in possible_take_out.iterrows():
                block_t = torch.FloatTensor([block[["length", "width", "height", "location","weight"]].values])
                pad_t = torch.zeros((4-len(count),5))
                input = torch.concat([block_t, yard_t, pad_t], dim=0)
                encoder_inputs.append(input)
                
                
        encoder_inputs = torch.stack(encoder_inputs)
        encoder_inputs = encoder_inputs.reshape(encoder_inputs.shape[0], -1)
        yard_slots = np.concatenate(yard_slots)
        
        return encoder_inputs, yard_slots, remaining_areas
    
    def calculate_batch_nums(self, blocks : pd.DataFrame):
        """
        1안) 넓이로 무식하게
        2안) 왼쪽에 쭉 정렬, 오른쪽에 쭉 정렬하는 방식으로 채우자
        3안) 모든 경우의 수 다 해보고 가능한 케이스
        """
        
        length_ub, width_ub = self.batch_limit
        
        remaining_space = length_ub * width_ub
        
        count = 1
        for _, block in blocks.iterrows():
            if remaining_space - block["length"] * block["width"] >= 0:
                remaining_space -= block["length"] * block["width"]
            else:
                count += 1
                remaining_space = length_ub * width_ub - block["length"] * block["width"]
        return count
            
    def step(self, infos: List[pd.DataFrame], inputs: torch.Tensor, remaining_areas: np.array):
        """
        infos: infos about block to be taken out, infos about block to be taken in, infos about slots in the yard 
        inputs: combination of block infos and yard slots
        """
        
        take_in, take_out, yard_slots = infos
        # print(take_out)
        take_in["location"] = take_in["location"].map(self.labels_encoder).astype(float)
        take_out["location"] = take_out["location"].map(self.labels_encoder).astype(float)
        probs, rewards, actions  = self.DQNet((take_in.reset_index(drop=True), take_out.reset_index(drop=True), yard_slots), inputs, remaining_areas)
        self.probs, self.rewards = probs, rewards
        actions[:, 1] += 1 # 인덱스라서 하나 더 추가해줘야 함
        
        batch_num = 0
        for name, yard in self.yards_out.items():
            
            # if name == "덕곡1":
            #     print("Before moving: \n")
            #     print(yard.area_slots)
            block_idxs = actions[actions[:, 1] == self.labels_encoder[name]][:, 0]
            if len(block_idxs) > 0:
                # print(name, block_idxs)
                blocks_out = take_out.iloc[block_idxs, :]
                # print(blocks_out.dtypes)
                yard.move_to_yard_out(blocks_out)
                self.whole_block.loc[self.whole_block.index[blocks_out.index], "location"] = name
                batch_num += self.calculate_batch_nums(blocks_out)
                
            # if name == "덕곡1":
            #     print("After moving: \n")
            #     print(yard.area_slots)
            
            take_in_part = take_in.loc[take_in["location"] == self.labels_encoder[name]]
            if len(take_in_part) != 0:
                yard.move_to_yard_in(take_in_part)
                self.whole_block.loc[self.whole_block.index[take_in_part.index], "location"] = "사내"
                batch_num += self.calculate_batch_nums(take_in_part)
            # print(yard.blocks)
            
        
        """
        Objective: maximize a*여유공간 - b*배치 수
        """
        return np.sum(remaining_areas)
    
    def reset(self):
        self.whole_block = deepcopy(self.first_step_infos[0])
        for yard in self.yards_out.values():
            yard._reset()
    
        
    def update_policy(self):
        R = 0
        policy_loss = torch.zeros((len(self.probs)))
        returns = []
        for r in self.rewards[::-1]:
            R = np.log1p(r) + 0.99 * R
            returns.insert(0, R)
        # print()
        returns = torch.tensor(returns)
        # print(len(self.probs), returns.shape)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        for idx, log_prob, R in zip(range(len(self.probs)), self.probs, returns):
            # print(type(-log_prob * R))
            policy_loss[idx] = -log_prob * R
        self.optimizer.zero_grad()
        # policy_loss = torch.sum(torch.mul(self.probs, (-1)*returns))
        policy_loss = torch.sum(policy_loss)
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.DQNet.parameters(), 20)
        self.optimizer.step()
        
        self.saved_log_probs = []
        self.rewards = []
        
        return policy_loss.item()
    
    def save_env(self, save_dir, exp_num, infos):
        whole_block = deepcopy(self.first_step_infos[0])
        whole_block.to_csv(os.path.join(save_dir, f"whole_block_{exp_num}.csv"), index=False)
        
        take_in, take_out, yard_slots = infos
        take_in.to_csv(os.path.join(save_dir, f"take_in_{exp_num}.csv"), index=False)
        take_out.to_csv(os.path.join(save_dir, f"take_out_{exp_num}.csv"), index=False)
        pd.DataFrame(yard_slots).to_csv(os.path.join(save_dir, f"yard_slots_{exp_num}.csv"), index=False)
    
    
def decide_blocks(env):
    n_take_out = np.random.randint(10, 25)
    n_take_in = np.random.randint(10, 25)
    
    blocks_to_be_out = env.whole_block.loc[(env.whole_block["location"] == "사내") & (env.whole_block["length"] < 25)].sample(n_take_out)
    # blocks_to_be_out_small = env.whole_block.loc[(env.whole_block["location"] == "사내") & (env.whole_block["length"] <= 15)].sample(n_take_out-5)
    # blocks_to_be_out_big = env.whole_block.loc[(env.whole_block["location"] == "사내") & (env.whole_block["length"] > 15) & (env.whole_block["length"] <= 24) & (env.whole_block["width"] <= 16)].sample(5)
    # blocks_to_be_out = pd.concat([blocks_to_be_out_small, blocks_to_be_out_big], axis=0)
    
    blocks_to_be_in = env.whole_block.loc[(env.whole_block["location"] != "사내") & (env.whole_block["length"] < 25)].sample(n_take_out)
    # blocks_to_be_in_small = env.whole_block.loc[(env.whole_block["location"] == "사내") & (env.whole_block["length"] <= 15)].sample(n_take_in-5)
    # blocks_to_be_in_big = env.whole_block.loc[(env.whole_block["location"] == "사내") & (env.whole_block["length"] > 15) & (env.whole_block["length"] <= 24) & (env.whole_block["width"] <= 16)].sample(5)
    # blocks_to_be_in = pd.concat([blocks_to_be_in_small, blocks_to_be_in_big], axis=0)
            
        
    encoder_inputs, yard_slots, remaining_areas = env.get_state(blocks_to_be_out.copy(), blocks_to_be_in.copy())
    take_in, take_out = blocks_to_be_in.copy(), blocks_to_be_out.copy()
    return (n_take_out, n_take_in), (take_in, take_out, yard_slots), encoder_inputs, remaining_areas
    
    
def run():
    
    """
    여기서 DQN의 역할은 블록과 사외적치장을 이어주는 거고
    나머지 어떤 블록을 보내고 어떤 블록을 가져올지는 하나의 선택지로 남아있게 되는 건데
    """
    current_dateTime = datetime.now()
    exp_dir = os.path.join(result_dir, f"{current_dateTime.year}_{current_dateTime.month}_{current_dateTime.day}_{current_dateTime.hour}_{current_dateTime.minute}_exp")
    # exp_dir = os.path.join(result_dir, f"test")
    os.makedirs(exp_dir, exist_ok=True)
    
    with open(os.path.join(exp_dir, "info.txt"), "w") as f:
        f.write("슬롯보다 크기가 큰 블록들이 존재해서 그런 블록들이 선택되지 않도록 제외")
        
    for exp_num in range(100):
        env = RLEnv()
        (n_take_out, n_take_in), (take_in, take_out, yard_slots), encoder_inputs, remaining_areas = decide_blocks(env)
        env.save_env(exp_dir, exp_num, (take_in, take_out, yard_slots))
        
        obj_loss_history = []
        loop = tqdm(range(500))
        for epo in loop:
            env.reset()
            for day in range(1):
                # print(f"\nDay: {day} / {n_take_out}, {n_take_in} ")
                objective = env.step((take_in.copy(), take_out.copy(), deepcopy(yard_slots)), deepcopy(encoder_inputs.to(device)), deepcopy(remaining_areas))
            loss = env.update_policy()
            loop.set_description(f"{loss:.3f} / {objective}")
            obj_loss_history.append(objective)
        plt.plot(obj_loss_history)
        plt.savefig(os.path.join(exp_dir, "loss_{}.png".format(exp_num)))
        plt.close()
        
    
    
    pass
    
    
if __name__ == "__main__":
    run()
    pass

#%%


# temp = np.array([True, False, False, True, False])
# print(temp.shape)
# print(np.expand_dims(temp, axis=1).shape)
# temp = np.expand_dims(temp, axis=1).repeat(5)
# # temp = temp.repeat(1, 5)
# print(temp)

# temp = torch.BoolTensor(np.array([True, False, False, True, False])).unsqueeze(-1)
# print(temp.shape)
# # temp = temp.repeat(1, 5)
# temp = temp.repeat(5)
# # temp = temp.repeat(5, 1)
# # temp = temp.repeat(1, 5)
# print(temp)
# print(temp.shape)

# import torch
# import numpy as np

# n_take_out = 15
# block_mask = torch.ones((1, n_take_out, 1), dtype=bool) # 제외 시키고 싶은 경우를 True
# block_mask[1:6:2] = False
# # print(block_mask.repeat(1, 10))
# print(block_mask.repeat(10, 1, 1).shape)
# print(block_mask.repeat(1, 10).reshape(-1))
# block_mask.repeat(1, 10).reshape(-1)


#%%