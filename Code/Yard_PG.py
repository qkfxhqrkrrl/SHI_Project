#%%

import numpy as np
np.set_printoptions(precision=4)
import pandas as pd

import torch
import torch.nn as nn
from torch.distributions import Categorical

import os
from copy import deepcopy
from random import sample
from collections import namedtuple

import matplotlib.pyplot as plt

from typing import List, Dict, Optional, Tuple

src_dir = os.path.abspath(os.path.join(__file__, * [os.pardir] * 2, "Src"))

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
    
    # def update(self):
    #     if self.blocks is None:
    #         raise Exception("Yard does not have blocks")
    #     # 블록 이동 후 block 과 관련된 정보들 업데이트 하는 함수
    #     # TODO: 블록 사이즈와 다른 적치 공간이 남으면 다른 적치 공간과의 조합도 고려해서 개수를 셀 수 있도록
    #     self.available_blocks_count = deepcopy(self.block_count_limit)
    #     # * 값을 확인 할 때, 아예 case_num를 부여하는 것도 좋은 방법일 것 같음
    #     for width_gap in [(i, i+5) for i in range(5, 26, 5)]:
    #         for length_gap in [(i, i+5) for i in range(5, 26, 5)]:
    #             # print((width_gap[1], length_gap[1]))
    #             # display(self.blocks.loc[(self.blocks["width"] >= width_gap[0]) & (self.blocks["width"] < width_gap[1]) & (self.blocks["length"] >= length_gap[0]) & (self.blocks["length"] < length_gap[1]), :].copy())
    #             self.blocks_by_size[(width_gap[1], length_gap[1])] = self.blocks.loc[(self.blocks["width"] >= width_gap[0]) & (self.blocks["width"] < width_gap[1]) \
    #                             & (self.blocks["length"] >= length_gap[0]) & (self.blocks["length"] < length_gap[1]), :].copy()
    #             self.blocks_count_by_size[width_gap[1]][length_gap[1]] = len(self.blocks_by_size[(width_gap[1], length_gap[1])])
    #             self.available_blocks_count[(max(width_gap[1], length_gap[1]), min(width_gap[1], length_gap[1]))] -= len(self.blocks_by_size[(width_gap[1], length_gap[1])])
        
        
class YardInside(Yard):
    def __init__(self, name: str, area_slots : Dict, blocks : pd.DataFrame = None):
        super().__init__(name, area_slots, blocks)
        
    
    def move_block_to(self, blocks_to_move: pd.DataFrame, yard: Yard):
        yard.blocks = pd.concat([yard.blocks, blocks_to_move], axis=0)
        self.blocks = self.blocks.loc[~self.blocks["id"].isin(blocks_to_move["id"]), :]
        # self.update()
        # yard.update()
        
class YardOutside(Yard):
    def __init__(self, name: str, area_slots : pd.DataFrame, blocks : pd.DataFrame = None):
        super().__init__(name, area_slots, blocks)
        self.orig_slots = area_slots.copy()
        # print(self.orig_slots.columns)
        # print(self.orig_slots)
        self.orig_slots["id"] = range(len(area_slots))
        self.area_slots = area_slots[["vessel_id", "block", "length", "width_max","height"]].rename(columns={"width_max": "width"})
        self.area_slots["id"] = range(len(area_slots))
        self.columns = self.area_slots.columns
        self.used_area = np.sum(self.area_slots["length"] * self.area_slots["width"])
        self._update_slot()
        

    def move_to_yard_out(self, slots, blocks: pd.DataFrame):
        self.blocks = pd.concat([self.blocks, blocks]).reset_index(drop=True)
        self.blocks["location"] = self.name
        
        # if self.name == "성동":
        #     print("\n\nBefore moving out: ")
        #     display(self.blocks)
        #     display(self.area_slots)
        # print(slots)
        # display(self.area_slots.loc[self.area_slots["block"].isnull(), :])
        for slot, (idx, block) in zip(slots, blocks.iterrows()):
            length, width = slot[0], slot[1]
            try:
                slot_idx = self.area_slots.loc[(self.area_slots["length"]==length) & (self.area_slots["width"]==width) & (self.area_slots["block"].isnull())].index[0]
            except:
                print(f"Error in {self.name}, {idx}")
                print(slots)
                print(self.area_slots.loc[self.area_slots["block"].isnull(), :])           
            # slot_idx = self.area_slots.loc[(self.area_slots["length"]==length) & (self.area_slots["width"]==width) & (self.area_slots["block"].isnull())].index[0]
            self.area_slots.loc[self.area_slots.index[slot_idx], "block"] = block["block"]
            self.area_slots.loc[self.area_slots.index[slot_idx], "vessel_id"] = block["vessel_id"]
            
        # if self.name == "성동":
        #     print("\n\nAfter moving out: ")
        #     display(self.blocks)
        #     display(self.area_slots)
        
        del self.area_slots["weight"]
        
    def move_to_yard_in(self, blocks: pd.DataFrame):
        # print("Before: ",self.blocks.shape[0])
        # if self.name == "성동":
        #     print("\n\nBefore moving in: ")
        #     display(self.blocks)
        #     display(self.area_slots)
        for _, block in blocks.iterrows():
            vessel_id, block_id = block["vessel_id"] ,block["block"]
            self.blocks = self.blocks.loc[~((self.blocks["vessel_id"] == vessel_id) & (self.blocks["block"] == block_id)), :]
            self.area_slots.loc[~((self.area_slots["vessel_id"] == vessel_id) & (self.area_slots["block"] == block_id)), ["vessel_id", "block"]] = np.nan
        
        # if self.name == "성동":
        #     print("\n\nAfter moving in: ")
        #     display(self.blocks)
        #     display(self.area_slots)
        
        # print("After: ",self.blocks.shape[0])
        
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
        return (possible_area - taken_area) / possible_area

class Args:
    def __init__(self) -> None:
        pass

class DQNAgent(nn.Module):
    def __init__(self, n_yard_out) -> None:
        super(DQNAgent, self).__init__()
        self.feat_ext = nn.Sequential(
            nn.Linear(1436, 512),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.feat_ext2 = nn.Linear(5, 1)
        
        self.n_yard_out = n_yard_out
        # self.rnn = nn.LSTM(input_size=5, hidden_size=256, batch_first=True)
        self.rnn = nn.LSTMCell(input_size=5, hidden_size=256)
        self.rnn_ti = nn.LSTMCell(input_size=5, hidden_size=256)
        self.fc_block = nn.Linear(256, 100)
        self.fc_slot = nn.Linear(256, 300)
        self.soft_max = nn.Softmax(dim=-1)
        
        self.prob_history = []
        
    def check_slot_count(self, block_info, slot_info):
        # TODO : If count == 0 , then delete row
        # print(slot_info.shape)
        # slot_info: [length  width  height  location  count]
        # output: [batch_size  n_block  n_yard]
        # 해당 블록이 들어갈 수 있는 슬롯이 없으면 False로 걸러주기
        # print(slot_info)
        
        batch_size = slot_info.shape[0]
        mask = torch.zeros((block_info.shape[0], 100, self.n_yard_out), dtype=torch.bool)
        
        for i in range(100): # 패딩 길이를 100으로 해줬으니까
            block_size = block_info[:, i, :2].unsqueeze(1) # (batch_size, 1, n_feat)
            mask_ = torch.all(block_size < slot_info[:, :, :2], dim=-1)
            for yard_idx in range(self.n_yard_out):
                mask_yard = mask_[:, yard_idx*50 : (yard_idx+1) * 50] # (batch_size, pad_len, n_feat) # 패딩을 안 하면 마스킹 할 때 차원이 뭉개지면서 제대로 된 값 추출 불가
                # 현재 고려하는 블록이 슬롯에 들어갈 수 있는지를 고려하고 False인 경우 softmax에서 제해줄 예정
                # 합이 0보다 크단 것은 들어갈 수 있는 슬롯이 있다란 얘기고, 
                # 모든 합이 pad_len 만큼이라는 것은 어떠한 슬롯에도 들어갈 수 있는 경우인데 이때는 블록이 존재하지 않는 경우인 -1이란 얘기다
                mask[:, i, yard_idx] = (torch.sum(mask_yard, dim=-1) > 0) & ~(torch.sum(mask_yard, dim=-1) == 50)
                # torch.any(block_size.unsqueeze(1)[:, :, :2] < yard_slot[:, :, :2], dim=-1, out=mask[:, i, yard_idx]
        
        return mask.reshape(batch_size, -1)
    
    def forward(self, to_tensor, ti_tensor, ys_tensor, slot_info : np.array = None):
        """
        take_out, take_in, feat_vec: ["length", "width", "height", "weight", "location"]
        slot_info: [length  width  height  location  count]
        """
        # TODO: Memory에 unsqueeze 돼서 들어가도록 수정
        # ti_tensor, to_tensor, ys_tensor, slot_info = ti_tensor.unsqueeze(0), to_tensor.unsqueeze(0), ys_tensor.unsqueeze(0), np.expand_dims(slot_info, axis=0)
        # ti_tensor, to_tensor, ys_tensor, slot_info = \
        #     torch.concat([ti_tensor, ti_tensor[:, torch.randperm(ti_tensor.shape[1])]], dim=0), torch.concat([to_tensor, to_tensor[:, torch.randperm(to_tensor.shape[1])]], dim=0), \
        #     torch.concat([ys_tensor]*2, dim=0), np.concatenate([slot_info]*2, axis=0)
        
        state = torch.concat([to_tensor, ti_tensor, ys_tensor], dim=1)
        ti_np, to_np, ys_np = ti_tensor.cpu().detach().numpy(), to_tensor.cpu().detach().numpy(), ys_tensor.cpu().detach().numpy()
        
        # ys_tensor = self.feat_ext(torch.concat([to_tensor, ti_tensor, ys_tensor], dim=-2).transpose(-1, -2)).reshape(batch_size, -1)
        feat_vec = self.feat_ext2(self.feat_ext(state.transpose(-1, -2)).transpose(-1, -2)).squeeze(-1)
        if slot_info is None:
            return feat_vec

        batch_size = slot_info.shape[0]
        h_to, c_to = feat_vec, torch.zeros(batch_size, 256)
        h_ti, c_ti = feat_vec, torch.zeros(batch_size, 256)
        input_to = torch.zeros((batch_size, 5))
        input_ti = torch.zeros((batch_size, 5))
        
        block_mask_to = np.ones((to_np.shape[0], to_np.shape[1]), dtype=bool)
        block_mask_to[to_np[:, :, 0] == -1] = False
        block_mask_ti = np.ones((ti_np.shape[0], ti_np.shape[1]), dtype=bool)
        block_mask_ti[ti_np[:, :, 0] == -1] = False
        actions = []
        
        # hasExceed = []
        for i in range(100): # 패딩 길이를 100으로 해줬으니까
            # mask = self.masking_by_slots(ti_tensor, slot_info)
            
            h_to, c_to = self.rnn(input_to.squeeze(1), (h_to, c_to))
            
            prob_dist_b = self.fc_block(h_to)
            prob_dist_b[~torch.BoolTensor(block_mask_to)] = -1000
            prob_dist_b = self.soft_max(prob_dist_b)
            block_selection = torch.max(prob_dist_b, dim=-1).indices
            # m_block = Categorical(prob_dist_b)
            # block_selection = m_block.sample()
            # print("Selected Block: ", block_selection) 
            # 선택된 블록들이 다시 인풋으로 들어갈 수 있게끔 값 저장
            input_to = torch.gather(to_tensor, 1, block_selection.unsqueeze(-1).repeat(1, 1, 5).permute(1, 0, 2))
            block_selection = block_selection.detach().cpu().numpy()
            block_mask_to[range(len(block_selection)), block_selection] = False
            # print(block_mask_to)
            
            # 선택된 블록들이 slot을 정하는 제약 조건이 될 수 있도록 마스킹
            prob_dist_s = self.fc_slot(h_to)
            slot_mask = np.any(input_to.detach().cpu().numpy()[:, :, :2] >= slot_info[:, :, :2], axis=-1) # 슬롯의 크기가 블록보다 작으면 제외
            slot_mask[slot_info[:, :, 0] == 0] = True # 패딩으로 인해 비어있는 곳 제외
            # if np.sum(slot_mask[0]) == 300:
            #     hasExceed.append(i)
            prob_dist_s[torch.BoolTensor(slot_mask)] = -1000
            slot_selection = torch.max(prob_dist_s, dim=-1).indices.detach().cpu().numpy() 
            slot_info[range(len(slot_selection)), slot_selection, 4] -= 1 # 선택됐으니 슬롯 크기 감소
            slot_info[slot_info[:, :, 4] == 0] = 0 # 선택으로 인해 슬롯이 없어진 경우엔 다른 값들도 모두 0으로 변경해서 다음 마스킹에도 제외
            
            h_ti, c_ti = self.rnn_ti(input_ti.squeeze(1), (h_ti, c_ti))
            prob_dist_b_ti = self.fc_block(h_ti)
            prob_dist_b_ti[~torch.BoolTensor(block_mask_ti)] = -1000
            prob_dist_b_ti = self.soft_max(prob_dist_b_ti)
            block_selection_ti = torch.max(prob_dist_b_ti, dim=-1).indices.detach().cpu().numpy()
            block_mask_ti[range(len(block_selection_ti)), block_selection_ti] = False
            
            
            actions.append(np.vstack([block_selection, slot_selection, block_selection_ti]).T)
        
        actions = np.array(actions).transpose([1, 0, 2])
        
        return state, actions
        

class RLEnv():
    def __init__(self) -> None:
        self.yard_in = YardInside(name="사내", area_slots=None, blocks=None)
        # Set by random for now
        self.whole_block : pd.DataFrame = sample_dataset(600)
        self.whole_block["location"] = None
        self.target_columns = ["length", "width", "height", "weight", "location"]
        self._init_env()
        self.batch_limit = (65, 20)
        self.probs = []
        self.rewards = []
        
        
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
            self.yards_out[name] = YardOutside(name=name, area_slots= area_slots, blocks=blocks.reset_index(drop=True))
            
        self.whole_block.loc[self.whole_block["location"].isnull(), "location"] = "사내"
        # print(self.whole_block["location"].value_counts())
        self.labels_encoder = dict(zip(self.yards_out_slot_sizes["name"].unique(), range(1, len(self.yards_out_slot_sizes["name"].unique())+1)))
        self.labels_encoder["사내"] = 0
        
    
    def get_state(self, possible_take_out: pd.DataFrame, take_in: pd.DataFrame):
        
        to_pad = (-1) * np.ones((100, len(self.target_columns)))
        # print(possible_take_out)
        possible_take_out["location"] = possible_take_out["location"].map(self.labels_encoder)
        to_pad[:possible_take_out.shape[0], :] = possible_take_out[self.target_columns].values
        
        ti_pad = (-1) * np.ones((100, len(self.target_columns)))
        take_in["location"] = take_in["location"].map(self.labels_encoder)
        ti_pad[:take_in.shape[0], :] = take_in[self.target_columns].values
        
        mask = np.zeros((100, 2), dtype=bool)
        mask[:possible_take_out.shape[0], 0] = True
        mask[:take_in.shape[0], 1] = True
        
        state_ = []
        yard_out_slots = []
        
        for name, yard in self.yards_out.items():
            state_part_ = yard.area_slots.copy()
            # display(state_part_)
            state_part_["location"] = self.labels_encoder[name]
            state_part_.loc[state_part_["height"] == float("inf"), "height"] = 1000
            state_part_.loc[state_part_["weight"].isnull(), "weight"] = 1000
            count_ = state_part_.loc[state_part_["vessel_id"].isnull(), ["length", "width", "height", "location"]].groupby(["length", "width"], as_index=False).value_counts()
            # display(count_)
            
            count = np.zeros((30, 5))
            count[:count_.shape[0], :] = count_.values
            # print(count)
            yard_out_slots.append(count)
            
            state_part_ = state_part_[self.target_columns].values
            
            pad_len = yard.orig_slots.shape[0] * 3
            state_part = (-1) * np.ones((pad_len, state_part_.shape[1]))
            state_part[:state_part_.shape[0]] = state_part_
            state_.append(state_part)
            
        yard_state = np.concatenate(state_, axis=0)
        yard_out_slots = np.concatenate(yard_out_slots, axis=0)
        # print(yard_out_slots)
        
        return to_pad, ti_pad, yard_state, yard_out_slots, mask
    
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
            
    def step(self, take_in, take_out, yard_out_slots, actions, mask):
        """
        feat: ["width", "length", "weight", "height", "location"]
        yard_out_slots: [length  width  height  location  count]
        """
        
        # Strain it according to actions
        take_out_yard_pair = actions[:, mask[:, 0], :][:, :, [0,1]]
        selected_slots = yard_out_slots[take_out_yard_pair[:, :, 1].reshape(-1)]
        selected_blocks_out = take_out.iloc[take_out_yard_pair[:, :, 0].reshape(-1)] # 굳이 reset index를 안 하는 이유는 아래 index로 self.whole_block에 접근하기 때문
        
        take_in_order = actions[:, mask[:, 1], :][:, :, [2]].reshape(-1)
        take_in_infos = take_in.iloc[take_in_order]
        
        # separate it by the yards
        batch_num = 0
        spaces = []
        
        for name, yard in self.yards_out.items():
            slots_ = selected_slots[selected_slots[:, 3] == self.labels_encoder[name]]
            blocks_out = selected_blocks_out.loc[selected_slots[:, 3] == self.labels_encoder[name]]
            blocks_in = take_in_infos.loc[take_in_infos["location"] == self.labels_encoder[name]]
            if len(blocks_out) != 0:
                yard.move_to_yard_out(slots_, blocks_out)
                self.whole_block.loc[self.whole_block.index[blocks_out.index], "location"] = name # Env 블록 적재 현황도 변경
                batch_num += self.calculate_batch_nums(blocks_out)
            if len(blocks_in) != 0:
                yard.move_to_yard_in(blocks_in)
                self.whole_block.loc[self.whole_block.index[blocks_in.index], "location"] = "사내"
                batch_num += self.calculate_batch_nums(blocks_in)
            yard._update_slot()
            remaining_space = yard.calc_remaining_area()
            spaces.append(remaining_space)
        
        """
        Objective: maximize a*여유공간 - b*배치 수
        """
        return batch_num, spaces
    
def decide_blocks(env):
    n_take_out = np.random.randint(10, 15)
    n_take_in = np.random.randint(10, 15)
    
    blocks_to_be_out_small = env.whole_block.loc[(env.whole_block["location"] == "사내") & (env.whole_block["length"] <= 15)].sample(n_take_out-5)
    blocks_to_be_out_big = env.whole_block.loc[(env.whole_block["location"] == "사내") & (env.whole_block["length"] > 15) & (env.whole_block["length"] <= 24) & (env.whole_block["width"] <= 16)].sample(5)
    blocks_to_be_out = pd.concat([blocks_to_be_out_small, blocks_to_be_out_big], axis=0)
    
    blocks_to_be_in_small = env.whole_block.loc[(env.whole_block["location"] == "사내") & (env.whole_block["length"] <= 15)].sample(n_take_in-5)
    blocks_to_be_in_big = env.whole_block.loc[(env.whole_block["location"] == "사내") & (env.whole_block["length"] > 15) & (env.whole_block["length"] <= 24) & (env.whole_block["width"] <= 16)].sample(5)
    blocks_to_be_in = pd.concat([blocks_to_be_in_small, blocks_to_be_in_big], axis=0)
    
    to_pad, ti_pad, yard_state, yard_out_slots, mask = env.get_state(blocks_to_be_out.copy(), blocks_to_be_in.copy())
    take_in, take_out = blocks_to_be_in.copy(), blocks_to_be_out.copy()
    yard_out_slots_ = deepcopy(np.expand_dims(yard_out_slots, axis=0))
    
    return (n_take_out, n_take_in) , (take_in, take_out), (to_pad, ti_pad,  yard_state), (yard_out_slots, yard_out_slots_), mask
    
    
class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
        
    def push(self, experience):
        if len(self.memory)<self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count%self.capacity] = experience
        self.push_count+=1
        
    def sample(self, batch_size):
        return sample(self.memory, batch_size)
    
    def can_provide_sample(self, batch_size):
        return len(self.memory)>=batch_size
    
def run():
    env = RLEnv()
    memories = ReplayMemory(10000)
    BATCH_SIZE = 1
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10
    
    """
    여기서 DQN의 역할은 블록과 사외적치장을 이어주는 거고
    나머지 어떤 블록을 보내고 어떤 블록을 가져올지는 하나의 선택지로 남아있게 되는 건데
    """
    memory_template = namedtuple('Experience', ('state', 'next_state', 'reward', 'action'))
    
    # eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    eps_threshold = 0.0
    # eps_threshold = 1.0
    policy_net = DQNAgent(n_yard_out=10)
    target_net = DQNAgent(n_yard_out=10)
    
    # * ______________________ Start_state _______________________
    (n_take_out, n_take_in) , (take_in, take_out), (to_pad, ti_pad,  yard_state), (yard_out_slots, yard_out_slots_), mask = decide_blocks(env)
    
    for day in range(7):
        
        print(f"\nDay: {day} / {n_take_out}, {n_take_in} ")
        """
        actions: [batch_size, pad_len, (blocks, slots, take_in_slots)]
        yard_out_slots: [length  width  height  location  count]
        """
        if np.random.random() > eps_threshold:
            to_pad, ti_pad, yard_state = torch.FloatTensor(to_pad).unsqueeze(0), torch.FloatTensor(ti_pad).unsqueeze(0), torch.FloatTensor(yard_state).unsqueeze(0)
            
            state, actions = policy_net(to_pad, ti_pad, yard_state, yard_out_slots_) 
        else:
            state = torch.concat([torch.FloatTensor(to_pad).unsqueeze(0), torch.FloatTensor(ti_pad).unsqueeze(0), torch.FloatTensor(yard_state).unsqueeze(0)], axis=1)
            actions = np.zeros((1, 100, 3), dtype=int)
            
            # random take_out order
            take_out_rand = sample(list(range(len(take_out))), len(take_out))
            actions[:, :len(take_out_rand), 0] = take_out_rand
            
            temp_history = []
            for idx, (_, block) in enumerate(take_out.iloc[take_out_rand].iterrows()):
                viable_slots = np.arange(300).reshape(1, -1)[np.all(block[["length", "width"]].values < yard_out_slots_[:, :, :2], axis=-1)]
                choices = np.random.choice(viable_slots, 1)
                actions[:, idx, 1] = choices
                yard_out_slots_[:, choices, 4] -= 1
                yard_out_slots_[yard_out_slots_[:, :, 4] == 0] = 0
                temp_history.append(np.sum(yard_out_slots_[:, :, 4]))
            
            take_in_rand = sample(list(range(len(take_in))), len(take_in))
            actions[:, :len(take_in_rand), 2] = take_in_rand

        n_batch, perc_area = env.step(actions=actions, take_in=take_in, take_out=take_out, yard_out_slots=yard_out_slots, mask=mask)
        reward = np.mean(perc_area) * 100 - n_batch
        
        # * ________________ Get Next State _________________
        (n_take_out, n_take_in) , (take_in, take_out), (to_pad, ti_pad,  yard_state), (yard_out_slots, yard_out_slots_), mask = decide_blocks(env)
        to_next, ti_next, yard_state_next = torch.FloatTensor(to_pad).unsqueeze(0), torch.FloatTensor(ti_pad).unsqueeze(0), torch.FloatTensor(yard_state).unsqueeze(0)

        next_state = torch.concat([to_next, ti_next, yard_state_next], axis=1)
        actions = torch.LongTensor(actions)
        reward = torch.FloatTensor([[reward]])
        # print(type(state), type(actions), type(next_state), type(reward))
        memories.push(memory_template(state, actions, next_state, reward))
        
        # print(env.whole_block["location"].value_counts())
        
        if memories.can_provide_sample(BATCH_SIZE):
            experiences = memories.sample(BATCH_SIZE)
            batch = memory_template(*zip(*experiences))
            
            batch_states, batch_next_states, batch_rewards, batch_actions = batch.state, batch.next_state, batch.reward, batch.action
            current_q_values = policy_net(batch_states)
            # print(experiences[0])
            
    # print(rewards)
    # print(env.block_infos["location"].value_counts())
    
    
    pass
    
    
if __name__ == "__main__":
    run()
    pass

#%%

def fit_blocks_in_area(blocks, area_size):
    area_count = 1  # We start with one area
    # List of tuples representing the remaining length and width of areas
    # Initialize with the first area_size
    remaining_areas = [area_size]
    
    for block in blocks:
        block_fitted = False
        for i, area in enumerate(remaining_areas):
            if block[0] <= area[0] and block[1] <= area[1]:
                # Fit the block in the current area by reducing the available space
                # This is a naive reduction for demonstration purposes
                new_area = (area[0] - block[0], area[1] - block[1])
                remaining_areas[i] = new_area
                block_fitted = True
                break
        
        if not block_fitted:
            # No existing area could fit the block, so add a new area
            area_count += 1
            remaining_areas.append((area_size[0] - block[0], area_size[1] - block[1]))
            
    return area_count

# Example usage
blocks = [(2, 2), (3, 2), (1, 2), (4, 3)]
area_size = (5, 5)
print(fit_blocks_in_area(blocks, area_size))

#%%