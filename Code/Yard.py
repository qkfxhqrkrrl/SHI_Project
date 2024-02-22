#%%

import numpy as np
np.set_printoptions(precision=4)
import pandas as pd

import torch
import torch.nn as nn

import os
from copy import deepcopy
from random import sample

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
        
    def get_movable_block(self, block_ids: List[int], yard: "YardOutside"):
        # 사외 적치장은 제약 조건이 있으므로 이 값이 조건을 여기서 고려해주고 처리 
        blocks_to_move = self.blocks.loc[self.blocks["id"].isin(block_ids)]
        blocks_to_move_info = blocks_to_move[["width", "length", "height", "weight"]].values
        mask = np.any(blocks_to_move_info >= yard.size_allowance, axis=1)
        possible_blocks = blocks_to_move.loc[~mask, :].reset_index(drop=True)
        impossible_blocks = blocks_to_move.loc[mask, :].reset_index(drop=True)
        # yard.blocks = pd.concat([yard.blocks, possible_blocks], axis=0).reset_index(drop=True)
        # TODO : 제외할 건 제외하고
        # self.blocks = self.blocks.loc[~self.blocks["id"].isin(block_ids), :].reset_index(drop=True)
        
        return possible_blocks, impossible_blocks
    
    def move_block_to(self, blocks_to_move: pd.DataFrame, yard: Yard):
        yard.blocks = pd.concat([yard.blocks, blocks_to_move], axis=0)
        self.blocks = self.blocks.loc[~self.blocks["id"].isin(blocks_to_move["id"]), :]
        # self.update()
        # yard.update()
        
class YardOutside(Yard):
    def __init__(self, name: str, area_slots : pd.DataFrame, blocks : pd.DataFrame = None):
        super().__init__(name, area_slots, blocks)
        self.orig_slots = area_slots.copy()
        self.orig_slots["id"] = range(len(area_slots))
        self.area_slots = area_slots[["vessel_id", "block", "length", "width_max","height"]].rename(columns={"width_max": "width"})
        self.area_slots["id"] = range(len(area_slots))
        self.columns = self.area_slots.columns
        self.whole_area = np.sum(self.area_slots["length"] * self.area_slots["width"])
        # print(self.area_slots)
        # self.area_slots = area_slots[["length", "width"]].values
        self._init_slot()
        # print(self.blocks)
        # self._init_slots()
        # if self.name == "신한내":
        #     self._init_slots()
        
    
    def move_block_to(self, block_ids: List[int], yard: "YardInside"):
        blocks_to_move = self.blocks.loc[self.blocks["id"].isin(block_ids)]
        # 옮기고자 하는 block id를 지정해서 변경
        yard.blocks = pd.concat([yard.blocks, blocks_to_move], axis=0)
        self.blocks = self.blocks.loc[~self.blocks["id"].isin(block_ids), :]
        self.update()
        yard.update()
        
    def generate_mask(self, blocks_to_move: pd.DataFrame):
        return np.any(blocks_to_move[["width", "length", "weight", "height"]].values >= self.size_allowance, axis=1)
        
    def _check_information(self):
        print("Area slots: \n", self.area_slots)
    
    def _init_slot(self):
        # 슬롯의 크기와 들어가 있는 블록의 크기를 비교해서 남는 공간을 새로운 슬롯으로 변경
        # 남는 공간은 (remaining_width * length)와 (width * remaining_length)큰 공간을 기준으로 선정
        blocks_ = []
        # display(self.area_slots)
        df_merged = pd.merge(self.area_slots, self.blocks , on=["vessel_id", "block"], suffixes=("_slots","_block"), how="outer")
        df_merged[["remaining_width", "remaining_length"]] = df_merged[["width_slots", "length_slots"]].values - df_merged[["width_block", "length_block"]].values
        df_merged["isFull"] = np.any(df_merged[["remaining_width", "remaining_length"]].values >= 11, axis=1) # 캐리어로 인해 11은 확보가 되어야 함
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
        # display(self.area_slots)
    
    def _concat_by_id(self):
        # TODO: 원래 하나의 칸이였던 경우를 블록이 비어서 없어지는 경우를 고려하기 위해서 여유가 되면 id별로 다시 합쳐주기 
        pass

class Args:
    def __init__(self) -> None:
        pass

class DQNAgent(nn.Module):
    def __init__(self, n_yard_out) -> None:
        super(DQNAgent, self).__init__()
        self.feat_ext = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256, 100),
            nn.ReLU()
        )
        
        self.n_yard_out = n_yard_out
        self.rnn = nn.LSTM(input_size=5, hidden_size=256, batch_first=True)
        self.fc = nn.Linear(256, n_yard_out)
        self.soft_max = nn.Softmax(dim=-1)
        
    def masking_by_slots(self, block_info, slot_info):
        # output: [batch_size  n_block  n_yard]
        # 해당 블록이 들어갈 수 있는 슬롯이 없으면 False로 걸러주기
        
        mask = torch.zeros((block_info.shape[0], 100, self.n_yard_out), dtype=torch.bool)
        
        for i in range(100): # 패딩 길이를 100으로 해줬으니까
            block_size = block_info[:, i]
            for yard_idx in range(self.n_yard_out):
                yard_slot = slot_info[slot_info[:, :, -1] == yard_idx]
                print(yard_slot)
            break
            
        
        return mask
    
    def forward(self, take_out, take_in, feat_vec, slot_info):
        # take_out, take_in, feat_vec: [length  width  height]
        # slot_info: [length  width  height  location  count]
        # print(take_in[take_in[:, 0] != -1])
        if len(take_out.shape) < 3:
            take_in, take_out, feat_vec, slot_info = take_in.unsqueeze(0), take_out.unsqueeze(0), feat_vec.unsqueeze(0), slot_info.unsqueeze(0)
        take_in, take_out, feat_vec, slot_info = \
            torch.concat([take_in, take_in[:, torch.randperm(take_in.shape[1])]], dim=0), torch.concat([take_out, take_out[:, torch.randperm(take_out.shape[1])]], dim=0), \
            torch.concat([feat_vec]*2, dim=0), torch.concat([slot_info]*2, dim=0)
        batch_size = slot_info.shape[0]
        # print(take_out[take_out[:, :, 0] != -1])
        # print((take_out[:, :, 0] != -1).shape)
        # print(slot_info)
        
        feat_vec = self.feat_ext(torch.concat([take_out, take_in, feat_vec], dim=-2).transpose(-1, -2)).transpose(-1, -2)
        
        take_out_state = take_out + feat_vec
        take_in_state = take_in + feat_vec
        
        hids, _ = self.rnn(take_out_state)
        
        dist_prob = self.fc(hids)
        if len(take_out.shape) < 3:
            self.masking_by_slots(take_out, slot_info)
        else :
            self.masking_by_slots(take_out, slot_info)
        # to_prob_dist = self.fc(to_hidden)
        
        
        # for prob_dist in to_prob_dist:
        #     prob_dist = torch.max(prob_dist, dim=-1)
        # to_prob_dist = self.soft_max(to_prob_dist)
        
        # print(to_prob_dist.shape)
        
        # print(result.shape)
        
        pass
        

class RLEnv():
    def __init__(self) -> None:
        self.yard_in = YardInside(name="사내", area_slots=None, blocks=None)
        # Set by random for now
        self.whole_block : pd.DataFrame = sample_dataset(600)
        self.whole_block["location"] = None
        self.target_columns = ["length", "width", "height", "weight", "location"]
        self._init_env()
        self.model = DQNAgent(n_yard_out=len(self.yards_out))
        
        # ! 바지선 정보 추가
        self.available_barges = []
        self.vessels_in_yard = []
        
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
        self.labels_encoder = dict(zip(self.yards_out_slot_sizes["name"], self.yards_out_slot_sizes["id"]))
        self.labels_encoder["사내"] = 0
        
    
    def get_state(self, possible_take_out: pd.DataFrame, take_in: pd.DataFrame):
        
        to_pad = (-1) * np.ones((100, len(self.target_columns)))
        # print(possible_take_out)
        possible_take_out["location"] = possible_take_out["location"].map(self.labels_encoder)
        to_pad[:possible_take_out.shape[0], :] = possible_take_out[self.target_columns].values
        
        ti_pad = (-1) * np.ones((100, len(self.target_columns)))
        take_in["location"] = take_in["location"].map(self.labels_encoder)
        ti_pad[:take_in.shape[0], :] = take_in[self.target_columns].values
        
        state_ = []
        yard_out_slots = []
        
        for name, yard in self.yards_out.items():
            state_part_ = yard.area_slots
            # display(state_part_)
            state_part_["location"] = self.labels_encoder[name]
            state_part_.loc[state_part_["height"] == float("inf"), "height"] = -1
            state_part_.loc[state_part_["weight"].isnull(), "weight"] = -1
            count = state_part_[["length", "width", "height", "location"]].groupby(["length", "width"], as_index=False).value_counts()
            yard_out_slots.append(count.values)
            
            state_part_ = state_part_[self.target_columns].values
            
            pad_len = yard.orig_slots.shape[0] * 2
            state_part = (-1) * np.ones((pad_len, state_part_.shape[1]))
            state_part[:state_part_.shape[0]] = state_part_
            # state_part.loc[state_part["height"] == float("inf"), "height"] = -1
            state_.append(state_part)
            
        yard_state = np.concatenate(state_, axis=0)
        yard_out_slots = np.concatenate(yard_out_slots, axis=0)
        
        return to_pad, ti_pad, yard_state, yard_out_slots
        
    def make_an_action(self, take_out: pd.DataFrame, take_in: pd.DataFrame):
        
        
        # 우선 사외 적치 현황을 표현하려면, 할당 가능한 전체 슬롯 수가 정해져있으니 그만큼 제로패딩을 통해 consistency 확보
        # ? 슬롯을 업데이트 해서 그 수가 달라지면 어떻게 해?
        # feat: ["width", "length", "weight", "height", "location"]
        
        # print(possible_blocks_out.shape, requested_blocks.shape)
        to_pad, ti_pad, yard_state, yard_out_slots = self.get_state(take_out, take_in)
        to_pad, ti_pad, yard_state, yard_out_slots = torch.FloatTensor(to_pad), torch.FloatTensor(ti_pad), torch.FloatTensor(yard_state), torch.FloatTensor(yard_out_slots)
        
        state = self.model(to_pad, ti_pad, yard_state, yard_out_slots)
        
        # print(state.shape)
        
        # model = nn.Sequential(
        #     nn.Linear(5, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 1)
        # )
        # # temp_state = torch.rand((self.yards_out_slot_sizes["count"].sum(), 5)) # 사외 적치 현황, 반입해야하는 블록들, 반출할 블록들
        # result = model(state).squeeze(-1)
        # TODO: Result의 형태를 어떻게 일정하게 만들지?
        # print(result.shape)
        # mask_yb = self.generate_mask(possible_blocks_out, requested_blocks)
        
        pass    
        
    def update_yard_out_info(self):
        # TODO : 사외 적치장의 적치율 업데이트
        pass
    
    def advance_time(self):
        # TODO : 바지선 ETA 업데이트, 사내 적치장에 선박 접안 가능한지 확인
        pass
    

    
def run():
    env = RLEnv()
    # print(env.block_infos["location"].value_counts())
    n_take_out = np.random.randint(25,35)
    n_take_in = np.random.randint(30,40)
    blocks_to_be_out = env.whole_block.loc[env.whole_block["location"] == "사내"].sample(n_take_out)
    blocks_to_be_in = env.whole_block.loc[env.whole_block["location"] != "사내"].sample(n_take_in)
    env.make_an_action(take_out=blocks_to_be_out.copy(), take_in=blocks_to_be_in.copy())
    
    pass
    
    
if __name__ == "__main__":
    run()
    pass

#%%

# batch_size = 2
# seq_len = 10
# n_feat = 4
# input_ = torch.rand((batch_size, n_feat, seq_len))
# model = nn.Conv1d(in_channels=n_feat, out_channels=32, kernel_size=3, padding=1)
# result = model(input_)
# print(result.shape)

# t_to_ = torch.arange(37 * 4).reshape(37, 4)
# t_to = torch.zeros((100, 4))
# t_to[:len(t_to_), :] = t_to_

# t_ti_ = torch.arange(47 * 4).reshape(47, 4)
# t_ti = torch.zeros((100, 4))
# t_ti[:len(t_ti_), :] = t_ti_

# t_ys_ = torch.arange(29 * 4).reshape(29, 4)
# t_ys = torch.zeros((100, 4))
# t_ys[:len(t_ys_), :] = t_ys_

# t_con = torch.concat([t_to, t_ti, t_ys], axis=0)
# m = nn.Linear(300, 100)
# latent_vec = m(t_con.permute(1, 0)).permute(1, 0)

# t_to = latent_vec + t_to
# t_ti = latent_vec + t_ti

# print(result.shape)

temp = [(1,2), (3,4), (1,3), (1,2)]
temp = pd.DataFrame(temp)
temp = temp.groupby([0, 1], as_index=False).value_counts()
print(temp)
# np.unique(temp)
# set(temp)

# temp_m = nn.Sequential(
#     nn.Linear(300, 64),
#     nn.ReLU(),
#     nn.Linear(64, 64),
#     nn.ReLU(),
#     nn.Linear(64, 4)
# )
# print(temp.shape)
# print(temp1_.reshape(-1, 4)) # 이렇게하면 한 행이 가지는 정보가 손실됨

#%%

# def temp_sample(n_blocks, n_yards, n_barges):
#     blocks_to_be_out = sample_dataset(100)
#     # print(blocks_to_be_out)
    
#     blocks_to_be_in = sample_dataset(100)
#     blocks_to_be_in["location"] = np.random.randint(0, n_yards, n_blocks)
#     # print(temp)

#     # yard_const = pd.read_csv("./Src/Outside_Yard_infos.csv", encoding="cp949")
#     yard_const = pd.read_csv(os.path.join(src_dir, "Outside_Yard_infos.csv"))
#     print(yard_const)
#     yard_const.rename(columns={"폭": "width", "길이": "length", "중량": "weight", "높이" : "height", "적치장" : "name"}, inplace=True) 
#     yard_const.fillna(float("inf"), inplace=True)
#     # yard_const = yard_const.sample(n_yards)
#     yard_const = yard_const.loc[yard_const["name"].isin(["덕곡1", "덕곡2", "오비3"]), :] # Constraint 만족하는지 확인용으로 선택
#     # print("Yard constraints: \n", yard_const)

#     barge_const = pd.read_csv(os.path.join(src_dir, "Barge_infos.csv"))
#     # print(barge_const)
#     barge_const = barge_const.loc[barge_const["name"].isin(["한진9001", "신동7000", "신정6001", "부경10001"]), :]
#     barge_const = barge_const[["name", "width", "length", "parallel"]] # 컬럼 위치 스왑
#     # barge_const = barge_const.sample(n_barges)
#     # print("Barge constraints: \n", barge_const)
    
#     return blocks_to_be_out, blocks_to_be_in, yard_const, barge_const

# n_blocks = 7 # I
# n_yards = 3 # J
# n_barges = 4 # K

# blocks_to_be_out, blocks_to_be_in, yard_const, barge_const = temp_sample(n_blocks, n_yards, n_barges)
# print("반출 블록: \n",blocks_to_be_out[["width", "length"]])
# print("반입 블록: \n",blocks_to_be_in[["width", "length", "location"]])
# print("적치장: \n", yard_const[["width", "length"]])
# print("바지선: \n", barge_const[["width", "length"]])

# #%%
# # print("반출 블록: \n",blocks_to_be_out[["width", "length"]])
# # print("반입 블록: \n",blocks_to_be_in[["width", "length", "location"]])
# # print("적치장: \n", yard_const[["width", "length"]])
# # print("바지선: \n", barge_const[["width", "length"]])

# print(blocks_to_be_out[["width", "length"]])
# print(blocks_to_be_in[["width", "length", "location"]])
# print( yard_const[["width", "length"]])
# print( barge_const[["width", "length"]])

# def masking_by_constraints(blocks_to_be_out, blocks_to_be_in, yards_const, barges_const):
    
#     n_blocks, n_yards, n_barges = blocks_to_be_out.shape[0], yards_const.shape[0], barges_const.shape[0]
#     # ? 크기 제한이 보통 가장 길이가 긴 곳으로 정렬하는 게 맞는 거 같은데 혹시 모르니까 나중에 질문하기 
#     # input = torch.randn(n_blocks, n_yards, n_barges) # (블록, 적치장, 바지선) -> 적치장 j에 바지선 k로 보낸다고 가정할 때 어느 블록이 할당되는 것이 가장 좋은지
#     # input = np.random.randn(n_barges, n_yards, n_blocks) # (바지선, 적치장, 블록) -> 블록 i를 적치장 j로 보낸다고 가정할 때 어느 바지선에 할당되는 것이 가장 좋은지
#     input = torch.randn(n_barges, n_yards, n_blocks) # (바지선, 적치장, 블록) -> 블록 i를 적치장 j로 보낸다고 가정할 때 어느 바지선에 할당되는 것이 가장 좋은지
    
#     # print("Before masking \n", input)
#     # print("\n")

#     # * 야드 제약 조건에 대해서 마스킹
#     # TODO : 적치 현황에 따라 추가적인 마스킹
#     mask = torch.zeros((n_yards, n_blocks), dtype=torch.bool)
#     for idx, yard_const in enumerate(yards_const[["width", "length"]].values):
#         mask[idx] = torch.any(torch.BoolTensor(blocks_to_be_out[["width", "length"]].values >= yard_const), dim=1)
#     input[:, mask] = -1e+5
    
#     # * 바지 제약 조건에 대해서 마스킹
#     input = input.permute(1, 0, 2) # 블록에 대해서는 그대로 놔두고 야드와 바지선 차원 변경 / (적치장, 바지선, 블록)
#     mask = torch.zeros((n_barges, n_blocks), dtype=torch.bool)
#     for idx, barge_const in enumerate(barges_const[["width", "length"]].values):
#         print(torch.any(torch.BoolTensor(blocks_to_be_out[["width", "length"]].values >= barge_const), dim=1))
#         mask[idx] = torch.any(torch.BoolTensor(blocks_to_be_out[["width", "length"]].values >= barge_const), dim=1)
#     input[:, mask] = -1e+5
    
#     # * 가져와야 하는 블록 제원에 의해 바지선이 특정 적치장에 가면 안 되는 제약조건
#     input = input.permute(2, 1, 0) # (블록, 바지선, 적치장)
#     mask = torch.zeros((n_barges, n_yards), dtype=torch.bool)
#     for idx, barge_const in enumerate(barges_const[["width", "length"]].values):
#         # 1개만 가져와도 되니까 np.all로 조건을 비교
#         # any로 하면 강한 제약 조건
#         forbidden_location = blocks_to_be_in.loc[np.all(blocks_to_be_in[["width", "length"]].values >= barge_const, axis=1), "location"].unique()
#         # print(np.any(blocks_to_be_in[["width", "length"]].values >= barge_const, axis=1))
#         mask[idx, forbidden_location] = True
#     input[:, mask] = -1e+5
    
    
#     soft_max = nn.Softmax(dim=-1)
#     print(input)
#     # reshape으로 풀어주는 이유는 제약 조건을 모두 동시에 고려하기 위해서
#     block_probs = soft_max(input.reshape(n_blocks, -1)) #  (블록, 바지선, 적치장) -> (블록, 바지선*적치장)
#     vals, block_idxs = torch.max(block_probs, dim=-1)
#     print(vals)
#     barge_choices = torch.div(block_idxs, n_yards, rounding_mode='trunc')
#     yard_choices = block_idxs % n_yards
#     print(yard_choices)
#     print(barge_choices)
    
#     # TODO: 선택한 경우에서 모두 바지선에 태울 수 있는 지 확인 
#     # print([(torch.div(val, n_yards, rounding_mode='trunc'), val%n_yards) for val in block_idxs])
    
#     # for i in range(n_blocks):
#     #     n_ith_blocks = torch.sum(idxs==i)
#     #     if n_ith_blocks > barge_limit:
#     #         # 바지선의 수용 가능한
#     #         pass

#     # print(output)
#     # print(torch.sum(output, dim=1))

# masking_by_constraints(blocks_to_be_out, blocks_to_be_in, yard_const, barge_const)

# def masking_by_constraints_2d_to_3d(blocks_to_be_out, blocks_to_be_in, yards_const, barges_const):
    
#     n_blocks, n_yards, n_barges = blocks_to_be_out.shape[0], yards_const.shape[0], barges_const.shape[0]
#     # ? 크기 제한이 보통 가장 길이가 긴 곳으로 정렬하는 게 맞는 거 같은데 혹시 모르니까 나중에 질문하기 
#     # input = torch.randn(n_blocks, n_yards, n_barges) # (블록, 적치장, 바지선) -> 적치장 j에 바지선 k로 보낸다고 가정할 때 어느 블록이 할당되는 것이 가장 좋은지
#     # input = np.random.randn(n_barges, n_yards, n_blocks) # (바지선, 적치장, 블록) -> 블록 i를 적치장 j로 보낸다고 가정할 때 어느 바지선에 할당되는 것이 가장 좋은지
#     input = torch.randn(n_barges, n_yards, n_blocks) # (바지선, 적치장, 블록) -> 블록 i를 적치장 j로 보낸다고 가정할 때 어느 바지선에 할당되는 것이 가장 좋은지
    
#     import itertools

#     blocks_to_be_out.rename(columns={"width": "block_out_width", "length": "block_out_length", "height": "block_out_height", "weight": "block_out_weight"}, inplace=True)
#     array1 = list(range(7))
#     blocks_to_be_out["block_out"] = array1
#     display(blocks_to_be_out)
    
#     blocks_to_be_in.rename(columns={"width": "block_in_width", "length": "block_in_length", "height": "block_in_height", "weight": "block_in_weight"}, inplace=True)
#     array1 = list(range(7))
#     blocks_to_be_in["block_in"] = array1
    
#     yards_const.rename(columns={"width": "yards_width", "length": "yards_length", "height": "yards_height", "weight": "yards_weight"}, inplace=True)
#     array2 = [f'yard_{chr(i)}' for i in range(ord('A'), ord('A')+3)]
#     yards_const["yard"] = array2
#     display(yards_const)
    
#     # display(barges_const)
#     barges_const.rename(columns={"width": "barges_width", "length": "barges_length"}, inplace=True)
#     array3 = [f'barge_{chr(i)}' for i in range(ord('A'), ord('A')+4)]
#     barges_const["barge"] = array3


#     combinations = list(itertools.product(array1, array2, array3))
#     combinations = pd.DataFrame(combinations, columns=["block", "yard", "barge"])
    
#     combinations = pd.merge(combinations, blocks_to_be_out, on="block_out")
#     combinations = pd.merge(combinations, blocks_to_be_in, on="block_in")
#     combinations = pd.merge(combinations, yards_const, on="yard")
#     combinations = pd.merge(combinations, barges_const, on="barge")
#     combinations["mask"] = True
    
#     combinations.loc[
#         (combinations["block_in_width"] > combinations["barges_width"]) |
#         (combinations["block_in_length"] > combinations["barges_length"]) |
#         (combinations["block_out_width"] > combinations["yards_width"]) | 
#         (combinations["block_out_length"] > combinations["yards_length"]) | 
#         (combinations["block_out_height"] > combinations["yards_height"]) |
#         (combinations["block_out_weight"] > combinations["yards_weight"]) |
#         (combinations["block_out_width"] > combinations["barges_width"]) | 
#         (combinations["block_out_length"] > combinations["barges_length"]) , "mask"] = False
    
#     mask = combinations["mask"].values.reshape(n_barges, n_yards, n_blocks)
#     mask = mask.transpose(2, 1, 0)
#     print(mask)
#     # display(combinations)
    
#%%