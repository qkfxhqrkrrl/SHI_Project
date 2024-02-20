#%%

import numpy as np
np.set_printoptions(precision=4)
import pandas as pd
from random import sample
from copy import deepcopy

import os
import matplotlib.pyplot as plt

from typing import List, Dict, Optional

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
    # return size_info_

def generate_dataset(ids, naming_rule, n_blocks):
    # sizes = pd.read_csv(r"block_size.txt", sep="\t", names=["width", "length", "height", "weight"])
    sizes = pd.read_csv(os.path.join(src_dir, "block_size.txt"), sep="\t") 
    sizes = sizes.loc[sizes["중량"] > 0.0, :]
    sizes.dropna(inplace=True)

    names = [naming_rule(i) for i in ids] # TODO : 호선-블록 키로 변경
    sampled_sizes = sizes.sample(n_blocks).values
    
    blocks = pd.DataFrame(zip(ids, names, *[elem.reshape(-1) for elem in np.split(sampled_sizes, 4, axis=1)]), columns=["id", "name", "length", "width", "height", "weight"])
    # 위치 재정렬
    blocks = blocks[["id", "name", "width", "length", "height", "weight"]]
    blocks = convert_size(blocks)
    
    return blocks

def sample_dataset(n_blocks):
    # sizes = pd.read_csv(r"block_size.txt", sep="\t", names=["width", "length", "height", "weight"])
    sizes = pd.read_csv(os.path.join(src_dir, "block_size.csv")) 
    sizes = sizes.loc[sizes["weight"] > 0.0, :]
    # print(sizes.shape)
    sizes.dropna(inplace=True)
    # print(sizes.shape)
    sampled_blocks = sizes.sample(n_blocks)
    result = convert_size(sampled_blocks)
    
    return result.reset_index(drop=True)

# blocks = sample_dataset(100)
# print(blocks.sort_values("vessel_id"))

class Yard:
    def __init__(self, name : str, area_slots : Dict, blocks : pd.DataFrame = None):
        self.name = name
        self.blocks : pd.DataFrame = None
        self.block_count_limit = deepcopy(area_slots)
        self.blocks_count_by_size : pd.DataFrame = pd.DataFrame(None, index=list(range(10, 31, 5)), columns=list(range(10, 31, 5)))
        self.available_blocks_count = area_slots
        self.blocks_by_size : Dict[str, pd.DataFrame] = {}
        if blocks is not None:
            self.init_blocks(blocks)
        else:
            self.blocks = None
    
    def move_block_to(self, block_ids : List[int], yard :"Yard"):
        pass
        # yard.blocks[] = self.blocks[name]
            
    def init_blocks(self, blocks : pd.DataFrame = None):   
        self.blocks = blocks
        # self.update()
    
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
    def __init__(self, name: str, area_slots : pd.DataFrame, size_allowance : np.array = None, blocks : pd.DataFrame = None):
        super().__init__(name, area_slots, blocks)
        self.size_allowance = size_allowance 
        self.area_slots = area_slots
        self._init_slots()
        
    
    def move_block_to(self, block_ids: List[int], yard: "YardInside"):
        blocks_to_move = self.blocks.loc[self.blocks["id"].isin(block_ids)]
        # 옮기고자 하는 block id를 지정해서 변경
        yard.blocks = pd.concat([yard.blocks, blocks_to_move], axis=0)
        self.blocks = self.blocks.loc[~self.blocks["id"].isin(block_ids), :]
        self.update()
        yard.update()
        
    def _check_information(self):
        print("Size allowance: \n", self.size_allowance)
        print("Area slots: \n", self.area_slots)
        
    def _init_slots(self):
        pass
        
class Barge():
    def __init__(self, name: str, size_allowance: np.array) -> None:
        self.name = name
        self.size_allowance = size_allowance
        self.eta = None
        self.destination = None
        self.isLoading = None

    def get_allowable_size(self) -> np.array:
        return self.allowable_size
    
    def _get_barge_info(self):
        return self.name, self.size_allowance
    
    def is_block_loadable(self, blocks: pd.DataFrame):
        return np.any(blocks[["width", "length", "weight", "height"]].values < self.size_allowance, axis=1)
        
    # def __repr__(self) -> str:
    #     return self.name
    
class RLEnv():
    def __init__(self) -> None:
        self.yard_in = YardInside(name="사내", area_slots=None, blocks=None)
        self.block_infos : pd.DataFrame = sample_dataset(600)
        self.block_infos["location"] = None
        self._init_env()
        
        # ! 바지선 정보 추가
        self.available_barges = []
        self.vessels_in_yard = []
    
    def _init_env(self):
        # ! 사외 적치장 정보 추가
        self.yards_out_info = pd.read_csv(os.path.join(src_dir, "Outside_Yard_infos.csv"))
        self.yards_out_slot_sizes = pd.read_csv(os.path.join(src_dir, "Yard_capacity.csv"))
        # print(self.yards_out_slot_sizes)
        self.yards_out_slot_sizes.fillna(float("inf"), inplace=True)
        self.yards_out_info.fillna(float("inf"), inplace=True)
        
        self.yards_out = {}
        start_idx = 0
        for idx, info in self.yards_out_info.iterrows():
            # _____________ Set by random for now _____________________
            count_max = self.yards_out_slot_sizes.loc[self.yards_out_slot_sizes["name"] == info["name"], "count"].sum()
            # print(info["name"], count_max)
            rand_n_block = np.random.randint(int(count_max*0.5), int(count_max*0.8))
            mask = np.zeros_like(self.block_infos.iloc[:, 0], dtype=bool)
            mask[start_idx:start_idx+rand_n_block] = True
            mask[np.any(self.block_infos[["width", "length", "weight", "height"]].values \
                >= info[["width", "length", "weight", "height"]].values, axis=1)] = False
            rand_blocks = self.block_infos.loc[mask, :].copy()
            self.block_infos.loc[mask, "location"] = info["name"]
            # _________________________________________________________
            self.yards_out[info["name"]] = YardOutside(name=info["name"], area_slots= self.yards_out_slot_sizes.loc[self.yards_out_slot_sizes["name"] == info["name"]], \
                size_allowance=np.array([info["width"], info["length"], info["weight"], info["height"]]), blocks=rand_blocks)
            
            # print(info["name"])
            # self.yards_out[info["name"]]._check_information()
            start_idx += rand_n_block
        self.block_infos.loc[self.block_infos["location"].isnull(), "location"] = "사내"
        
        self.barge_info = pd.read_csv(os.path.join(src_dir, "Barge_infos.csv"))
        self.barges = {}
        for idx, info in self.barge_info.iterrows():
            self.barges[info["name"]] = Barge(name=info["name"], size_allowance=np.array([info["width"], info["length"]]))
        # self.barges["name"] =
        # display(self.block_infos["location"].value_counts())
        # display(self.block_infos)
    
    def generate_mask(self, barge_list: List["Barge"], possible_blocks_out: List[pd.DataFrame], requested_blocks: List[pd.DataFrame]):
        yard_list: List["YardOutside"] = [self.yards_out[name] for name in requested_blocks["location"].unique()]
        # * barge <-> blocks 
        mask_bb = np.ones((len(barge_list), len(possible_blocks_out)))
        for idx, barge in enumerate(barge_list):
            mask_bb[idx, :] = barge.is_block_loadable(possible_blocks_out)
        
    def update_yard_out_info(self):
        # TODO : 사외 적치장의 적치율 업데이트
        pass
    
    def advance_time(self):
        # TODO : 바지선 ETA 업데이트, 사내 적치장에 선박 접안 가능한지 확인
        pass

def test():
    env = RLEnv()
    # TODO : 샘플 데이터 불러와서 generate mask 잘 되는지 확인
    env.generate_mask(sample(env.barges.values(), 10), sample(env.yards_out.values(), 10), sample(env.block_infos.values(), 10))
    # print(temp.loc[temp["width"] >= 18])
test()
    

# temp = generate_dataset(np.arange(0, 10), lambda x : f"block_temp_{x}", 10)
# print(temp)
def generate_random_area_slots():
    area_slots = {}
    for i in range(10, 31, 5):
        for j in range(i, 31, 5):
            area_slots[(j, i)] = np.random.randint(100, 150)
            
    return area_slots

    
def run():
    
    area_slots = generate_random_area_slots()
    blocks = generate_dataset(np.arange(500), lambda x : f"block_in_{x}", 500)
    yard_in = YardInside("사내", area_slots, blocks)

    area_slots = generate_random_area_slots()
    scale = np.array([15, 17, 5, 80]) # *(Length, Width, Height, Weight)
    blocks = generate_dataset(np.arange(500, 600), lambda x : f"block_out_{x}", 100)
    yard_out1 = YardOutside(name="덕곡2", area_slots=area_slots, size_allowance=scale, blocks=blocks)
    
    area_slots = generate_random_area_slots()
    scale = np.array([22, 25, float("inf"), 300]) # * 해당없음은 고려 안해도 된다고 이해하고 모든 경우가 포함될 수 있도록 하기 위해 무한대로 설정
    blocks = generate_dataset(np.arange(600, 700), lambda x : f"block_out_{x}", 100)
    yard_out2 = YardOutside(name="오비1", area_slots=area_slots, size_allowance=scale, blocks=blocks)
    
    # print(yard_in.available_blocks_count)
    # print(yard_out1.available_blocks_count)
    # print(yard_in.blocks_count_by_size)
    # print(yard_out1.blocks_count_by_size)
    # print()

    sample_ids = sample(yard_in.blocks["id"].to_list(), 200)
    # yard_in.move_block_to(sample_ids, yard_out2)
    movable_block, immovable_block = yard_in.get_movable_block(sample_ids, yard_out1)
    print(movable_block.shape)
    
    movable_block, immovable_block = yard_in.get_movable_block(sample_ids, yard_out2)
    print(movable_block.shape)
    
    
if __name__ == "__main__":
    run()
    pass

#%%
import torch
import torch.nn as nn

def temp_sample(n_blocks, n_yards, n_barges):
    blocks_to_be_out = generate_dataset(np.arange(0, n_blocks), lambda x : f"block_in_{x}", n_blocks)
    # print(blocks_to_be_out)
    
    blocks_to_be_in = generate_dataset(np.arange(0, n_blocks), lambda x : f"block_out_{x}", n_blocks)
    blocks_to_be_in["location"] = np.random.randint(0, n_yards, n_blocks)
    # print(temp)

    # yard_const = pd.read_csv("./Src/Outside_Yard_infos.csv", encoding="cp949")
    yard_const = pd.read_csv(os.path.join(src_dir, "Outside_Yard_infos.csv"))
    print(yard_const)
    yard_const.rename(columns={"폭": "width", "길이": "length", "중량": "weight", "높이" : "height", "적치장" : "name"}, inplace=True) 
    yard_const.fillna(float("inf"), inplace=True)
    # yard_const = yard_const.sample(n_yards)
    yard_const = yard_const.loc[yard_const["name"].isin(["덕곡1", "덕곡2", "오비3"]), :] # Constraint 만족하는지 확인용으로 선택
    # print("Yard constraints: \n", yard_const)

    barge_const = pd.read_csv(os.path.join(src_dir, "Barge_infos.csv"))
    # print(barge_const)
    barge_const = barge_const.loc[barge_const["name"].isin(["한진9001", "신동7000", "신정6001", "부경10001"]), :]
    barge_const = barge_const[["name", "width", "length", "parallel"]] # 컬럼 위치 스왑
    # barge_const = barge_const.sample(n_barges)
    # print("Barge constraints: \n", barge_const)
    
    return blocks_to_be_out, blocks_to_be_in, yard_const, barge_const

n_blocks = 7 # I
n_yards = 3 # J
n_barges = 4 # K

blocks_to_be_out, blocks_to_be_in, yard_const, barge_const = temp_sample(n_blocks, n_yards, n_barges)
print("반출 블록: \n",blocks_to_be_out[["width", "length"]])
print("반입 블록: \n",blocks_to_be_in[["width", "length", "location"]])
print("적치장: \n", yard_const[["width", "length"]])
print("바지선: \n", barge_const[["width", "length"]])

#%%
# print("반출 블록: \n",blocks_to_be_out[["width", "length"]])
# print("반입 블록: \n",blocks_to_be_in[["width", "length", "location"]])
# print("적치장: \n", yard_const[["width", "length"]])
# print("바지선: \n", barge_const[["width", "length"]])

print(blocks_to_be_out[["width", "length"]])
print(blocks_to_be_in[["width", "length", "location"]])
print( yard_const[["width", "length"]])
print( barge_const[["width", "length"]])

def masking_by_constraints(blocks_to_be_out, blocks_to_be_in, yards_const, barges_const):
    
    n_blocks, n_yards, n_barges = blocks_to_be_out.shape[0], yards_const.shape[0], barges_const.shape[0]
    # ? 크기 제한이 보통 가장 길이가 긴 곳으로 정렬하는 게 맞는 거 같은데 혹시 모르니까 나중에 질문하기 
    # input = torch.randn(n_blocks, n_yards, n_barges) # (블록, 적치장, 바지선) -> 적치장 j에 바지선 k로 보낸다고 가정할 때 어느 블록이 할당되는 것이 가장 좋은지
    # input = np.random.randn(n_barges, n_yards, n_blocks) # (바지선, 적치장, 블록) -> 블록 i를 적치장 j로 보낸다고 가정할 때 어느 바지선에 할당되는 것이 가장 좋은지
    input = torch.randn(n_barges, n_yards, n_blocks) # (바지선, 적치장, 블록) -> 블록 i를 적치장 j로 보낸다고 가정할 때 어느 바지선에 할당되는 것이 가장 좋은지
    
    # print("Before masking \n", input)
    # print("\n")

    # * 야드 제약 조건에 대해서 마스킹
    # TODO : 적치 현황에 따라 추가적인 마스킹
    mask = torch.zeros((n_yards, n_blocks), dtype=torch.bool)
    for idx, yard_const in enumerate(yards_const[["width", "length"]].values):
        mask[idx] = torch.any(torch.BoolTensor(blocks_to_be_out[["width", "length"]].values >= yard_const), dim=1)
    input[:, mask] = -1e+5
    
    # * 바지 제약 조건에 대해서 마스킹
    input = input.permute(1, 0, 2) # 블록에 대해서는 그대로 놔두고 야드와 바지선 차원 변경 / (적치장, 바지선, 블록)
    mask = torch.zeros((n_barges, n_blocks), dtype=torch.bool)
    for idx, barge_const in enumerate(barges_const[["width", "length"]].values):
        print(torch.any(torch.BoolTensor(blocks_to_be_out[["width", "length"]].values >= barge_const), dim=1))
        mask[idx] = torch.any(torch.BoolTensor(blocks_to_be_out[["width", "length"]].values >= barge_const), dim=1)
    input[:, mask] = -1e+5
    
    # * 가져와야 하는 블록 제원에 의해 바지선이 특정 적치장에 가면 안 되는 제약조건
    input = input.permute(2, 1, 0) # (블록, 바지선, 적치장)
    mask = torch.zeros((n_barges, n_yards), dtype=torch.bool)
    for idx, barge_const in enumerate(barges_const[["width", "length"]].values):
        # 1개만 가져와도 되니까 np.all로 조건을 비교
        # any로 하면 강한 제약 조건
        forbidden_location = blocks_to_be_in.loc[np.all(blocks_to_be_in[["width", "length"]].values >= barge_const, axis=1), "location"].unique()
        # print(np.any(blocks_to_be_in[["width", "length"]].values >= barge_const, axis=1))
        mask[idx, forbidden_location] = True
    input[:, mask] = -1e+5
    
    
    soft_max = nn.Softmax(dim=-1)
    print(input)
    # reshape으로 풀어주는 이유는 제약 조건을 모두 동시에 고려하기 위해서
    block_probs = soft_max(input.reshape(n_blocks, -1)) #  (블록, 바지선, 적치장) -> (블록, 바지선*적치장)
    vals, block_idxs = torch.max(block_probs, dim=-1)
    print(vals)
    barge_choices = torch.div(block_idxs, n_yards, rounding_mode='trunc')
    yard_choices = block_idxs % n_yards
    print(yard_choices)
    print(barge_choices)
    
    # TODO: 선택한 경우에서 모두 바지선에 태울 수 있는 지 확인 
    # print([(torch.div(val, n_yards, rounding_mode='trunc'), val%n_yards) for val in block_idxs])
    
    # for i in range(n_blocks):
    #     n_ith_blocks = torch.sum(idxs==i)
    #     if n_ith_blocks > barge_limit:
    #         # 바지선의 수용 가능한
    #         pass

    # print(output)
    # print(torch.sum(output, dim=1))

masking_by_constraints(blocks_to_be_out, blocks_to_be_in, yard_const, barge_const)

def masking_by_constraints_2d_to_3d(blocks_to_be_out, blocks_to_be_in, yards_const, barges_const):
    
    n_blocks, n_yards, n_barges = blocks_to_be_out.shape[0], yards_const.shape[0], barges_const.shape[0]
    # ? 크기 제한이 보통 가장 길이가 긴 곳으로 정렬하는 게 맞는 거 같은데 혹시 모르니까 나중에 질문하기 
    # input = torch.randn(n_blocks, n_yards, n_barges) # (블록, 적치장, 바지선) -> 적치장 j에 바지선 k로 보낸다고 가정할 때 어느 블록이 할당되는 것이 가장 좋은지
    # input = np.random.randn(n_barges, n_yards, n_blocks) # (바지선, 적치장, 블록) -> 블록 i를 적치장 j로 보낸다고 가정할 때 어느 바지선에 할당되는 것이 가장 좋은지
    input = torch.randn(n_barges, n_yards, n_blocks) # (바지선, 적치장, 블록) -> 블록 i를 적치장 j로 보낸다고 가정할 때 어느 바지선에 할당되는 것이 가장 좋은지
    
    import itertools

    blocks_to_be_out.rename(columns={"width": "block_out_width", "length": "block_out_length", "height": "block_out_height", "weight": "block_out_weight"}, inplace=True)
    array1 = list(range(7))
    blocks_to_be_out["block_out"] = array1
    display(blocks_to_be_out)
    
    blocks_to_be_in.rename(columns={"width": "block_in_width", "length": "block_in_length", "height": "block_in_height", "weight": "block_in_weight"}, inplace=True)
    array1 = list(range(7))
    blocks_to_be_in["block_in"] = array1
    
    yards_const.rename(columns={"width": "yards_width", "length": "yards_length", "height": "yards_height", "weight": "yards_weight"}, inplace=True)
    array2 = [f'yard_{chr(i)}' for i in range(ord('A'), ord('A')+3)]
    yards_const["yard"] = array2
    display(yards_const)
    
    # display(barges_const)
    barges_const.rename(columns={"width": "barges_width", "length": "barges_length"}, inplace=True)
    array3 = [f'barge_{chr(i)}' for i in range(ord('A'), ord('A')+4)]
    barges_const["barge"] = array3


    combinations = list(itertools.product(array1, array2, array3))
    combinations = pd.DataFrame(combinations, columns=["block", "yard", "barge"])
    
    combinations = pd.merge(combinations, blocks_to_be_out, on="block_out")
    combinations = pd.merge(combinations, blocks_to_be_in, on="block_in")
    combinations = pd.merge(combinations, yards_const, on="yard")
    combinations = pd.merge(combinations, barges_const, on="barge")
    combinations["mask"] = True
    
    combinations.loc[
        (combinations["block_in_width"] > combinations["barges_width"]) |
        (combinations["block_in_length"] > combinations["barges_length"]) |
        (combinations["block_out_width"] > combinations["yards_width"]) | 
        (combinations["block_out_length"] > combinations["yards_length"]) | 
        (combinations["block_out_height"] > combinations["yards_height"]) |
        (combinations["block_out_weight"] > combinations["yards_weight"]) |
        (combinations["block_out_width"] > combinations["barges_width"]) | 
        (combinations["block_out_length"] > combinations["barges_length"]) , "mask"] = False
    
    mask = combinations["mask"].values.reshape(n_barges, n_yards, n_blocks)
    mask = mask.transpose(2, 1, 0)
    print(mask)
    # display(combinations)
    

# masking_by_constraints_2d_to_3d(blocks_to_be_out.copy(), blocks_to_be_in.copy(), yard_const.copy(), barge_const.copy())
# %%

# * Constraint toy examples
# blocks_to_be_out = pd.DataFrame([
#     [5, 11],
#     [12, 14],
#     [11, 17],
#     [14, 17],
#     [17, 17],
#     [17, 21],
#     [20, 24],
#     [22, 24],
#     ],columns=["width", "length"])

# blocks_to_be_in = pd.DataFrame([
#     [5, 11, 0],
#     [12, 14, 0],
#     [11, 17, 1],
#     [14, 17, 1],
#     [17, 17, 1],
#     [17, 19, 2],
#     [19, 24, 2],
#     [22, 24, 2],
#     ],columns=["width", "length", "location"])

# yard_const = pd.DataFrame([
#     [15, 15],
#     [18, 18],
#     [25, 25],
# ], columns=["width", "length"])

# barge_const = pd.DataFrame([
#     [15, 50],
#     [17, 50],
#     [25, 50],
#     [25, 50],
# ], columns=["width", "length"])


# import numpy as np

# rand_mask1 = np.random.choice([True, False], size=(4,2), p=[0.8, 0.2])
# rand_mask2 = np.random.choice([True, False], size=(2,3), p=[0.8, 0.2])
# rand_mask3 = np.random.choice([True, False], size=(3,4), p=[0.8, 0.2])
# print(rand_mask1)
# print(rand_mask2)
# print(rand_mask3)

# mask = np.ones((rand_mask1.shape[0], rand_mask2.shape[0], rand_mask3.shape[0]))
# for axis_1 in range(rand_mask1.shape[0]):
#     for axis_2 in range(rand_mask2.shape[0]):
#         for axis_3 in range(rand_mask3.shape[0]):
#             mask[axis_1, axis_2, axis_3] = rand_mask1[axis_1, axis_2] * rand_mask2[axis_2, axis_3] * rand_mask3[axis_3, axis_1]
# print(mask)

# plane = []
# for r_axis in rand_mask1:
#     plane.append(r_axis.reshape(-1,1)*rand_mask2)
    
# plane = np.array(plane)

# print(plane.shape)
# print(plane)

# %%
