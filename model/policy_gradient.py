import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from copy import deepcopy
import os
from torch.distributions import Categorical
from script.util import *

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
    def __init__(self, args, name: str, area_slots : pd.DataFrame, blocks : pd.DataFrame = None):
        super().__init__(name, area_slots, blocks)
        if True:
            self.area_slots = area_slots
            self.blocks = blocks
        else :
            # TODO: 아직 슬롯이 업데이트 되는 상황은 가정 X
            self.orig_slots = area_slots.copy()
            self.orig_slots["id"] = range(len(area_slots))
            
            self.area_slots = area_slots[[VESSEL_ID, BLOCK, LENGTH, "width_max",HEIGHT, LOCATION]].rename(columns={"width_max": WIDTH})
            self.area_slots["id"] = range(len(area_slots))
            
            self.blocks[LOCATION] = self.name
            self._update_slot()
            
            self.first_step_infos = {
                "area_slots": deepcopy(self.area_slots), 
                "blocks": deepcopy(self.blocks)
            }
            
            self.columns = self.area_slots.columns
            self.used_area = np.sum(self.area_slots[LENGTH] * self.area_slots[WIDTH])
        
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
        self.blocks[LOCATION] = self.name
        for idx, block in blocks.iterrows():
            count = self.area_slots.loc[self.area_slots[VESSEL_ID].isnull(), [LENGTH, WIDTH, HEIGHT, LOCATION]].groupby([LENGTH, WIDTH], as_index=False).value_counts().values
            fitting_slot = np.argmin(np.abs((block[LENGTH] * block[WIDTH]) - (count[:, 0] * count[:, 1])))
            length, width = count[fitting_slot, 0], count[fitting_slot, 1]
            slot_idx = self.area_slots[(self.area_slots[VESSEL_ID].isnull()) & (self.area_slots[LENGTH] == length) & (self.area_slots[WIDTH] == width)].index[0]
            self.area_slots.loc[self.area_slots.index[slot_idx], BLOCK] = block[BLOCK]
            self.area_slots.loc[self.area_slots.index[slot_idx], VESSEL_ID] = block[VESSEL_ID]
        
    def move_to_yard_in(self, blocks: pd.DataFrame):
        for _, block in blocks.iterrows():
            vessel_id, block_id = block[VESSEL_ID] ,block[BLOCK]
            self.blocks = self.blocks.loc[~((self.blocks[VESSEL_ID] == vessel_id) & (self.blocks[BLOCK] == block_id)), :]
            self.area_slots.loc[~((self.area_slots[VESSEL_ID] == vessel_id) & (self.area_slots[BLOCK] == block_id)), [VESSEL_ID, BLOCK]] = np.nan
        
    def _check_information(self):
        print("Area slots: \n", self.area_slots)
    
        
    def calc_remaining_area(self):
        taken_area = np.sum(self.blocks[LENGTH] * self.blocks[WIDTH])
        possible_area = np.sum(self.orig_slots[LENGTH] * self.orig_slots["width_max"])
        # TODO: 원래 하나의 칸이였던 경우를 블록이 비어서 없어지는 경우를 고려하기 위해서 여유가 되면 id별로 다시 합쳐주기 
        return possible_area, taken_area
    
    def calc_remaining_area_by_slot(self):
        remaining_area = []
        for slot_info in self.area_slots.loc[~(self.area_slots[VESSEL_ID].isnull()),[VESSEL_ID, BLOCK,LENGTH, WIDTH]].values:
            taken_space = self.blocks.loc[(self.blocks[VESSEL_ID] == slot_info[0]) & (self.blocks[BLOCK] == slot_info[1]), [LENGTH, WIDTH]].values[0]
            remaining_area.append(((slot_info[2] * slot_info[3]) - (taken_space[0] * taken_space[1])) ** 2)
        # TODO: 원래 하나의 칸이였던 경우를 블록이 비어서 없어지는 경우를 고려하기 위해서 여유가 되면 id별로 다시 합쳐주기 
        return np.sum(remaining_area)
    
#%%

import torch
import torch.nn as nn
# Assuming this is in policy_gradient.py or util.py if you add it there
class Attention(nn.Module):
    def __init__(self, hidden_dim, n_block, out_dim):
        super(Attention, self).__init__()
        # W_q, W_k, W_v for the standard attention, but Pointer-Net uses
        # W1 * (Decoder Hidden State) and W2 * (Encoder Output) + Bias
        self.w1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(n_block * hidden_dim, out_dim, bias=False)
        
    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        # encoder_outputs: [B, N_blocks, H]
        
        # [B, 1, H] -> [B, N_blocks, H]
        scores_w1 = self.w1(decoder_hidden.unsqueeze(1)).repeat(1, encoder_outputs.size(1), 1)
        # [B, N_blocks, H]
        scores_w2 = self.w2(encoder_outputs)

        # u_i = v^T tanh(W1 * h_t + W2 * e_i)
        # u_i: [B, -1] -> [B, out_dim]
        a_i = self.v(torch.tanh(scores_w1 + scores_w2).reshape(decoder_hidden.size(0), -1))
            
        # a_i: [B, N_blocks]
        # a_i = torch.softmax(u_i.reshape(decoder_hidden.size(0), -1), dim=-1)
        
        return a_i
    
#%%
class PGAgent_Att(nn.Module):
    def __init__(self, args) -> None:
        super(PGAgent_Att, self).__init__()
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
        

        # self.to_decoder = nn.LSTMCell(input_size=self.emb_dim * (self.emb_dim+1), hidden_size=args.hid_dim)
        self.to_encoder = nn.GRU(input_size=5, hidden_size=args.hid_dim, batch_first=True)
        self.activ = nn.ReLU()
        self.to_decoder = nn.GRUCell(input_size=self.emb_dim * 2, hidden_size=args.hid_dim)
        self.att = Attention(args.hid_dim, self.n_take_out, self.n_yard * self.n_max_block)
        # self.to_decoder = nn.LSTMCell(input_size=self.n_take_out * self.pad_len * self.n_yard, hidden_size=args.hid_dim)
        # self.to_fc_block = nn.Linear(args.hid_dim, self.n_max_block*self.n_yard) # 400 = 40(n_block) * 10(n_yard)
        
        # self.ti_decoder = nn.LSTMCell(input_size=self.emb_dim * (self.emb_dim+1), hidden_size=args.hid_dim)
        self.ti_decoder = nn.GRUCell(input_size=self.emb_dim * 2, hidden_size=args.hid_dim)
        self.ti_fc_block = nn.Linear(args.hid_dim, self.n_max_block*self.n_yard)

        self.yard_emb = nn.Embedding(self.n_yard, self.emb_dim)
        self.block_size_emb = nn.Linear(2, self.emb_dim)
        self.slot_size_emb = nn.Linear(2, self.emb_dim)
        self.length_emb = nn.Linear(1, self.emb_dim)
        self.width_emb = nn.Linear(1, self.emb_dim)
        
        self.device = args.device
        # TODO: 제원 정보 임베딩으로 변환하는 방법 적용 고려해보기
        
        self.soft_max = nn.Softmax(dim=-1)
    
    def init_env_info(self, infos):
        # 위에서 Init을 안 하는 이유는 이건 Env.reset과 함께 다시 초기화되어야 하는 값이기 때문
        take_in, take_out, yard_slots = infos
        
        input0 = torch.zeros(self.batch_size, self.emb_dim*2).to(self.device)
        # input0 = torch.zeros(self.batch_size, 5*(self.pad_len+1)).to(self.device)
        h0, c0 = torch.zeros((self.batch_size, self.hid_dim), device=self.device), torch.zeros((self.batch_size, self.hid_dim), device=self.device)
        
        if self.n_trip_infos_list is not None:
            self.n_trip_infos_tensor = torch.FloatTensor(self.n_trip_infos_list).to(self.device).repeat(self.batch_size, 1)
        yard_slots = np.expand_dims(yard_slots, axis=0).repeat(self.batch_size, axis=0)
        self.yard_slots = torch.FloatTensor(yard_slots).to(self.device)
        
        
        # _______________________________ Take out _______________________________
        self.block_size_info_to = torch.FloatTensor(np.expand_dims(take_out[[LENGTH, WIDTH, HEIGHT, WEIGHT, LOCATION]].values, axis=0).repeat(self.batch_size, axis=0)).to(self.device)
        encoder_mask = self.generate_mask(take_out, yard_slots) # [B n_take_out n_yard n_pad]
        self.block_mask_to = torch.zeros((self.batch_size, self.n_take_out, 1), dtype=bool) # 제외 시키고 싶은 경우를 True
        self.feasible_mask = torch.ones((self.batch_size, self.n_max_block*self.n_yard), dtype=torch.bool, device=self.device)
        self.feasible_mask[:, self.n_yard*self.n_take_out:] = False
        
        self.barge_count_to = torch.zeros((self.batch_size, self.n_yard), dtype=torch.int64, device=self.device)
        self.barge_slot_to = -1 * torch.ones((self.batch_size, self.n_yard, self.max_trip, self.barge_max_row, 2), dtype=torch.int64, device=self.device) # 블록 인덱스 저장
        self.block_lengths_on_barge_to = torch.zeros((self.batch_size, self.n_yard, self.max_trip, self.barge_max_row), dtype=torch.float32, device=self.device) # 블록 길이 저장
        self.block_width_on_barge_to = torch.zeros((self.batch_size, self.n_yard, self.max_trip, self.barge_max_row), dtype=torch.float32, device=self.device) # 블록 너비 저장
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
        self.block_size_info_ti = torch.FloatTensor(np.expand_dims(take_in[[LENGTH, WIDTH, HEIGHT, WEIGHT, LOCATION]].values, axis=0).repeat(self.batch_size, axis=0)).to(self.device)
        self.block_mask_ti = torch.zeros((self.batch_size, self.n_take_in), dtype=bool) # 제외 시키고 싶은 경우를 True
        
        self.barge_count_ti = torch.zeros((self.batch_size, self.n_yard), dtype=torch.int64, device=self.device)
        self.barge_slot_ti = -1 * torch.ones((self.batch_size, self.n_yard, self.max_trip, self.barge_max_row, 2), dtype=torch.int64, device=self.device) # 블록 인덱스 저장
        self.block_lengths_on_barge_ti = torch.zeros((self.batch_size, self.n_yard, self.max_trip, self.barge_max_row), dtype=torch.float32, device=self.device) # 블록 길이 저장
        self.block_width_on_barge_ti = torch.zeros((self.batch_size, self.n_yard, self.max_trip, self.barge_max_row), dtype=torch.float32, device=self.device) # 블록 너비 저장
        
        self.probs_history_ti = []
        self.rewards_history_ti = []
        self.action_history_ti = []
        
        return input0, h0, c0, encoder_mask
        
    def generate_mask(self, take_out: pd.DataFrame, yard_slots: np.ndarray):
        # encoder_mask = np.zeros((self.batch_size, take_out.shape[0], self.n_yard, self.pad_len), dtype=bool)
        encoder_mask = np.zeros((self.batch_size, take_out.shape[0], self.n_yard, self.pad_len), dtype=float)
        
        for b_idx, block_info in enumerate(take_out[[LENGTH, WIDTH, HEIGHT]].values):
            mask = np.all(block_info <= yard_slots[:, :, :, :3], axis=-1)# 블록 크기가 슬롯의 모든 면보다 작으면 True, 블록이 슬롯보다 큰게 하나라도 있다면 FalseZZ
            encoder_mask[:, b_idx] = mask # 
        
        # 불가능한 경우에 값들을 변경해야 하므로 위에서 구한 가능한 Case 들을 False가 되도록 Not 적용
        return ~torch.any(torch.BoolTensor(encoder_mask).to(self.device), dim=-1)
        # return torch.BoolTensor(encoder_mask).to(self.device)
        # return torch.FloatTensor(encoder_mask).to(self.device)
    
    
    def allocate_to_barge(self, block_selection, block_size, yard_selection, barge_infos):
        # TODO: 바지선에 병렬 제약을 10이 아니라 합이 23 이하가 되도록
        barge_count ,barge_slot, block_lengths_on_barge, block_width_on_barge = barge_infos
        
        cur_barge_num = barge_count[range(self.batch_size), yard_selection] # [B]
    
        block_length_b = block_size[:, 0, 0] # Unsqueeze in [1]
        block_width_b = block_size[:, 0, 1] # [B]
        
        # ! block: parallel, slot: have_space is decided here
        does_it_have_existing = block_width_on_barge[range(self.batch_size), yard_selection, cur_barge_num] > 0
        is_allocable_mask = block_width_on_barge[range(self.batch_size), yard_selection, cur_barge_num] == 10.5#  # [B R]: 배치별로 선택된 야드의 현재 바지선에 중에서 병렬이 가능한 슬롯에 대한 진위 여부
        already_have_parallel = barge_slot[range(self.batch_size), yard_selection, cur_barge_num, :, 1] == -1  # [B R]: 병렬 열이 비어있는 경우
        is_allocable_and_have_existing_mask = torch.logical_and(torch.logical_and(is_allocable_mask, does_it_have_existing), already_have_parallel) 
        is_allocable_and_have_existing = torch.logical_and(torch.any(is_allocable_and_have_existing_mask, dim=-1), block_width_b == 10.5)
        # print(is_allocable_and_have_existing.shape)
        # quit()
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
    
    def make_to_choice(self, idx, agent_output, encoder_mask):
        # print(agent_output.shape)
        agent_output[:, :self.n_yard*self.n_take_out][encoder_mask.reshape(self.batch_size, -1)] = -10000 # *(기존) 보내고자 하는 블록이 적치장에 안 맞으면 마스킹, 제약 조건을 어겨도 적치장에 보낼 수는 있도록
        if len(self.priori_index)  != 0:
            agent_output[:, self.priori_index] = -15000 # !(추가) 특정 블록을 해당 지역에는 보내면 안 되므로 값을 제한
        if self.n_trip_infos_list is not None:
            agent_output[self.mask_wrong_barge_selection.transpose(1,2).reshape(self.batch_size, -1)] = -30000 # !(추가) 최대 항차 수가 넘은 경우 제한 
        agent_output[:, :self.n_yard*self.n_take_out][self.block_mask_to.repeat(1, 1, self.n_yard).reshape(self.batch_size, -1)] = -30000 # *(기존) 선택된 블록은 선택 안 되도록
        agent_output[:, self.n_yard*self.n_take_out:] = -30000 # *(기존) 고려하고자 하는 외의 Padding 부분은 마스킹
        
        self.feasible_mask[:, :self.n_yard*self.n_take_out][self.block_mask_to.repeat(1, 1, self.n_yard).reshape(self.batch_size, -1)] = False
        self.feasible_mask[:, :self.n_yard*self.n_take_out][encoder_mask.reshape(self.batch_size, -1)] = False
        if len(self.priori_index)  != 0:
            self.feasible_mask[:, self.priori_index] = False # 특정 블록을 보내야 하므로 배정된 야드 외에 값들은 제한
        
        probs = self.soft_max(agent_output)
        m = Categorical(probs)
        action = m.sample() # [B]
        
        # ________________________________________________
        block_selection, yard_selection = torch.div(action, self.n_yard, rounding_mode='trunc').type(torch.int64), (action % self.n_yard)
        # print("\n", block_selection[0], "\n", agent_output.reshape(self.batch_size, self.n_take_out, self.n_yard)[0, 6], "\n\n\n")
        isFeasible = self.feasible_mask[range(self.batch_size), action]
        
        block_size = self.block_size_info_to[range(self.batch_size), block_selection].unsqueeze(1) # [B, 1, 5]
        slot_info = torch.clone(self.yard_slots[range(self.batch_size), yard_selection]) # [B, pad_len, 5]
        
        # print(self.yard_slots[0, 2])

        ####
        
        embedded_block = self.block_size_emb(block_size[:, 0, [0,1]])
        # print(embedded_block.shape)

        embedded_slot_pre = self.slot_size_emb(self.yard_slots[:, :, :, [0,1]])
        count_of_slots = self.yard_slots[:, :, :, -1].clone().unsqueeze(-1)
        embedded_slot = torch.sum(embedded_slot_pre * count_of_slots, dim=2).transpose(1,2) # [B, emb_dim, n_yard]
    
        yard_emb = self.yard_emb(torch.arange(self.n_yard).to(self.device)).unsqueeze(0).repeat(self.batch_size, 1, 1) # [B, n_yard, emd_dim]
        # print(yard_emb.shape, embedded_slot.shape)
        # quit()
        embedded_yard = torch.sum(torch.matmul(embedded_slot, yard_emb), dim=-1) # [B, emb_dim, emb_dim]
    
        next_input = self.activ(torch.cat([embedded_block, embedded_yard], dim=1))
        
        self.barge_count_to, self.barge_slot_to, self.block_lengths_on_barge_to, self.block_width_on_barge_to = \
            self.allocate_to_barge(block_selection=block_selection, yard_selection=yard_selection, block_size=block_size, 
                                    barge_infos=(self.barge_count_to, self.barge_slot_to, self.block_lengths_on_barge_to, self.block_width_on_barge_to))
            
        # ______________________      Masking      __________________________
        
        #### Maximize
        slot_info[slot_info[:, :, 0] == 0] = 100 # 가능한 슬롯이 없는 경우에 값 최대화
        slot_info[torch.any(block_size[:, :, 0:3] > slot_info[:, :, 0:3], dim=-1)] = 100 # 크기가 큰 경우는 선택이 안 되도록 최대화 
        # print(block_size[0], " / ", yard_selection[0], " / ", isFeasible[0])
        # print(slot_info[0])
        # print(self.yard_slots[0, yard_selection[0]])
        # print("\n\n\n")
        yard_offset = torch.argmax((block_size[:, :, 0] * block_size[:, :, 1]) - (slot_info[:, :, 0] * slot_info[:, :, 1]), dim=-1) # [B]
        
        self.yard_slots[range(self.batch_size), yard_selection, yard_offset, -1] = self.yard_slots[range(self.batch_size), yard_selection, yard_offset, -1] - 1 # 선택 됐으면 감소
        slot_size = torch.clone(self.yard_slots[range(self.batch_size), yard_selection, yard_offset, 0:4]) # 선택된 슬롯의 크기 불러오기
        left_space_after_block = (slot_size[:, 0] * slot_size[:, 1] - (block_size[:, :, 0] * block_size[:, :, 1]).reshape(-1)) ** 2 * (torch.where(isFeasible, 1.0, -1.0))  # 불가능한 선택이면 Reward 최소화
        
        whole_slots = self.yard_slots[:, :, :, :3].unsqueeze(1).repeat(1, self.n_take_out, 1, 1, 1)
        whole_block_size = self.block_size_info_to[:, :,:3].unsqueeze(-2).unsqueeze(-2)
        # print(whole_block_size.shape)
        # print(whole_slots.shape)
        # 적당한 slot이 하나라도 있는 경우 True로 Compare between: (B, n_take_out, 1, 1, 3) <= (B, n_take_out, n_yard, pad_len, 3)
        new_encoder_mask = torch.all(whole_block_size <= whole_slots, dim=-1)
        # new_encoder_mask = torch.any(torch.all(whole_block_size <= whole_slots, dim=-1)
        # 크기 비교 후 카운트를 비교해야 하는데, 만약 카운트 값이 0인 경우는 제외 
        # print((self.yard_slots[:, :, :, -1] == 0).shape)
        new_encoder_mask[(self.yard_slots[:, :, :, -1] == 0).unsqueeze(1).repeat(1, self.n_take_out, 1, 1).to(self.device)] = False
        new_encoder_mask = ~torch.any(new_encoder_mask, dim=-1)
        # print(new_encoder_mask[0, 6])
        # print(new_encoder_mask.shape)
        # quit()
        
        self.yard_slots[self.yard_slots[:, :, :, -1] == 0] = 0 # 해당 크기의 슬롯이 남아있지 않으면 선택이 안 되도록 제거 
        self.yard_slots, _ = self.yard_slots.sort(dim=2, descending=True)
        
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
        
        return block_selection, next_input, new_encoder_mask
    
    def make_ti_choice(self, agent_output):
        
        agent_output[:, :self.n_take_in][self.block_mask_ti] = -20000
        agent_output[:, self.n_take_in:] = -30000
        
        probs = self.soft_max(agent_output)
        m = Categorical(probs)
        action = m.sample()
        
        block_size = self.block_size_info_ti[range(self.batch_size), action]
        
        yard = block_size[:, -1] -1
        # print(torch.unique(yard))
        yard = yard.type(torch.LongTensor).to(self.device)
        # print(torch.sum(torch.isnan(yard)))
        # print(yard)
        slot_info = self.yard_slots[range(self.batch_size), yard] 
        
        embedded_block = self.block_size_emb(block_size[:, [0,1]])
        # print(embedded_block.shape)

        embedded_slot_pre = self.slot_size_emb(self.yard_slots[:, :, :, [0,1]])
        count_of_slots = self.yard_slots[:, :, :, -1].clone().unsqueeze(-1)
        embedded_slot = torch.sum(embedded_slot_pre * count_of_slots, dim=2).transpose(1,2) # [B, emb_dim, n_yard]
    
        yard_emb = self.yard_emb(torch.arange(self.n_yard).to(self.device)).unsqueeze(0).repeat(self.batch_size, 1, 1) # [B, n_yard, emd_dim]
        # print(yard_emb.shape, embedded_slot.shape)
        # quit()
        embedded_yard = torch.sum(torch.matmul(embedded_slot, yard_emb), dim=-1) # [B, emb_dim, emb_dim]
    
        next_input = self.activ(torch.cat([embedded_block, embedded_yard], dim=1))
        
        self.barge_count_ti, self.barge_slot_ti, self.block_lengths_on_barge_ti, self.block_width_on_barge_ti = \
            self.allocate_to_barge(block_selection=action, yard_selection=yard, block_size=block_size.unsqueeze(1), 
                                    barge_infos=(self.barge_count_ti, self.barge_slot_ti, self.block_lengths_on_barge_ti, self.block_width_on_barge_ti))
        
        barge_with_blocks = torch.any(torch.any(self.barge_slot_ti != -1, dim=-1), dim=-1)
        barge_num_by_yard = torch.sum(barge_with_blocks, dim=-1) # [B Y]
        total_barge_num = torch.sum(barge_num_by_yard, dim=-1) # [B]
        
        if self.n_trip_infos_list is not None:
            mask_wrong_barge_selection = torch.any(barge_num_by_yard >= self.n_trip_infos_tensor, dim=-1)
            penalty_for_wrong_barge_selection = (block_size[:, 0] * block_size[:, 1]).reshape(-1) * (torch.where(mask_wrong_barge_selection, -1, 0))
        
        obj = self.beta * total_barge_num
        if self.n_trip_infos_list is not None:
            obj = obj + penalty_for_wrong_barge_selection
        
        self.probs_history_ti.append(m.log_prob(action))
        self.action_history_ti.append(action)
        self.rewards_history_ti.append(obj)
        
        return action, next_input
    
    def forward(self, infos: List[pd.DataFrame], encoder_inputs : torch.Tensor = None, remaining_area = None):
        """
        Feature information
        take_out, take_in, feat_vec: [LENGTH, WIDTH, HEIGHT, WEIGHT, LOCATION]
        slot_info: [length  width  height  location  count]
        """
        next_input, h0, c0, encoder_mask = self.init_env_info(infos)
        
        for idx in range(self.n_take_in):
            
            if self.n_trip_infos_list is not None:
                barge_with_blocks = torch.any(torch.any(self.barge_slot_ti != -1, dim=-1), dim=-1) 
                barge_num_by_yard = torch.sum(barge_with_blocks, dim=-1) # [B Y]
                wrong_barge_selection = barge_num_by_yard >= self.n_trip_infos_tensor # [B Y]
                
            # h0, c0 = self.ti_decoder(next_input, (h0, c0))
            h0 = self.ti_decoder(next_input, h0)
            out = self.ti_fc_block(self.activ(h0))
            # out = self.att(h0)
            action, next_input = self.make_ti_choice(out)
            
            # Gradient Error 때문에 매번 새로운 객체 형성 후 대체
            new_block_mask = self.block_mask_ti.clone()
            new_block_mask[range(self.batch_size), action] = True
            self.block_mask_ti = new_block_mask

        self.probs_history_ti = torch.stack(self.probs_history_ti).transpose(1, 0).to(self.device)
        self.rewards_history_ti = torch.stack(self.rewards_history_ti).transpose(1, 0).to(self.device)
        self.action_history_ti = torch.stack(self.action_history_ti).transpose(1, 0).to(self.device)
        
        # print(self.block_size_info_to.shape)
        encoder_output, h0 = self.to_encoder(self.block_size_info_to, torch.zeros((1, self.hid_dim, self.hid_dim)).to(self.device))
        h0 = h0.squeeze(0)

        for idx in range(self.n_take_out):
            # ________________________________________________
            
            if self.n_trip_infos_list is not None:
                barge_with_blocks = torch.any(torch.any(self.barge_slot_to != -1, dim=-1), dim=-1) 
                barge_num_by_yard = torch.sum(barge_with_blocks, dim=-1) # [B Y]
                wrong_barge_selection = barge_num_by_yard > self.n_trip_infos_tensor # [B Y]
                self.mask_wrong_barge_selection[wrong_barge_selection] = True
            
            # next_input = torch.clone(encoder_mask).type(torch.float32).reshape(self.batch_size, -1)
            # encoder_mask = ~torch.any(encoder_mask, dim=-1)
            # h0, c0 = self.to_decoder(next_input, (h0, c0))
            h0 = self.to_decoder(next_input, h0)
            # h0, c0 = self.to_decoder(encoder_mask.reshape(self.batch_size, -1), (h0, c0))
            # out = self.to_fc_block(self.activ(h0))
            out = self.att(h0, encoder_output)
            block_selection, next_input, encoder_mask = self.make_to_choice(idx, out, encoder_mask)
            # ________________________________________________
            
            # Gradient Error 때문에 매번 새로운 객체 형성 후 대체
            new_block_mask = self.block_mask_to.clone()
            new_block_mask[range(self.batch_size), block_selection] = True
            self.block_mask_to = new_block_mask
        
        self.probs_history_to = torch.stack(self.probs_history_to).transpose(1, 0).to(self.device)
        self.rewards_history_to = torch.stack(self.rewards_history_to).transpose(1, 0).to(self.device)
        self.action_history_to = torch.stack(self.action_history_to).transpose(1, 0).to(self.device)

        # print(self.action_history_to)
        # quit()
        
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
        

class PGAgent(nn.Module):
    def __init__(self, args) -> None:
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
        

        # self.to_decoder = nn.LSTMCell(input_size=self.emb_dim * (self.emb_dim+1), hidden_size=args.hid_dim)
        self.to_encoder = nn.GRU(input_size=5, hidden_size=args.hid_dim, batch_first=True)
        self.activ = nn.ReLU()
        self.to_decoder = nn.GRUCell(input_size=self.emb_dim * 2, hidden_size=args.hid_dim)
        # self.att = Attention(args.hid_dim, self.n_take_out, self.n_yard * self.n_max_block)
        # self.to_decoder = nn.LSTMCell(input_size=self.n_take_out * self.pad_len * self.n_yard, hidden_size=args.hid_dim)
        self.to_fc_block = nn.Linear(args.hid_dim, self.n_max_block*self.n_yard) # 400 = 40(n_block) * 10(n_yard)
        
        # self.ti_decoder = nn.LSTMCell(input_size=self.emb_dim * (self.emb_dim+1), hidden_size=args.hid_dim)
        self.ti_decoder = nn.GRUCell(input_size=self.emb_dim * 2, hidden_size=args.hid_dim)
        self.ti_fc_block = nn.Linear(args.hid_dim, self.n_max_block*self.n_yard)

        self.yard_emb = nn.Embedding(self.n_yard, self.emb_dim)
        self.block_size_emb = nn.Linear(2, self.emb_dim)
        self.slot_size_emb = nn.Linear(2, self.emb_dim)
        self.length_emb = nn.Linear(1, self.emb_dim)
        self.width_emb = nn.Linear(1, self.emb_dim)
        
        self.device = args.device
        # TODO: 제원 정보 임베딩으로 변환하는 방법 적용 고려해보기
        
        self.soft_max = nn.Softmax(dim=-1)
    
    def init_env_info(self, infos):
        # 위에서 Init을 안 하는 이유는 이건 Env.reset과 함께 다시 초기화되어야 하는 값이기 때문
        take_in, take_out, yard_slots = infos
        
        input0 = torch.zeros(self.batch_size, self.emb_dim*2).to(self.device)
        # input0 = torch.zeros(self.batch_size, 5*(self.pad_len+1)).to(self.device)
        h0, c0 = torch.zeros((self.batch_size, self.hid_dim), device=self.device), torch.zeros((self.batch_size, self.hid_dim), device=self.device)
        
        if self.n_trip_infos_list is not None:
            self.n_trip_infos_tensor = torch.FloatTensor(self.n_trip_infos_list).to(self.device).repeat(self.batch_size, 1)
        yard_slots = np.expand_dims(yard_slots, axis=0).repeat(self.batch_size, axis=0)
        self.yard_slots = torch.FloatTensor(yard_slots).to(self.device)
        
        
        # _______________________________ Take out _______________________________
        self.block_size_info_to = torch.FloatTensor(np.expand_dims(take_out[[LENGTH, WIDTH, HEIGHT, WEIGHT, LOCATION]].values, axis=0).repeat(self.batch_size, axis=0)).to(self.device)
        encoder_mask = self.generate_mask(take_out, yard_slots) # [B n_take_out n_yard n_pad]
        self.block_mask_to = torch.zeros((self.batch_size, self.n_take_out, 1), dtype=bool) # 제외 시키고 싶은 경우를 True
        self.feasible_mask = torch.ones((self.batch_size, self.n_max_block*self.n_yard), dtype=torch.bool, device=self.device)
        self.feasible_mask[:, self.n_yard*self.n_take_out:] = False
        
        self.barge_count_to = torch.zeros((self.batch_size, self.n_yard), dtype=torch.int64, device=self.device)
        self.barge_slot_to = -1 * torch.ones((self.batch_size, self.n_yard, self.max_trip, self.barge_max_row, 2), dtype=torch.int64, device=self.device) # 블록 인덱스 저장
        self.block_lengths_on_barge_to = torch.zeros((self.batch_size, self.n_yard, self.max_trip, self.barge_max_row), dtype=torch.float32, device=self.device) # 블록 길이 저장
        self.block_width_on_barge_to = torch.zeros((self.batch_size, self.n_yard, self.max_trip, self.barge_max_row), dtype=torch.float32, device=self.device) # 블록 너비 저장
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
        self.block_size_info_ti = torch.FloatTensor(np.expand_dims(take_in[[LENGTH, WIDTH, HEIGHT, WEIGHT, LOCATION]].values, axis=0).repeat(self.batch_size, axis=0)).to(self.device)
        self.block_mask_ti = torch.zeros((self.batch_size, self.n_take_in), dtype=bool) # 제외 시키고 싶은 경우를 True
        
        self.barge_count_ti = torch.zeros((self.batch_size, self.n_yard), dtype=torch.int64, device=self.device)
        self.barge_slot_ti = -1 * torch.ones((self.batch_size, self.n_yard, self.max_trip, self.barge_max_row, 2), dtype=torch.int64, device=self.device) # 블록 인덱스 저장
        self.block_lengths_on_barge_ti = torch.zeros((self.batch_size, self.n_yard, self.max_trip, self.barge_max_row), dtype=torch.float32, device=self.device) # 블록 길이 저장
        self.block_width_on_barge_ti = torch.zeros((self.batch_size, self.n_yard, self.max_trip, self.barge_max_row), dtype=torch.float32, device=self.device) # 블록 너비 저장
        
        self.probs_history_ti = []
        self.rewards_history_ti = []
        self.action_history_ti = []
        
        return input0, h0, c0, encoder_mask
        
    def generate_mask(self, take_out: pd.DataFrame, yard_slots: np.ndarray):
        # encoder_mask = np.zeros((self.batch_size, take_out.shape[0], self.n_yard, self.pad_len), dtype=bool)
        encoder_mask = np.zeros((self.batch_size, take_out.shape[0], self.n_yard, self.pad_len), dtype=float)
        
        for b_idx, block_info in enumerate(take_out[[LENGTH, WIDTH, HEIGHT]].values):
            mask = np.all(block_info <= yard_slots[:, :, :, :3], axis=-1)# 블록 크기가 슬롯의 모든 면보다 작으면 True, 블록이 슬롯보다 큰게 하나라도 있다면 FalseZZ
            encoder_mask[:, b_idx] = mask # 
        
        # 불가능한 경우에 값들을 변경해야 하므로 위에서 구한 가능한 Case 들을 False가 되도록 Not 적용
        return ~torch.any(torch.BoolTensor(encoder_mask).to(self.device), dim=-1)
        # return torch.BoolTensor(encoder_mask).to(self.device)
        # return torch.FloatTensor(encoder_mask).to(self.device)
    
    
    def allocate_to_barge(self, block_selection, block_size, yard_selection, barge_infos):
        # TODO: 바지선에 병렬 제약을 10이 아니라 합이 23 이하가 되도록
        barge_count ,barge_slot, block_lengths_on_barge, block_width_on_barge = barge_infos
        
        cur_barge_num = barge_count[range(self.batch_size), yard_selection] # [B]
    
        block_length_b = block_size[:, 0, 0] # Unsqueeze in [1]
        block_width_b = block_size[:, 0, 1] # [B]
        
        # ! block: parallel, slot: have_space is decided here
        does_it_have_existing = block_width_on_barge[range(self.batch_size), yard_selection, cur_barge_num] > 0
        is_allocable_mask = block_width_on_barge[range(self.batch_size), yard_selection, cur_barge_num] == 10.5#  # [B R]: 배치별로 선택된 야드의 현재 바지선에 중에서 병렬이 가능한 슬롯에 대한 진위 여부
        already_have_parallel = barge_slot[range(self.batch_size), yard_selection, cur_barge_num, :, 1] == -1  # [B R]: 병렬 열이 비어있는 경우
        is_allocable_and_have_existing_mask = torch.logical_and(torch.logical_and(is_allocable_mask, does_it_have_existing), already_have_parallel) 
        is_allocable_and_have_existing = torch.logical_and(torch.any(is_allocable_and_have_existing_mask, dim=-1), block_width_b == 10.5)
        # print(is_allocable_and_have_existing.shape)
        # quit()
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
    
    def make_to_choice(self, idx, agent_output, encoder_mask):
        # print(agent_output.shape)
        agent_output[:, :self.n_yard*self.n_take_out][encoder_mask.reshape(self.batch_size, -1)] = -10000 # *(기존) 보내고자 하는 블록이 적치장에 안 맞으면 마스킹, 제약 조건을 어겨도 적치장에 보낼 수는 있도록
        if len(self.priori_index)  != 0:
            agent_output[:, self.priori_index] = -15000 # !(추가) 특정 블록을 해당 지역에는 보내면 안 되므로 값을 제한
        if self.n_trip_infos_list is not None:
            agent_output[self.mask_wrong_barge_selection.transpose(1,2).reshape(self.batch_size, -1)] = -30000 # !(추가) 최대 항차 수가 넘은 경우 제한 
        agent_output[:, :self.n_yard*self.n_take_out][self.block_mask_to.repeat(1, 1, self.n_yard).reshape(self.batch_size, -1)] = -30000 # *(기존) 선택된 블록은 선택 안 되도록
        agent_output[:, self.n_yard*self.n_take_out:] = -30000 # *(기존) 고려하고자 하는 외의 Padding 부분은 마스킹
        
        self.feasible_mask[:, :self.n_yard*self.n_take_out][self.block_mask_to.repeat(1, 1, self.n_yard).reshape(self.batch_size, -1)] = False
        self.feasible_mask[:, :self.n_yard*self.n_take_out][encoder_mask.reshape(self.batch_size, -1)] = False
        if len(self.priori_index)  != 0:
            self.feasible_mask[:, self.priori_index] = False # 특정 블록을 보내야 하므로 배정된 야드 외에 값들은 제한
        
        probs = self.soft_max(agent_output)
        m = Categorical(probs)
        action = m.sample() # [B]
        
        # ________________________________________________
        block_selection, yard_selection = torch.div(action, self.n_yard, rounding_mode='trunc').type(torch.int64), (action % self.n_yard)
        # print("\n", block_selection[0], "\n", agent_output.reshape(self.batch_size, self.n_take_out, self.n_yard)[0, 6], "\n\n\n")
        isFeasible = self.feasible_mask[range(self.batch_size), action]
        
        block_size = self.block_size_info_to[range(self.batch_size), block_selection].unsqueeze(1) # [B, 1, 5]
        slot_info = torch.clone(self.yard_slots[range(self.batch_size), yard_selection]) # [B, pad_len, 5]
        
        # print(self.yard_slots[0, 2])

        ####
        embedded_block = self.block_size_emb(block_size[:, 0, [0,1]])
        # print(embedded_block.shape)

        embedded_slot_pre = self.slot_size_emb(self.yard_slots[:, :, :, [0,1]])
        count_of_slots = self.yard_slots[:, :, :, -1].clone().unsqueeze(-1)
        embedded_slot = torch.sum(embedded_slot_pre * count_of_slots, dim=2).transpose(1,2) # [B, emb_dim, n_yard]
    
        yard_emb = self.yard_emb(torch.arange(self.n_yard).to(self.device)).unsqueeze(0).repeat(self.batch_size, 1, 1) # [B, n_yard, emd_dim]
        # print(yard_emb.shape, embedded_slot.shape)
        # quit()
        embedded_yard = torch.sum(torch.matmul(embedded_slot, yard_emb), dim=-1) # [B, emb_dim, emb_dim]
    
        next_input = self.activ(torch.cat([embedded_block, embedded_yard], dim=1))
        
        self.barge_count_to, self.barge_slot_to, self.block_lengths_on_barge_to, self.block_width_on_barge_to = \
            self.allocate_to_barge(block_selection=block_selection, yard_selection=yard_selection, block_size=block_size, 
                                    barge_infos=(self.barge_count_to, self.barge_slot_to, self.block_lengths_on_barge_to, self.block_width_on_barge_to))
            
        # ______________________      Masking      __________________________
        
        #### Maximize
        slot_info[slot_info[:, :, 0] == 0] = 100 # 가능한 슬롯이 없는 경우에 값 최대화
        slot_info[torch.any(block_size[:, :, 0:3] > slot_info[:, :, 0:3], dim=-1)] = 100 # 크기가 큰 경우는 선택이 안 되도록 최대화 
        # print(block_size[0], " / ", yard_selection[0], " / ", isFeasible[0])
        # print(slot_info[0])
        # print(self.yard_slots[0, yard_selection[0]])
        # print("\n\n\n")
        yard_offset = torch.argmax((block_size[:, :, 0] * block_size[:, :, 1]) - (slot_info[:, :, 0] * slot_info[:, :, 1]), dim=-1) # [B]
        
        self.yard_slots[range(self.batch_size), yard_selection, yard_offset, -1] = self.yard_slots[range(self.batch_size), yard_selection, yard_offset, -1] - 1 # 선택 됐으면 감소
        slot_size = torch.clone(self.yard_slots[range(self.batch_size), yard_selection, yard_offset, 0:4]) # 선택된 슬롯의 크기 불러오기
        left_space_after_block = (slot_size[:, 0] * slot_size[:, 1] - (block_size[:, :, 0] * block_size[:, :, 1]).reshape(-1)) ** 2 * (torch.where(isFeasible, 1.0, -1.0))  # 불가능한 선택이면 Reward 최소화
        
        whole_slots = self.yard_slots[:, :, :, :3].unsqueeze(1).repeat(1, self.n_take_out, 1, 1, 1)
        whole_block_size = self.block_size_info_to[:, :,:3].unsqueeze(-2).unsqueeze(-2)
        # print(whole_block_size.shape)
        # print(whole_slots.shape)
        # 적당한 slot이 하나라도 있는 경우 True로 Compare between: (B, n_take_out, 1, 1, 3) <= (B, n_take_out, n_yard, pad_len, 3)
        new_encoder_mask = torch.all(whole_block_size <= whole_slots, dim=-1)
        # new_encoder_mask = torch.any(torch.all(whole_block_size <= whole_slots, dim=-1)
        # 크기 비교 후 카운트를 비교해야 하는데, 만약 카운트 값이 0인 경우는 제외 
        # print((self.yard_slots[:, :, :, -1] == 0).shape)
        new_encoder_mask[(self.yard_slots[:, :, :, -1] == 0).unsqueeze(1).repeat(1, self.n_take_out, 1, 1).to(self.device)] = False
        new_encoder_mask = ~torch.any(new_encoder_mask, dim=-1)
        # print(new_encoder_mask[0, 6])
        # print(new_encoder_mask.shape)
        # quit()
        
        self.yard_slots[self.yard_slots[:, :, :, -1] == 0] = 0 # 해당 크기의 슬롯이 남아있지 않으면 선택이 안 되도록 제거 
        self.yard_slots, _ = self.yard_slots.sort(dim=2, descending=True)
        
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
        
        return block_selection, next_input, new_encoder_mask
    
    def make_ti_choice(self, agent_output):
        
        agent_output[:, :self.n_take_in][self.block_mask_ti] = -20000
        agent_output[:, self.n_take_in:] = -30000
        
        probs = self.soft_max(agent_output)
        m = Categorical(probs)
        action = m.sample()
        
        block_size = self.block_size_info_ti[range(self.batch_size), action]
        
        yard = block_size[:, -1] -1
        # print(torch.unique(yard))
        yard = yard.type(torch.LongTensor).to(self.device)
        # print(torch.sum(torch.isnan(yard)))
        # print(yard)
        slot_info = self.yard_slots[range(self.batch_size), yard] 
        
        embedded_block = self.block_size_emb(block_size[:, [0,1]])
        # print(embedded_block.shape)

        embedded_slot_pre = self.slot_size_emb(self.yard_slots[:, :, :, [0,1]])
        count_of_slots = self.yard_slots[:, :, :, -1].clone().unsqueeze(-1)
        embedded_slot = torch.sum(embedded_slot_pre * count_of_slots, dim=2).transpose(1,2) # [B, emb_dim, n_yard]
    
        yard_emb = self.yard_emb(torch.arange(self.n_yard).to(self.device)).unsqueeze(0).repeat(self.batch_size, 1, 1) # [B, n_yard, emd_dim]
        # print(yard_emb.shape, embedded_slot.shape)
        # quit()
        embedded_yard = torch.sum(torch.matmul(embedded_slot, yard_emb), dim=-1) # [B, emb_dim, emb_dim]
    
        next_input = self.activ(torch.cat([embedded_block, embedded_yard], dim=1))
        
        self.barge_count_ti, self.barge_slot_ti, self.block_lengths_on_barge_ti, self.block_width_on_barge_ti = \
            self.allocate_to_barge(block_selection=action, yard_selection=yard, block_size=block_size.unsqueeze(1), 
                                    barge_infos=(self.barge_count_ti, self.barge_slot_ti, self.block_lengths_on_barge_ti, self.block_width_on_barge_ti))
        
        barge_with_blocks = torch.any(torch.any(self.barge_slot_ti != -1, dim=-1), dim=-1)
        barge_num_by_yard = torch.sum(barge_with_blocks, dim=-1) # [B Y]
        total_barge_num = torch.sum(barge_num_by_yard, dim=-1) # [B]
        
        if self.n_trip_infos_list is not None:
            mask_wrong_barge_selection = torch.any(barge_num_by_yard >= self.n_trip_infos_tensor, dim=-1)
            penalty_for_wrong_barge_selection = (block_size[:, 0] * block_size[:, 1]).reshape(-1) * (torch.where(mask_wrong_barge_selection, -1, 0))
        
        obj = self.beta * total_barge_num
        if self.n_trip_infos_list is not None:
            obj = obj + penalty_for_wrong_barge_selection
        
        self.probs_history_ti.append(m.log_prob(action))
        self.action_history_ti.append(action)
        self.rewards_history_ti.append(obj)
        
        return action, next_input
    
    def forward(self, infos: List[pd.DataFrame], encoder_inputs : torch.Tensor = None, remaining_area = None):
        """
        Feature information
        take_out, take_in, feat_vec: [LENGTH, WIDTH, HEIGHT, WEIGHT, LOCATION]
        slot_info: [length  width  height  location  count]
        """
        next_input, h0, c0, encoder_mask = self.init_env_info(infos)
        
        for idx in range(self.n_take_in):
            
            if self.n_trip_infos_list is not None:
                barge_with_blocks = torch.any(torch.any(self.barge_slot_ti != -1, dim=-1), dim=-1) 
                barge_num_by_yard = torch.sum(barge_with_blocks, dim=-1) # [B Y]
                wrong_barge_selection = barge_num_by_yard >= self.n_trip_infos_tensor # [B Y]
                
            # h0, c0 = self.ti_decoder(next_input, (h0, c0))
            h0 = self.ti_decoder(next_input, h0)
            out = self.ti_fc_block(self.activ(h0))
            # out = self.att(h0)
            action, next_input = self.make_ti_choice(out)
            
            # Gradient Error 때문에 매번 새로운 객체 형성 후 대체
            new_block_mask = self.block_mask_ti.clone()
            new_block_mask[range(self.batch_size), action] = True
            self.block_mask_ti = new_block_mask

        self.probs_history_ti = torch.stack(self.probs_history_ti).transpose(1, 0).to(self.device)
        self.rewards_history_ti = torch.stack(self.rewards_history_ti).transpose(1, 0).to(self.device)
        self.action_history_ti = torch.stack(self.action_history_ti).transpose(1, 0).to(self.device)
        
        # print(self.block_size_info_to.shape)
        # encoder_output, h0 = self.to_encoder(self.block_size_info_to, torch.zeros((1, self.hid_dim, self.hid_dim)).to(self.device))
        # h0 = h0.squeeze(0)

        for idx in range(self.n_take_out):
            # ________________________________________________
            
            if self.n_trip_infos_list is not None:
                barge_with_blocks = torch.any(torch.any(self.barge_slot_to != -1, dim=-1), dim=-1) 
                barge_num_by_yard = torch.sum(barge_with_blocks, dim=-1) # [B Y]
                wrong_barge_selection = barge_num_by_yard > self.n_trip_infos_tensor # [B Y]
                self.mask_wrong_barge_selection[wrong_barge_selection] = True
            
            # next_input = torch.clone(encoder_mask).type(torch.float32).reshape(self.batch_size, -1)
            # encoder_mask = ~torch.any(encoder_mask, dim=-1)
            # h0, c0 = self.to_decoder(next_input, (h0, c0))
            h0 = self.to_decoder(next_input, h0)
            # h0, c0 = self.to_decoder(encoder_mask.reshape(self.batch_size, -1), (h0, c0))
            out = self.to_fc_block(self.activ(h0))
            # out = self.att(h0, encoder_output)
            block_selection, next_input, encoder_mask = self.make_to_choice(idx, out, encoder_mask)
            # ________________________________________________
            
            # Gradient Error 때문에 매번 새로운 객체 형성 후 대체
            new_block_mask = self.block_mask_to.clone()
            new_block_mask[range(self.batch_size), block_selection] = True
            self.block_mask_to = new_block_mask
        
        self.probs_history_to = torch.stack(self.probs_history_to).transpose(1, 0).to(self.device)
        self.rewards_history_to = torch.stack(self.rewards_history_to).transpose(1, 0).to(self.device)
        self.action_history_to = torch.stack(self.action_history_to).transpose(1, 0).to(self.device)

        # print(self.action_history_to)
        # quit()
        
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
        

class PGAgent_NoEmb(nn.Module):
    def __init__(self, args) -> None:
        super(PGAgent_NoEmb, self).__init__()
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
        

        # self.to_decoder = nn.LSTMCell(input_size=self.emb_dim * (self.emb_dim+1), hidden_size=args.hid_dim)
        # self.to_encoder = nn.GRU(input_size=5, hidden_size=args.hid_dim, batch_first=True)
        self.activ = nn.ReLU()
        self.to_decoder = nn.GRUCell(input_size=2+self.pad_len*5, hidden_size=args.hid_dim)
        # self.att = Attention(args.hid_dim, self.n_take_out, self.n_yard * self.n_max_block)
        # self.to_decoder = nn.LSTMCell(input_size=self.n_take_out * self.pad_len * self.n_yard, hidden_size=args.hid_dim)
        self.to_fc_block = nn.Linear(args.hid_dim, self.n_max_block*self.n_yard) # 400 = 40(n_block) * 10(n_yard)
        
        # self.ti_decoder = nn.LSTMCell(input_size=self.emb_dim * (self.emb_dim+1), hidden_size=args.hid_dim)
        self.ti_decoder = nn.GRUCell(input_size=2+self.pad_len*5, hidden_size=args.hid_dim)
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
        
        input0 = torch.zeros(self.batch_size, 2+self.pad_len*5).to(self.device)
        # input0 = torch.zeros(self.batch_size, 5*(self.pad_len+1)).to(self.device)
        h0, c0 = torch.zeros((self.batch_size, self.hid_dim), device=self.device), torch.zeros((self.batch_size, self.hid_dim), device=self.device)
        
        if self.n_trip_infos_list is not None:
            self.n_trip_infos_tensor = torch.FloatTensor(self.n_trip_infos_list).to(self.device).repeat(self.batch_size, 1)
        yard_slots = np.expand_dims(yard_slots, axis=0).repeat(self.batch_size, axis=0)
        self.yard_slots = torch.FloatTensor(yard_slots).to(self.device)
        
        
        # _______________________________ Take out _______________________________
        self.block_size_info_to = torch.FloatTensor(np.expand_dims(take_out[[LENGTH, WIDTH, HEIGHT, WEIGHT, LOCATION]].values, axis=0).repeat(self.batch_size, axis=0)).to(self.device)
        encoder_mask = self.generate_mask(take_out, yard_slots) # [B n_take_out n_yard n_pad]
        self.block_mask_to = torch.zeros((self.batch_size, self.n_take_out, 1), dtype=bool) # 제외 시키고 싶은 경우를 True
        self.feasible_mask = torch.ones((self.batch_size, self.n_max_block*self.n_yard), dtype=torch.bool, device=self.device)
        self.feasible_mask[:, self.n_yard*self.n_take_out:] = False
        
        self.barge_count_to = torch.zeros((self.batch_size, self.n_yard), dtype=torch.int64, device=self.device)
        self.barge_slot_to = -1 * torch.ones((self.batch_size, self.n_yard, self.max_trip, self.barge_max_row, 2), dtype=torch.int64, device=self.device) # 블록 인덱스 저장
        self.block_lengths_on_barge_to = torch.zeros((self.batch_size, self.n_yard, self.max_trip, self.barge_max_row), dtype=torch.float32, device=self.device) # 블록 길이 저장
        self.block_width_on_barge_to = torch.zeros((self.batch_size, self.n_yard, self.max_trip, self.barge_max_row), dtype=torch.float32, device=self.device) # 블록 너비 저장
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
        self.block_size_info_ti = torch.FloatTensor(np.expand_dims(take_in[[LENGTH, WIDTH, HEIGHT, WEIGHT, LOCATION]].values, axis=0).repeat(self.batch_size, axis=0)).to(self.device)
        self.block_mask_ti = torch.zeros((self.batch_size, self.n_take_in), dtype=bool) # 제외 시키고 싶은 경우를 True
        
        self.barge_count_ti = torch.zeros((self.batch_size, self.n_yard), dtype=torch.int64, device=self.device)
        self.barge_slot_ti = -1 * torch.ones((self.batch_size, self.n_yard, self.max_trip, self.barge_max_row, 2), dtype=torch.int64, device=self.device) # 블록 인덱스 저장
        self.block_lengths_on_barge_ti = torch.zeros((self.batch_size, self.n_yard, self.max_trip, self.barge_max_row), dtype=torch.float32, device=self.device) # 블록 길이 저장
        self.block_width_on_barge_ti = torch.zeros((self.batch_size, self.n_yard, self.max_trip, self.barge_max_row), dtype=torch.float32, device=self.device) # 블록 너비 저장
        
        self.probs_history_ti = []
        self.rewards_history_ti = []
        self.action_history_ti = []
        
        return input0, h0, c0, encoder_mask
        
    def generate_mask(self, take_out: pd.DataFrame, yard_slots: np.ndarray):
        # encoder_mask = np.zeros((self.batch_size, take_out.shape[0], self.n_yard, self.pad_len), dtype=bool)
        encoder_mask = np.zeros((self.batch_size, take_out.shape[0], self.n_yard, self.pad_len), dtype=float)
        
        for b_idx, block_info in enumerate(take_out[[LENGTH, WIDTH, HEIGHT]].values):
            mask = np.all(block_info <= yard_slots[:, :, :, :3], axis=-1)# 블록 크기가 슬롯의 모든 면보다 작으면 True, 블록이 슬롯보다 큰게 하나라도 있다면 FalseZZ
            encoder_mask[:, b_idx] = mask # 
        
        # 불가능한 경우에 값들을 변경해야 하므로 위에서 구한 가능한 Case 들을 False가 되도록 Not 적용
        return ~torch.any(torch.BoolTensor(encoder_mask).to(self.device), dim=-1)
        # return torch.BoolTensor(encoder_mask).to(self.device)
        # return torch.FloatTensor(encoder_mask).to(self.device)
    
    
    def allocate_to_barge(self, block_selection, block_size, yard_selection, barge_infos):
        # TODO: 바지선에 병렬 제약을 10이 아니라 합이 23 이하가 되도록
        barge_count ,barge_slot, block_lengths_on_barge, block_width_on_barge = barge_infos
        
        cur_barge_num = barge_count[range(self.batch_size), yard_selection] # [B]
    
        block_length_b = block_size[:, 0, 0] # Unsqueeze in [1]
        block_width_b = block_size[:, 0, 1] # [B]
        
        # ! block: parallel, slot: have_space is decided here
        does_it_have_existing = block_width_on_barge[range(self.batch_size), yard_selection, cur_barge_num] > 0
        is_allocable_mask = block_width_on_barge[range(self.batch_size), yard_selection, cur_barge_num] == 10.5#  # [B R]: 배치별로 선택된 야드의 현재 바지선에 중에서 병렬이 가능한 슬롯에 대한 진위 여부
        already_have_parallel = barge_slot[range(self.batch_size), yard_selection, cur_barge_num, :, 1] == -1  # [B R]: 병렬 열이 비어있는 경우
        is_allocable_and_have_existing_mask = torch.logical_and(torch.logical_and(is_allocable_mask, does_it_have_existing), already_have_parallel) 
        is_allocable_and_have_existing = torch.logical_and(torch.any(is_allocable_and_have_existing_mask, dim=-1), block_width_b == 10.5)
        # print(is_allocable_and_have_existing.shape)
        # quit()
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
    
    def make_to_choice(self, idx, agent_output, encoder_mask):
        # print(agent_output.shape)
        agent_output[:, :self.n_yard*self.n_take_out][encoder_mask.reshape(self.batch_size, -1)] = -10000 # *(기존) 보내고자 하는 블록이 적치장에 안 맞으면 마스킹, 제약 조건을 어겨도 적치장에 보낼 수는 있도록
        if len(self.priori_index)  != 0:
            agent_output[:, self.priori_index] = -15000 # !(추가) 특정 블록을 해당 지역에는 보내면 안 되므로 값을 제한
        if self.n_trip_infos_list is not None:
            agent_output[self.mask_wrong_barge_selection.transpose(1,2).reshape(self.batch_size, -1)] = -30000 # !(추가) 최대 항차 수가 넘은 경우 제한 
        agent_output[:, :self.n_yard*self.n_take_out][self.block_mask_to.repeat(1, 1, self.n_yard).reshape(self.batch_size, -1)] = -30000 # *(기존) 선택된 블록은 선택 안 되도록
        agent_output[:, self.n_yard*self.n_take_out:] = -30000 # *(기존) 고려하고자 하는 외의 Padding 부분은 마스킹
        
        self.feasible_mask[:, :self.n_yard*self.n_take_out][self.block_mask_to.repeat(1, 1, self.n_yard).reshape(self.batch_size, -1)] = False
        self.feasible_mask[:, :self.n_yard*self.n_take_out][encoder_mask.reshape(self.batch_size, -1)] = False
        if len(self.priori_index)  != 0:
            self.feasible_mask[:, self.priori_index] = False # 특정 블록을 보내야 하므로 배정된 야드 외에 값들은 제한
        
        probs = self.soft_max(agent_output)
        m = Categorical(probs)
        action = m.sample() # [B]
        
        # ________________________________________________
        block_selection, yard_selection = torch.div(action, self.n_yard, rounding_mode='trunc').type(torch.int64), (action % self.n_yard)
        # print("\n", block_selection[0], "\n", agent_output.reshape(self.batch_size, self.n_take_out, self.n_yard)[0, 6], "\n\n\n")
        isFeasible = self.feasible_mask[range(self.batch_size), action]
        
        block_size = self.block_size_info_to[range(self.batch_size), block_selection].unsqueeze(1) # [B, 1, 5]
        slot_info = torch.clone(self.yard_slots[range(self.batch_size), yard_selection]) # [B, pad_len, 5]
        
        # print(block_size[:, 0, [0,1]].shape, slot_info.shape)
        next_input = torch.cat([block_size[:, 0, [0,1]], slot_info.reshape(self.batch_size, -1)], dim=1)
        # print(self.yard_slots[0, 2])
    
        
        self.barge_count_to, self.barge_slot_to, self.block_lengths_on_barge_to, self.block_width_on_barge_to = \
            self.allocate_to_barge(block_selection=block_selection, yard_selection=yard_selection, block_size=block_size, 
                                    barge_infos=(self.barge_count_to, self.barge_slot_to, self.block_lengths_on_barge_to, self.block_width_on_barge_to))
            
        # ______________________      Masking      __________________________
        
        #### Maximize
        slot_info[slot_info[:, :, 0] == 0] = 100 # 가능한 슬롯이 없는 경우에 값 최대화
        slot_info[torch.any(block_size[:, :, 0:3] > slot_info[:, :, 0:3], dim=-1)] = 100 # 크기가 큰 경우는 선택이 안 되도록 최대화 
        # print(block_size[0], " / ", yard_selection[0], " / ", isFeasible[0])
        # print(slot_info[0])
        # print(self.yard_slots[0, yard_selection[0]])
        # print("\n\n\n")
        yard_offset = torch.argmax((block_size[:, :, 0] * block_size[:, :, 1]) - (slot_info[:, :, 0] * slot_info[:, :, 1]), dim=-1) # [B]
        
        self.yard_slots[range(self.batch_size), yard_selection, yard_offset, -1] = self.yard_slots[range(self.batch_size), yard_selection, yard_offset, -1] - 1 # 선택 됐으면 감소
        slot_size = torch.clone(self.yard_slots[range(self.batch_size), yard_selection, yard_offset, 0:4]) # 선택된 슬롯의 크기 불러오기
        left_space_after_block = (slot_size[:, 0] * slot_size[:, 1] - (block_size[:, :, 0] * block_size[:, :, 1]).reshape(-1)) ** 2 * (torch.where(isFeasible, 1.0, -1.0))  # 불가능한 선택이면 Reward 최소화
        
        whole_slots = self.yard_slots[:, :, :, :3].unsqueeze(1).repeat(1, self.n_take_out, 1, 1, 1)
        whole_block_size = self.block_size_info_to[:, :,:3].unsqueeze(-2).unsqueeze(-2)
        # print(whole_block_size.shape)
        # print(whole_slots.shape)
        # 적당한 slot이 하나라도 있는 경우 True로 Compare between: (B, n_take_out, 1, 1, 3) <= (B, n_take_out, n_yard, pad_len, 3)
        new_encoder_mask = torch.all(whole_block_size <= whole_slots, dim=-1)
        # new_encoder_mask = torch.any(torch.all(whole_block_size <= whole_slots, dim=-1)
        # 크기 비교 후 카운트를 비교해야 하는데, 만약 카운트 값이 0인 경우는 제외 
        # print((self.yard_slots[:, :, :, -1] == 0).shape)
        new_encoder_mask[(self.yard_slots[:, :, :, -1] == 0).unsqueeze(1).repeat(1, self.n_take_out, 1, 1).to(self.device)] = False
        new_encoder_mask = ~torch.any(new_encoder_mask, dim=-1)
        # print(new_encoder_mask[0, 6])
        # print(new_encoder_mask.shape)
        # quit()
        
        self.yard_slots[self.yard_slots[:, :, :, -1] == 0] = 0 # 해당 크기의 슬롯이 남아있지 않으면 선택이 안 되도록 제거 
        self.yard_slots, _ = self.yard_slots.sort(dim=2, descending=True)
        
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
        
        return block_selection, next_input, new_encoder_mask
    
    def make_ti_choice(self, agent_output):
        
        agent_output[:, :self.n_take_in][self.block_mask_ti] = -20000
        agent_output[:, self.n_take_in:] = -30000
        
        probs = self.soft_max(agent_output)
        m = Categorical(probs)
        action = m.sample()
        
        block_size = self.block_size_info_ti[range(self.batch_size), action]
        
        yard = block_size[:, -1] -1
        # print(torch.unique(yard))
        yard = yard.type(torch.LongTensor).to(self.device)
        # print(torch.sum(torch.isnan(yard)))
        # print(yard)
        slot_info = self.yard_slots[range(self.batch_size), yard] 
        
        # print(block_size[:, [0,1]].shape, slot_info.shape)
        next_input = torch.cat([block_size[:, [0,1]], slot_info.reshape(self.batch_size, -1)], dim=1)
        # print(next_input)
        # quit()
        
        self.barge_count_ti, self.barge_slot_ti, self.block_lengths_on_barge_ti, self.block_width_on_barge_ti = \
            self.allocate_to_barge(block_selection=action, yard_selection=yard, block_size=block_size.unsqueeze(1), 
                                    barge_infos=(self.barge_count_ti, self.barge_slot_ti, self.block_lengths_on_barge_ti, self.block_width_on_barge_ti))
        
        barge_with_blocks = torch.any(torch.any(self.barge_slot_ti != -1, dim=-1), dim=-1)
        barge_num_by_yard = torch.sum(barge_with_blocks, dim=-1) # [B Y]
        total_barge_num = torch.sum(barge_num_by_yard, dim=-1) # [B]
        
        if self.n_trip_infos_list is not None:
            mask_wrong_barge_selection = torch.any(barge_num_by_yard >= self.n_trip_infos_tensor, dim=-1)
            penalty_for_wrong_barge_selection = (block_size[:, 0] * block_size[:, 1]).reshape(-1) * (torch.where(mask_wrong_barge_selection, -1, 0))
        
        obj = self.beta * total_barge_num
        if self.n_trip_infos_list is not None:
            obj = obj + penalty_for_wrong_barge_selection
        
        self.probs_history_ti.append(m.log_prob(action))
        self.action_history_ti.append(action)
        self.rewards_history_ti.append(obj)
        
        return action, next_input
    
    def forward(self, infos: List[pd.DataFrame], encoder_inputs : torch.Tensor = None, remaining_area = None):
        """
        Feature information
        take_out, take_in, feat_vec: [LENGTH, WIDTH, HEIGHT, WEIGHT, LOCATION]
        slot_info: [length  width  height  location  count]
        """
        next_input, h0, c0, encoder_mask = self.init_env_info(infos)
        
        for idx in range(self.n_take_in):
            
            if self.n_trip_infos_list is not None:
                barge_with_blocks = torch.any(torch.any(self.barge_slot_ti != -1, dim=-1), dim=-1) 
                barge_num_by_yard = torch.sum(barge_with_blocks, dim=-1) # [B Y]
                wrong_barge_selection = barge_num_by_yard >= self.n_trip_infos_tensor # [B Y]
                
            # h0, c0 = self.ti_decoder(next_input, (h0, c0))
            h0 = self.ti_decoder(next_input, h0)
            out = self.ti_fc_block(self.activ(h0))
            # out = self.att(h0)
            action, next_input = self.make_ti_choice(out)
            
            # Gradient Error 때문에 매번 새로운 객체 형성 후 대체
            new_block_mask = self.block_mask_ti.clone()
            new_block_mask[range(self.batch_size), action] = True
            self.block_mask_ti = new_block_mask

        self.probs_history_ti = torch.stack(self.probs_history_ti).transpose(1, 0).to(self.device)
        self.rewards_history_ti = torch.stack(self.rewards_history_ti).transpose(1, 0).to(self.device)
        self.action_history_ti = torch.stack(self.action_history_ti).transpose(1, 0).to(self.device)
        
        # print(self.block_size_info_to.shape)
        # encoder_output, h0 = self.to_encoder(self.block_size_info_to, torch.zeros((1, self.hid_dim, self.hid_dim)).to(self.device))
        # h0 = h0.squeeze(0)

        for idx in range(self.n_take_out):
            # ________________________________________________
            
            if self.n_trip_infos_list is not None:
                barge_with_blocks = torch.any(torch.any(self.barge_slot_to != -1, dim=-1), dim=-1) 
                barge_num_by_yard = torch.sum(barge_with_blocks, dim=-1) # [B Y]
                wrong_barge_selection = barge_num_by_yard > self.n_trip_infos_tensor # [B Y]
                self.mask_wrong_barge_selection[wrong_barge_selection] = True
            
            # next_input = torch.clone(encoder_mask).type(torch.float32).reshape(self.batch_size, -1)
            # encoder_mask = ~torch.any(encoder_mask, dim=-1)
            # h0, c0 = self.to_decoder(next_input, (h0, c0))
            h0 = self.to_decoder(next_input, h0)
            # h0, c0 = self.to_decoder(encoder_mask.reshape(self.batch_size, -1), (h0, c0))
            out = self.to_fc_block(self.activ(h0))
            # out = self.att(h0, encoder_output)
            block_selection, next_input, encoder_mask = self.make_to_choice(idx, out, encoder_mask)
            # ________________________________________________
            
            # Gradient Error 때문에 매번 새로운 객체 형성 후 대체
            new_block_mask = self.block_mask_to.clone()
            new_block_mask[range(self.batch_size), block_selection] = True
            self.block_mask_to = new_block_mask
        
        self.probs_history_to = torch.stack(self.probs_history_to).transpose(1, 0).to(self.device)
        self.rewards_history_to = torch.stack(self.rewards_history_to).transpose(1, 0).to(self.device)
        self.action_history_to = torch.stack(self.action_history_to).transpose(1, 0).to(self.device)

        # print("Complete")
        # quit()
        
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
    def __init__(self, args, n_take_out=None) -> None:
        self.args = args
        self.yard_in = YardInside(name="WY", area_slots=None, blocks=None)
        # Set by random for now
        self.target_columns = [LENGTH, WIDTH, HEIGHT, WEIGHT, LOCATION]
        self.yards_out : Dict[str, "YardOutside"]= {}
        self.first_step_infos = {}
        # self.load_env()
        self.probs = []
        self.rewards = []
        self.max_result = dict().fromkeys(["take_out_probs", "take_out_rewards", "take_out_actions", "take_out_barges", "take_in_probs", "take_in_rewards", "take_in_actions", "take_in_barges"]) # Value of reward and combination of actions
        
        # TODO: Input 파일에 맞게 로딩되도록
        self.labels_encoder = dict(zip(DOCK_NAME, range(len(DOCK_NAME))))
        # print(self.labels_encoder)
        # print(self.labels_encoder)
        # quit()
        args.n_yard = len(self.labels_encoder) - 1
        # 만약 적치장 전체에 대한 정보를 얻을 수 있다면 해당 부분을 직접 입력하는 형태가 아니라 파일을 불러오는 방식으로 변경하면 될거 같습니다.
        # 모두 Batch 형태로 변환하여 병렬 연산으로 수행하기 위해섭니다. 
        self.labels_encoder_inv = dict(zip(self.labels_encoder.values(), self.labels_encoder.keys()))
        
        
    def get_init_state(self, possible_take_out: pd.DataFrame, take_in: pd.DataFrame):
        "Modify the 'block_mask_to' and 'yard_slot' according to initial choices"
        "+ barge should be allocated according to the rule"
        # Encoder에 Input으로 들어갈수록 
        possible_take_out[LOCATION] = possible_take_out[LOCATION].map(self.labels_encoder) # [n_b 3]
        take_out_block_size = torch.FloatTensor(possible_take_out[[LENGTH, WIDTH, HEIGHT]].values)
        take_out_block_size = take_out_block_size.unsqueeze(1).repeat(1, len(self.labels_encoder)-1, 1).unsqueeze(2)
        yard_slots = []
        
        for name in self.labels_encoder.keys():
            if name == "WY": continue
            yard = self.yards_out[name]
            if yard.area_slots is None: 
                count_ = np.zeros((self.args.pad_len, 5))
            else:
                state_part_ = yard.area_slots.copy()
                state_part_[NAME] = self.labels_encoder[name]
                state_part_.loc[state_part_[HEIGHT] == float("inf"), HEIGHT] = 1000
                count = state_part_[[LENGTH, WIDTH, HEIGHT, LOCATION]].groupby([LENGTH, WIDTH], as_index=False).value_counts() # [LENGTH, WIDTH, HEIGHT, LOCATION, COUNT]
                
                # if count.shape[0] == 0:                 # continue
                count_ = np.zeros((self.args.pad_len, 5))
                count_[:count.shape[0], :] = count
            # print(name, count)
            
            
            yard_slots.append(count_)
        # quit()    
        yard_slots = np.array(yard_slots)
        
        slot_size = torch.FloatTensor(yard_slots[:, :, 0:3])
        slot_size = slot_size.unsqueeze(0).repeat(len(possible_take_out), 1, 1, 1)
        
        # print(take_out_block_size.shape, slot_size.shape)
        is_allocable_mask = torch.any(torch.any(take_out_block_size <= slot_size , dim=-1), dim=-1).reshape(-1)
        alloc_idxs = is_allocable_mask.nonzero()
        
        # self.args.batch_size = torch.sum(is_allocable_mask)
        
        return alloc_idxs, yard_slots, None
            
    def step(self, infos: List[pd.DataFrame], inputs: torch.Tensor, remaining_areas: np.array):
        
        # TODO: Implement the block moving to the yard
        take_in, take_out, yard_slots = infos
        
        take_in[LOCATION] = take_in[LOCATION].map(self.labels_encoder).astype(float)
        take_out[LOCATION] = take_out[LOCATION].map(self.labels_encoder).astype(float)
        
        kwargs  = self.RLAgent((take_in.reset_index().copy(), take_out.reset_index(drop=True).copy(), yard_slots.copy()), inputs, remaining_areas)
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
        for idx in range(args.n_take_out-1, -1, -1):
            if idx == args.n_take_out-1:
                take_out_returns[:, 0] = take_out_rewards[:, idx] # 처음에는 그냥 리워드 값
            else:
                take_out_returns[:, args.n_take_out-1-idx] = (take_out_rewards[:, idx] + args.decay * take_out_returns[:, args.n_take_out-idx-2])
                
        norm_returns = (take_out_returns - take_out_returns.mean()) / (take_out_returns.std() + args.std_eps)
        # norm_returns = (take_out_returns - take_out_returns.mean(dim=-1).unsqueeze(-1)) / (take_out_returns.std(dim=-1).unsqueeze(-1) + args.std_eps)
        take_out_loss = torch.mul(-take_out_probs, norm_returns).sum(-1) # Maximize
        
        
        take_in_probs = self.result["take_in_probs"]
        take_in_rewards = self.result["take_in_rewards"].type(torch.float32)
        
        take_in_loss = torch.zeros((len(take_in_probs)))
        take_in_returns = torch.zeros_like(take_in_rewards)
        for idx in range(args.n_take_in-1, -1, -1):
            if idx == args.n_take_in-1:
                take_in_returns[:, 0] = take_in_rewards[:, idx] # 처음에는 그냥 리워드 값
            else:
                take_in_returns[:, args.n_take_in-1-idx] = (take_in_rewards[:, idx] + self.args.decay * take_in_returns[:, args.n_take_in-idx-2])
                
        norm_returns = (take_in_returns - take_in_returns.mean()) / (take_in_returns.std() + args.std_eps)
        # norm_returns = (take_in_returns - take_in_returns.mean(dim=-1).unsqueeze(-1)) / (take_in_returns.std(dim=-1).unsqueeze(-1) + args.std_eps)
        take_in_loss = torch.mul(-take_in_probs, norm_returns).sum(-1) # Maximize
        
        # total_loss = torch.mean(take_out_loss) + 0.1 * torch.mean(take_in_loss)
        # total_loss = torch.mean(take_out_loss) + torch.mean(take_in_loss)
        total_loss = torch.mean(take_out_loss)
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.RLAgent.parameters(), self.args.clipping_size)
        self.optimizer.step()
        self.scheduler.step()
        # print(self.scheduler.get_lr()[0])
        
        return total_loss.item()
    
        
    def load_env(self):
        # TODO: 최적화와 동기화
        
        empty_slots = pd.read_csv(os.path.join(self.args.input_dir, f"{self.args.test_case}_slot_infos.csv"), encoding="cp949")
        del empty_slots["Unnamed: 0"]
        for name in self.labels_encoder.keys():
            if name == "WY": continue
            area_slots = empty_slots.loc[empty_slots[NAME] == name].copy().reset_index(drop=True)
            
            empty_location_slot_extended = []
            for idx, row in area_slots.iterrows():
                temp_slots = np.zeros((row[COUNT], 6))
                temp_slots[:, :2] = np.nan
                temp_slots[:, 2:5] = np.repeat(row[[LENGTH, WIDTH, HEIGHT]].values.reshape(-1, 1), row[COUNT], axis=1).T
                temp_slots[:, 5] = 1000
                temp_slots = pd.DataFrame(temp_slots, columns=[VESSEL_ID, BLOCK, LENGTH, WIDTH, HEIGHT, LOCATION])
                empty_location_slot_extended.append(temp_slots)
            area_slots = pd.concat(empty_location_slot_extended) if len(empty_location_slot_extended) > 0 else None
            
            self.yards_out[name] = YardOutside(args=self.args, name=name, area_slots= area_slots)
            self.yards_out[name]._load_env()
            
        if self.args.priori_conditions is not None:
            # self.args.priori_conditions["temp1"] = self.args.priori_conditions[NAME]
            self.args.priori_conditions[NAME] = self.args.priori_conditions[NAME].map(self.labels_encoder)
            self.args.priori_conditions = self.args.priori_conditions.values
            self.args.priori_conditions[:, 1] -= 1
            
        if self.args.n_trip_infos is not None:
            self.args.n_trip_infos_list = []
            for key in self.labels_encoder.keys():
                if key == "WY" : continue
                if key not in self.args.n_trip_infos.keys():
                    self.args.n_trip_infos_list.append(0)
                else:
                    self.args.n_trip_infos_list.append(self.args.n_trip_infos[key])
                    
    def init_agent(self):
        if self.args.do_att:
            self.RLAgent = PGAgent_Att(self.args)
        else:
            if self.args.do_emb:
                self.RLAgent = PGAgent(self.args)
            else:
                self.RLAgent = PGAgent_NoEmb(self.args)
        self.RLAgent.to(self.args.device)
        # self.optimizer = torch.optim.AdamW(self.RLAgent.parameters(), lr=self.args.lr)
        self.optimizer = torch.optim.Adam(self.RLAgent.parameters(), lr=self.args.lr)
        # self.optimizer = torch.optim.SGD(self.RLAgent.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.lr_decay_step, gamma=self.args.lr_decay_factor)
