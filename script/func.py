import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
import time

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.policy_gradient import *


class Args:
    def __init__(self) -> None:
        self.device = None
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


def match_batch_num(result, result_barge, n_trip_infos):
    var1, var2, var3, var4 = result_barge.shape
    for y_idx in range(var1):  # 야드 인덱스, 굳이 result1에는 안 쓰임. 이미 destination으로 매칭되어 있기 때문
        for t_idx in range(var2):  # 항차 인덱스, 몇 번째 항차인지
            for s_idx in range(var3):  # 슬롯 인덱스
                for p_idx in range(var4):  # 병렬 여부
                    block_num = result_barge[y_idx, t_idx, s_idx, p_idx]
                    if block_num == -1:
                        continue
                    # * 블록 번호랑, index랑 매칭
                    # print(block_num.item())
                    if p_idx == 0:
                        is_parallel = (result_barge[y_idx, t_idx, s_idx, p_idx + 1] != -1)
                    elif p_idx == 1:
                        is_parallel = True
                    else:
                        is_parallel = False

                    max_trip = n_trip_infos[y_idx]
                    if t_idx >= max_trip:
                        result.loc[block_num.item(), ["바지", "슬롯", "병렬"]] = [-1, -1, False]
                    else:
                        result.loc[block_num.item(), ["바지", "슬롯", "병렬"]] = [t_idx, s_idx, is_parallel]


def make_input_from_loaded(blocks_to_be_out, blocks_to_be_in, env):
    encoder_inputs, yard_slots, remaining_areas = env.get_init_state(blocks_to_be_out.copy(), blocks_to_be_in.copy())
    return encoder_inputs, remaining_areas, yard_slots


def save_result(result, args, infos, labels_encoder, obj_loss_history=None, dl_loss_history=None):
    take_in, take_out = infos
    temp_dir = os.path.join(args.save_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    if obj_loss_history is not None:
        plt.plot(obj_loss_history)
        plt.savefig(os.path.join(temp_dir, "loss.png"))
        plt.close()
        pd.Series(obj_loss_history).to_csv(os.path.join(temp_dir, "loss.csv"))
    if dl_loss_history is not None:
        plt.plot(dl_loss_history)
        plt.savefig(os.path.join(temp_dir, "dl_loss.png"))
        plt.close()
        pd.Series(dl_loss_history).to_csv(os.path.join(temp_dir, "dl_loss.csv"))

    take_out_blocks = result["take_out_actions"]
    result1 = take_out.iloc[take_out_blocks[:, 0]]
    result1 = result1[[col for col in result1.columns if col != LOCATION]]
    # print(result1.to_csv(r"C:\Users\qkfxh\Desktop\test.csv"))
    # print(take_out_blocks.to_csv(r"C:\Users\qkfxh\Desktop\test2.csv"))
    result1[LOCATION] = "사내"
    result1["destination"] = take_out_blocks[:, 1] + 1
    result1["destination"] = result1["destination"].map(labels_encoder)
    result1["slot_length"] = take_out_blocks[:, 2]
    result1["slot_width"] = take_out_blocks[:, 3]
    result1["slot_height"] = take_out_blocks[:, 4]
    result1["slot_space"] = result1["slot_length"] * result1["slot_width"]
    result1["block_space"] = result1[LENGTH] * result1[WIDTH]
    result1["empty_space"] = result1["slot_space"] - result1["block_space"]
    match_batch_num(result1, result["take_out_barges"], args.n_trip_infos_list)

    take_in_blocks = result["take_in_actions"]
    result2 = take_in.iloc[take_in_blocks]
    result2["block_space"] = result2[LENGTH] * result2[WIDTH]
    result2["destination"] = "사내"
    match_batch_num(result2, result["take_in_barges"], args.n_trip_infos_list)

    df_result = pd.concat([result1, result2])

    df_result.rename(columns={
        LOCATION: 현위치,
        "destination": 목적지,
        "slot_length": 슬롯길이,
        "slot_width": 슬롯너비,
        "slot_height": 슬롯높이,
        # "Batch": "배치번호", 
        "slot_space": 슬롯면적,
        "block_space": 블록면적,
        LENGTH: 블록길이,
        WIDTH: 블록너비,
        "height": 블록높이,
        "empty_space": "유휴면적"}, inplace=True)

    df_result = df_result[
        [VESSEL_ID, BLOCK, 현위치, 목적지, "바지", "슬롯", "병렬", 슬롯면적, 블록면적, 유휴면적, 슬롯너비, 슬롯길이, 슬롯높이, 블록너비, 블록길이, 블록높이]]

    blocks_at_yards = []
    for label in labels_encoder.values():
        if label == "사내": continue
        take_in_temp = df_result.loc[(df_result[현위치] == label)]
        take_out_temp = df_result.loc[(df_result[목적지] == label)]
        if take_in_temp.empty and take_out_temp.empty:
            # print("Empty", label)
            continue
        blocks_at_yard = pd.concat([take_in_temp, take_out_temp], axis=0)
        if take_out_temp.shape[0] == 0:
            blocks_at_yard = blocks_at_yard.sort_values(by=["바지", 현위치]).reset_index(drop=True)
        else:
            blocks_at_yard = blocks_at_yard.sort_values(by=["바지", 목적지]).reset_index(drop=True)
        blocks_at_yards.append(blocks_at_yard)

    df_result = pd.concat(blocks_at_yards, axis=0)

    if obj_loss_history is not None:
        df_result.to_csv(os.path.join(args.save_dir, f"{args.test_case}_RL_result.csv"), index=False, encoding="cp949")
    return df_result


def input_process(args: Args):
    block_plan = pd.read_csv(os.path.join(args.input_dir, f"{args.test_case}_block_plan_preproc.csv"), encoding="cp949")
    # block_plan = pd.read_csv(os.path.join(args.input_dir, "shi_block_plan_preproc.csv"))

    take_in = block_plan.loc[block_plan[LOCATION] != "WY"].copy().reset_index(drop=True)
    take_out = block_plan.loc[block_plan[LOCATION] == "WY"].copy().reset_index(drop=True)


    take_out["key"] = take_out[VESSEL_ID] + "_" + take_out[BLOCK]
    take_in["key"] = take_in[VESSEL_ID] + "_" + take_in[BLOCK]

    priori_blocks = take_out.loc[~take_out[NOT_IN].isnull()]
    if not priori_blocks.empty:
        args.priori_conditions = []
        for idx, row in priori_blocks.iterrows():
            args.priori_conditions.extend([(idx, elem.replace(" ", "")) for elem in row[NOT_IN].split(",")])
    else:
        args.priori_conditions = None

    args.priori_conditions = pd.DataFrame(args.priori_conditions, columns=["block_num", NAME])

    n_trip_infos_path = os.path.join(args.input_dir, f"{args.test_case}_times.csv")
    if os.path.exists(n_trip_infos_path):
        n_trip_infos = pd.read_csv(n_trip_infos_path, encoding="cp949")
        args.n_trip_infos = dict(zip(n_trip_infos[NAME].values, n_trip_infos[NUMBER].values))
    else:
        args.n_trip_infos = None

    args.n_take_in = len(take_in)
    args.n_take_out = len(take_out)

    return take_in, take_out


def train_RL(args: Args):
    take_in, take_out = input_process(args)

    temp_dir = os.path.join(args.save_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    with open(os.path.join(temp_dir, "info.txt"), "w") as f:
        for k, v in args.__dict__.items():
            f.write(f"{k} : {v}\n")

    with open(os.path.join(temp_dir, "result.txt"), "w") as f:
        f.write("exp_num,objective,time\n")
        env = RLEnv(args)
        env.load_env()

        encoder_inputs, remaining_areas, yard_slots = make_input_from_loaded(take_out, take_in,
                                                                             env)  # == env.get_init_state()와 같다
        env.init_agent()

        time1 = time.time()
        obj_loss_history = []
        dl_loss_history = []
        loop = tqdm(range(args.n_epoch), leave=False)
        for epo in loop:
            # print(f"Epoch: {epo}")
            env.reset()
            for day in range(1):
                # objective = env.step((take_in.copy(), take_out.copy(), deepcopy(yard_slots)), deepcopy(encoder_inputs.to(args.device)), deepcopy(remaining_areas))
                objective = env.step((take_in.copy(), take_out.copy(), deepcopy(yard_slots)), deepcopy(encoder_inputs),
                                     deepcopy(remaining_areas))
            loss = env.update_policy()
            loop.set_description(f"Loss: {loss:.3f} / Average Reward: {objective:.3f}")
            # loss2 = env.update_policy2(len(take_in))
            # loop.set_description(f"{exp_num} / {loss:.3f} + {loss2:.3f} / {objective}")
            obj_loss_history.append(objective)
            dl_loss_history.append(loss)
            # break

        time2 = time.time()
        kwargs = env.get_result()
        result_df = save_result(obj_loss_history=obj_loss_history, dl_loss_history=dl_loss_history, result=kwargs,
                                args=args, infos=(take_in, take_out), labels_encoder=env.labels_encoder_inv)

        print(f"Took {(time2 - time1):.3f} seconds")
        f.write(f"{kwargs['take_out_rewards']},{time2 - time1}\n")
