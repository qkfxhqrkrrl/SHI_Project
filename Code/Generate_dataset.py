#%%

import numpy as np
import pandas as pd

import os
from glob import glob

data_dir = r"C:\Users\test\Downloads\SHI_real_data"
os.makedirs(os.path.join(data_dir, "dataset"), exist_ok=True)

csv_paths = glob(os.path.join(data_dir, "*.csv"))

csv_paths_by_case = {}

Not_yards = ["DB_empty_slots", "DB_whole_block", "shi_take_in", "shi_take_out", "whole_block"]

slot_infos = []
for path in csv_paths:
    name = os.path.basename(path).replace(".csv", "")
    
    if name in Not_yards:
        continue
    else: 
        df_temp = pd.read_csv(path)
        df_temp["name"] = name
        # display(df_temp.loc[df_temp["vessel_id"].isnull()].value_counts())
        count = df_temp.loc[df_temp["vessel_id"].isnull(), ["name", "length", "width", "height"]].groupby(["length", "width"], as_index=False).value_counts()
        count = count[["length", "width", "height", "count", "name"]]
        slot_infos.append(count)
slot_infos = pd.concat(slot_infos).reset_index(drop=True)
slot_infos.to_csv(os.path.join(data_dir, "dataset", f"slots.csv"), index=False, encoding="cp949")
    # if name == ""

blocks_in = pd.read_csv(os.path.join(data_dir, "shi_take_in.csv"), encoding="cp949")
blocks_out = pd.read_csv(os.path.join(data_dir, "shi_take_out.csv"), encoding="cp949")
pd.concat([blocks_in, blocks_out]).to_csv(os.path.join(data_dir, "dataset", "block_plan.csv"), index=False, encoding="cp949")
#%%
for case in range(10):
    block_plan = []
    slot_infos = []
    
    paths = csv_paths_by_case[str(case)]
    
    for path in paths:
        if "take_in" in path:
            block_plan.append(pd.read_csv(path))
        elif "take_out" in path:
            block_plan.append(pd.read_csv(path))
        else:
            name = os.path.basename(path).split("_")[1].replace(".csv", "")
            if name == "yard" or name == "whole":
                continue
            df_temp = pd.read_csv(path)
            df_temp["name"] = name
            # display(df_temp.loc[df_temp["vessel_id"].isnull()].value_counts())
            count = df_temp.loc[df_temp["vessel_id"].isnull(), ["name", "length", "width", "height"]].groupby(["length", "width"], as_index=False).value_counts()
            count = count[["name", "length", "width", "height", "count"]]
            # display(count)
            slot_infos.append(count)
    
    block_plan = pd.concat(block_plan).reset_index(drop=True)
    slot_infos = pd.concat(slot_infos).reset_index(drop=True)
    
    block_plan.to_csv(os.path.join(data_dir, "dataset", f"{case}_block_plan.csv"), index=False)
    slot_infos.to_csv(os.path.join(data_dir, "dataset", f"{case}_slot_infos.csv"), index=False)
    # display(slot_infos)


# print(slot_infos)


# %%
