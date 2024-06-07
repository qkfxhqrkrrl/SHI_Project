#%%

import numpy as np
import pandas as pd

import os
from glob import glob

data_dir = r"D:\Dropbox\Projects_Mine\삼성중공업\code\SHI_Project\Result\2024_3_4_17_48_test_data"
os.makedirs(os.path.join(data_dir, "dataset"), exist_ok=True)

csv_paths = glob(os.path.join(data_dir, "*.csv"))

csv_paths_by_case = {}

for path in csv_paths:
    case_num = os.path.basename(path).split("_")[0]
    if case_num not in csv_paths_by_case.keys():
        csv_paths_by_case[case_num] = []
    csv_paths_by_case[case_num].append(path)

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
