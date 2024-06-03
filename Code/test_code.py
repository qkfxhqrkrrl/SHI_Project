#%%

import pandas as pd
import numpy as np
import os
from random import sample

def random_result():
    data_dir = r"D:\Dropbox\Projects_Mine\삼성중공업\code\SHI_Project\Result\2024_3_4_17_48_test_data\dataset"
    action_dir = r"D:\Dropbox\Projects_Mine\삼성중공업\code\SHI_Project\Result\2024_3_5_17_59_dataset2"
    temp_file = open(os.path.join(data_dir, "rand_remaining.txt"), "w")

    for case_num in range(10):
        block_info = pd.read_csv(os.path.join(data_dir, f"{case_num}_block_plan.csv"))
        block_info = block_info.loc[block_info["location"] == "사내"].reset_index(drop=True)
        block_info = block_info.sample(len(block_info))
        slot_info = pd.read_csv(os.path.join(data_dir, f"{case_num}_slot_infos.csv"))
        before = slot_info["count"].sum() 
        
        # display(block_info.head())
        # display(slot_info.head())
        yard_cases = slot_info["name"].unique().tolist()
        # print(yard_cases)
        
        remaining_space = 0
        for block in block_info.values:
            possible_yard_cases = []
            for yard in yard_cases:
                yard_info = slot_info.loc[slot_info["name"] == yard].values
                yard_mask = np.any(np.logical_and(np.all(block[2:5] < yard_info[:, 1:4], axis=-1), yard_info[:, 4] > 0))
                if yard_mask:
                    possible_yard_cases.append(yard)
                
            rand_yard = sample(possible_yard_cases, 1)[0]
            yard_info = slot_info.loc[slot_info["name"] == rand_yard].values
            start_index = slot_info.loc[slot_info["name"] == rand_yard].index[0]
            yard_offset = np.argmin(np.abs((float(block[2]) * float(block[3])) - (yard_info[:, 1] * yard_info[:, 2])), axis=-1)
            
            remaining_space += np.min(np.abs((float(block[2]) * float(block[3])) - (yard_info[:, 1] * yard_info[:, 2])), axis=-1) ** 2
            
            slot_info.loc[slot_info.index[start_index+yard_offset], "count"] -= 1
            slot_info = slot_info.loc[slot_info["count"] != 0].reset_index(drop=True)
            
        after = slot_info["count"].sum() 
        # print(remaining_space)
        # print(before - after)
        # temp_file.write(f"{case_num},{remaining_space}\n") 
        temp_file.write(f"{remaining_space}\n") 
    temp_file.close()
    
# random_result() 
def RL_result():
    data_dir = r"D:\Dropbox\Projects_Mine\삼성중공업\code\SHI_Project\Result\2024_3_4_17_48_test_data\dataset"
    action_dir = r"D:\Dropbox\Projects_Mine\삼성중공업\code\SHI_Project\Result\2024_3_5_17_59_dataset2"
    temp_file = open(os.path.join(data_dir, "RL_remaining.txt"), "w")

    # for case_num in range(10):
    for case_num in range(1, 2):
        slot_info = pd.read_csv(os.path.join(data_dir, f"{case_num}_slot_infos.csv"))
        before = slot_info["count"].sum() 
        
        action = pd.read_csv(os.path.join(action_dir, f"{case_num}_actions.csv"))
        # display(action.head())
        # display(block_info.head())
        # display(slot_info.head())
        yard_cases = slot_info["name"].unique().tolist()
        # print(yard_cases)
        
        remaining_space = 0
        for block in action.values[:-2]:
                
            rand_yard = block[-1]
            # print(rand_yard)
            yard_info = slot_info.loc[(slot_info["name"] == rand_yard) & np.all(block[2:5] < slot_info.values[:, 1:4], axis=-1)].values
            start_index = slot_info.loc[slot_info["name"] == rand_yard].index[0]
            yard_offset = np.argmin(np.abs((float(block[2]) * float(block[3])) - (yard_info[:, 1] * yard_info[:, 2])), axis=-1)
            
            print(block)
            print(slot_info.loc[slot_info.index[start_index+yard_offset]].values)
            print()
            # if 0 in np.abs((float(block[2]) * float(block[3])) - (yard_info[:, 1] * yard_info[:, 2])):
                # print(block, yard_info)
            # print(np.abs((float(block[2]) * float(block[3])) - (yard_info[:, 1] * yard_info[:, 2])))
            
            remaining_space += np.min(np.abs((float(block[2]) * float(block[3])) - (yard_info[:, 1] * yard_info[:, 2])), axis=-1) ** 2
            
            slot_info.loc[slot_info.index[start_index+yard_offset], "count"] -= 1
            slot_info = slot_info.loc[slot_info["count"] != 0].reset_index(drop=True)
        # print(slot_info)
        print(remaining_space)
        # after = slot_info["count"].sum() 
        # print(before - after)
        # temp_file.write(f"{case_num},{remaining_space}\n") 
        temp_file.write(f"{remaining_space}\n") 
    temp_file.close()
#%%

# import pandas as pd

# temp = pd.read_csv(r"D:\Dropbox\Projects_Mine\삼성중공업\code\SHI_Project\Result\2024_3_5_17_59_dataset2\0_actions.csv")
# temp.loc[temp.index[-2],"vessel_id"] = "Remaining space"
# temp.loc[temp.index[-1],"vessel_id"] = "Time"
# display(temp)

    
# RL_result() 
    
# batch_num = self.calculate_batch_nums(block_size, yard)


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
# block_mask = torch.ones((n_take_out, 1), dtype=bool) # 제외 시키고 싶은 경우를 True
# block_mask[1:6:2] = False
# print(block_mask.repeat(1, 10))
# print(block_mask.repeat(1, 10).reshape(-1))

# block_mask.repeat(1, 10).reshape(-1)