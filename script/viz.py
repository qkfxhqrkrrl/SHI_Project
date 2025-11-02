#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#%%

def viz_loss():
    # fig, axs = plt.subplots(2, 1, figsize=(20, 10))
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    fig2, ax2 = plt.subplots(1, 1, figsize=(15, 10))
    for chr_dir in os.listdir("."):
        if os.path.isdir(chr_dir):
            if len(chr_dir.split("_")) != 4: continue
            data_size, do_att, do_emb, data_case = chr_dir.split("_")
            data_size = "Small" if "Small" in data_size else "Large"
            do_att = "True" if "True" in do_att else "False"
            do_emb = "True" if "True" in do_emb else "False"

            result_dir = os.path.join(chr_dir, "temp")
            # for result_file in os.listdir(result_dir):
            df = pd.read_csv(os.path.join(result_dir, "loss.csv"))

            if do_att == "True" and do_emb == "True": color = plt.cm.tab10(0)
            elif do_att == "False" and do_emb == "True": color = plt.cm.tab10(1)
            elif do_att == "False" and do_emb == "False": color = plt.cm.tab10(2)

            if data_case == "Strict" and data_size == "Large":
                ax.plot(df.iloc[:, 1], label=f"Attention: {do_att}, Emb: {do_emb}", color=color)
            elif data_case == "Strict" and data_size == "Small":
                ax2.plot(df.iloc[:, 1], label=f"Attention: {do_att}, Emb: {do_emb}", color=color)
                
    ax.set_title("Dataset size: Large", fontsize=20)
    ax.legend(fontsize=16)
    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_xticklabels([int(elem) for elem in ax.get_xticks()], fontsize=14)
    ax.set_ylabel("Loss", fontsize=16)
    ax.set_yticklabels([int(elem) for elem in ax.get_yticks()], fontsize=14)
    fig.show()

    ax2.set_title("Dataset size: Small", fontsize=20)
    ax2.legend(fontsize=16)
    ax2.set_xlabel("Epoch", fontsize=16)
    ax2.set_xticklabels([int(elem) for elem in ax2.get_xticks()], fontsize=14)
    ax2.set_ylabel("Loss", fontsize=16)
    ax2.set_yticklabels([int(elem) for elem in ax2.get_yticks()], fontsize=14)
    fig2.show()
    
viz_loss()

# %%

def viz_time():
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    for chr_dir in os.listdir("."):
        # print(chr_dir)
        if os.path.isdir(chr_dir):
            if len(chr_dir.split("_")) != 4: continue
            data_size, do_att, do_emb, data_case = chr_dir.split("_")
            data_size = "Small" if "Small" in data_size else "Large"
            do_att = True if "True" in do_att else False
            do_emb = True if "True" in do_emb else False

            result_dir = os.path.join(chr_dir, "temp")
            inf = open(os.path.join(result_dir, "result.txt"), "r")

            start_idx = 0
            next_start_idx = 1
            width = 0.1
            time = round(float(inf.readlines()[-1].split(",")[-1].rstrip("\n")), 3)
            if do_att and do_emb and data_size == "Large" and data_case == "Strict":
                ax.bar(start_idx + 0*width, time, width=width, color=plt.cm.tab10(0), label=f"Attention: {do_att} / Embedding: {do_emb}")
            elif not do_att and do_emb and data_size == "Large" and data_case == "Strict":
                ax.bar(start_idx + 1*width, time, width=width, color=plt.cm.tab10(1), label=f"Attention: {do_att} / Embedding: {do_emb}")
            elif not do_att and not do_emb and data_size == "Large" and data_case == "Strict":
                ax.bar(start_idx + 2*width, time, width=width, color=plt.cm.tab10(2), label=f"Attention: {do_att} / Embedding: {do_emb}")
            elif do_att and do_emb and data_size == "Small" and data_case == "Strict":
                ax.bar(next_start_idx + 0*width, time, width=width, color=plt.cm.tab10(0))
            elif not do_att and do_emb and data_size == "Small" and data_case == "Strict":
                ax.bar(next_start_idx + 1*width, time, width=width, color=plt.cm.tab10(1))
            elif not do_att and not do_emb and data_size == "Small" and data_case == "Strict":
                ax.bar(next_start_idx + 2*width, time, width=width, color=plt.cm.tab10(2))
    ax.set_xticks([start_idx + width, next_start_idx + width])
    ax.set_xticklabels(["Large", "Small"], fontsize=18)
    ax.set_xlabel("Data Size", fontsize=20)
    ax.set_yticklabels(ax.get_yticks(), fontsize=18)
    ax.set_ylabel("Time (sec)", fontsize=20)
    plt.legend(fontsize=18)
    plt.title("Time comparison", fontsize=24)
    plt.show()
viz_time()
# %%

def viz_hyperparam():
    # fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    for chr_dir in os.listdir("."):
        # print(chr_dir)
        if os.path.isdir(chr_dir):
            if len(chr_dir.split("_")) != 6: continue
            data_size, do_att, do_emb, data_case, alpha, beta = chr_dir.split("_")
            data_size = "Small" if "Small" in data_size else "Large"
            do_att = True if "True" in do_att else False
            do_emb = True if "True" in do_emb else False
            alpha = float(alpha.lstrip("alpha"))
            beta = float(beta.lstrip("beta"))
            
            result_dir = os.path.join(chr_dir, "temp")

            # print(os.path.exists(result_dir))
            # print(os.path.exists(os.path.join(result_dir, "shi_RL_result.csv")))
            df = pd.read_csv(os.path.join(chr_dir, "shi_RL_result.csv"), encoding="cp949")
            remaining_space = df.loc[df["유휴면적"]>=0]["유휴면적"].sum()
            alloc_error = df.loc[df["유휴면적"]<0]["유휴면적"].count() 
            barge_error = df.loc[df["바지"]<0]["바지"].count()

            print(f"Alpha: {alpha}, Beta: {beta} -\t Remaining Space: {remaining_space} / Num Errors: {alloc_error} / Barge Errors: {barge_error}")
            display(df.loc[df["유휴면적"]<0][["슬롯길이", "슬롯너비", "블록길이", "블록너비"]])

viz_hyperparam()

# %%
