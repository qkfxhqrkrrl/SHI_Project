import os

import torch
import pandas as pd

from script.func import *


def run_optimizer(par_dir, num_batch,alpha, beta, gamma):
    data_dir = os.path.join(par_dir, 'data')
    """
    batch_capa : 바지선 길이
    num_batch : 사용가능한 최대 바지선 수
    """
    batch_capa = 70

    df_block = pd.read_csv(os.path.join(data_dir, f'{test_case}_block_plan_preproc.csv'), encoding='cp949')
    df_slot = pd.read_csv(os.path.join(data_dir, f'{test_case}_slot_infos.csv'), encoding='cp949')
    df_times = pd.read_csv(os.path.join(data_dir, f'{test_case}_times.csv'), encoding='cp949')

    """ 
    모델 사용법 1
    """
    # # 모델 생성
    # model = MathOpt(df_block, df_slot, df_times, name=test_case)
    # # 모델 전처리
    # model.pre_process()
    # # 불가능한 블록 체크
    # model.check_impossible_block()
    # # 최적화 모델 실행
    # model.cplex_model_run()
    # # 후처리
    # df_result = model.post_process()
    # # 항차 최적화
    # MathOpt.optimize_batch(df_result, name=test_case)

    """ 
    모델 사용법 2
    """
    # # 모델 생성
    model = MathOpt(df_block, df_slot, df_times, batch_capa=batch_capa, num_batch=num_batch, alpha=alpha,
                    beta=beta, gamma=gamma, name=test_case)
    # 모델 전처리
    df_result = model.run()
    MathOpt.optimize_batch(df_result)
    #
    """
    항차 최적화는 파일 양식만 맞으면 개별적으로 가능합니다.
    """
    # Data_Result = os.path.join(os.getcwd(), 'result')
    # # 파일 읽기
    # df_result = pd.read_csv(os.path.join(Data_Result, f'{test_case}_result.csv'), encoding='cp949')
    # ShipOpt.optimize_batch(df_result, test_case)

def run_RL(par_dir, num_batch, alpha, beta, param):

    torch.manual_seed(3407)
    torch.cuda.manual_seed(3407)

    args = Args()

    args.input_dir = os.path.join(par_dir, f"data")
    args.save_dir = os.path.join(par_dir, f"result")

    args.test_case = test_case

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.do_att = False
    args.do_emb = True
    # args.device = "cpu"
    args.lr = 5e-3
    args.std_eps = 1e-5
    args.clipping_size = 1  # RNN 모델의 그래디언트 폭주 문제를 막기 위해 설정하는 파라미터입니다. 크기가 너무 작으면 업데이트가 충분히 이루어지지 않아 설정 해둔 값이 제일 좋은 걸로 확인했습니다
    args.batch_size = 256  # 딥러닝 모델에서 동시에 실험해보는 경우의 수입니다. 크기가 너무 커져도 분산의 크기 증가로 성능 하락을 초래 할 수 있습니다.
    args.n_epoch = 10  # 빠르게 확인해보고 싶은 경우 200까지 돌려도 무난합니다만, Epoch 크기가 커질수록 성능이 향상됩니다
    args.hid_dim = 256  # 모델의 크기
    args.emb_dim = 8 
    args.decay = 0.95
    args.pad_len = 10  # 야드 슬롯의 Case를 표현하는 값입니다. 크기가 클수록 다양한 Slot들의 경우를 고려해줄 수 있지만 속도는 느려집니다.
    args.n_max_block = 50  # 최대 고려 가능한 블록 수: 수가 많아질 수록 느려집니다. 진행했던 실험은 60으로 두고 진행했습니다.
    args.max_trip = 10  # 바지선 최대 항차 수 입니다. 기록하는 Times보단 클 수 있지만, 해당 경우는 Penalty를 통해 처리할 예정입니다.
    args.barge_max_row = 5  # 한 바지선이 최대 수용할 수 있는 블록의 수입니다.
    args.barge_par_width = 23
    args.barge_max_length = 70
    args.lr_decay_step = 50 
    args.lr_decay_factor = 0.1
    args.alpha = alpha  # 목적식 파라미터 - 유휴 공간
    args.beta = -beta  # 목적식 파라미터 - 바지선 개수
    args.num_batch = num_batch

    train_RL(args)


if __name__ == '__main__':
    """
    alpha 값이 높을수록 바지선 유휴면적을 최소화합니다.
    beta 값이 높을수록 항차수를 최소화합니다
    gamma 값이 높을수록 높이 제약이 있는 적치장에 블록을 할당합니다(수리모델 한정)
    
    num_batch : 수리모델 : 최대 항차수를 제한합니다
                강화학습 : shi_impossible 파일이 없다면 num_batch를 기반으로 shi_impossible 파일을 생성합니다.
                          shi_impossible 파일이 존재한다면 강화학습에서 사용되지 않습니다. 
    """
    alpha, beta, gamma = 0.001, 1, 1
    num_batch = 25

    par_dir = os.path.abspath(os.path.join(__file__, *[os.pardir] * 1))

    test_case = "shi"
    
    run_RL(par_dir, num_batch, alpha, beta, (alpha, beta, "", "Strict", False, True))

    # for data_size in ["Small", ""]:
    #     for data_case in ["Strict", "LessStrict"]:
    #         for exp_case in [(True, True), (False, False), (False, True)]:
    #             run_RL(par_dir, num_batch, alpha, beta, (data_size, data_case, *exp_case))

    # for alpha in [0.01, 0.001, 0.0001]:
    #     for beta in [1, 5, 10]:
    #         run_RL(par_dir, num_batch, alpha, beta, (alpha, beta, "", "Strict", False, True))

# %%
