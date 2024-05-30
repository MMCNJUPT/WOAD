import numpy as np
import d3rlpy
import torch
import pandas as pd

excel_file = "./RL_data/RL_data.xlsx"  # 将此处替换为您的Excel文件路径
df = pd.read_excel(excel_file)

all_columns_data = df.values.T
features = all_columns_data[:, 0:6]
features = features / [420, 420, 420, 420, 400, 400]
sparse_reward = all_columns_data[:, 7]

observations = np.load('./RL_data/state.npy')
observations = observations[:, :]
observations = np.swapaxes(observations, 0, 1)
terminals = np.load('./RL_data/terminals.npy')
actions = np.random.randint(2, size=328)
reward = sparse_reward

dataset = d3rlpy.dataset.MDPDataset(
    observations=observations,
    actions=actions,
    rewards=reward,
    terminals=terminals,
)
dataset.dump('./RL_data/random_dataset_RL.h5')
