import numpy as np
import d3rlpy
import torch
import pandas as pd

excel_file = "data.xlsx"  # 将此处替换为您的Excel文件路径
df = pd.read_excel(excel_file)

all_columns_data = df.values.T
features = all_columns_data[:, 0:6]
features = features / [420, 420, 420, 420, 400, 400]
sparse_reward = all_columns_data[:, 7]

observations = np.load('./state.npy')
observations = observations[:, :]
observations = np.swapaxes(observations, 0, 1)
terminals = np.load('./terminals.npy')
print(terminals)
exit()
actions = np.random.randint(2, size=328)
reward = sparse_reward
print(reward)
print(observations.shape)
print(terminals.shape)
print(reward.shape)
print(terminals.shape)
exit()
dataset = d3rlpy.dataset.MDPDataset(
    observations=observations,
    actions=actions,
    rewards=reward,
    terminals=terminals,
)
dataset.dump('random_dataset_RL.h5')
