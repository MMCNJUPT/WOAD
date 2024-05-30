import d3rlpy.ope
from sklearn.model_selection import train_test_split
import numpy as np
from d3rlpy.preprocessing import MinMaxScaler
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteSAC
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer

new_dataset = MDPDataset.load('./RL_data/random_dataset_RL.h5')
train_episodes, test_episodes = train_test_split(new_dataset, test_size=0.2)

discreteSAC = DiscreteSAC(use_gpu=True)
discreteSAC.build_with_dataset(new_dataset)
td_error = td_error_scorer(discreteSAC, test_episodes)

results = discreteSAC.fit(train_episodes,
                          eval_episodes=test_episodes,
                          n_epochs=100,
                          scorers={
                              'td_error': td_error_scorer,
                              'value_scale': average_value_estimation_scorer,
                          })

discreteSAC.save_model('./RL_data/DiscreteSAC.pt')


