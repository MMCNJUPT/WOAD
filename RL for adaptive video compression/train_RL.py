import d3rlpy.ope
from sklearn.model_selection import train_test_split
import numpy as np
from d3rlpy.preprocessing import MinMaxScaler
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteSAC, DiscreteBCQ, DiscreteCQL, DoubleDQN, DQN, NFQ
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer

d3rlpy.seed(42)

new_dataset = MDPDataset.load('./cerebellar_dataset_RL.h5')
train_episodes, test_episodes = train_test_split(new_dataset, test_size=0.2)

scaler = MinMaxScaler(new_dataset)

# discreteSAC = DiscreteSAC(use_gpu=False)
# discreteSAC.build_with_dataset(new_dataset)
# td_error = td_error_scorer(discreteSAC, test_episodes)
#
# results = discreteSAC.fit(train_episodes,
#                           eval_episodes=test_episodes,
#                           n_epochs=100,
#                           scorers={
#                               'td_error': td_error_scorer,
#                               'value_scale': average_value_estimation_scorer,
#                           })
# discreteSAC.save_model('./save_model/DiscreteSAC_cerebellar.pt')

# discreteBCQ = DiscreteBCQ(use_gpu=False)
# discreteBCQ.build_with_dataset(new_dataset)
# td_error = td_error_scorer(discreteBCQ, test_episodes)
#
# results = discreteBCQ.fit(train_episodes,
#                           eval_episodes=test_episodes,
#                           n_epochs=100,
#                           scorers={
#                               'td_error': td_error_scorer,
#                               'value_scale': average_value_estimation_scorer,
#                           })
# discreteBCQ.save_model('./save_model/DiscreteBCQ_cerebellar.pt')

# DiscreteCQL = DiscreteCQL(use_gpu=False)
# DiscreteCQL.build_with_dataset(new_dataset)
# td_error = td_error_scorer(DiscreteCQL, test_episodes)
#
# results = DiscreteCQL.fit(train_episodes,
#                           eval_episodes=test_episodes,
#                           n_epochs=100,
#                           scorers={
#                               'td_error': td_error_scorer,
#                               'value_scale': average_value_estimation_scorer,
#                           })
# DiscreteCQL.save_model('./save_model/DiscreteCQL_cerebellar.pt')

DoubleDQN = DoubleDQN(use_gpu=False)
DoubleDQN.build_with_dataset(new_dataset)
td_error = td_error_scorer(DoubleDQN, test_episodes)

results = DoubleDQN.fit(train_episodes,
                          eval_episodes=test_episodes,
                          n_epochs=100,
                          scorers={
                              'td_error': td_error_scorer,
                              'value_scale': average_value_estimation_scorer,
                          })
DoubleDQN.save_model('./save_model/DoubleDQN_cerebellar.pt')

# DQN = DQN(use_gpu=False)
# DQN.build_with_dataset(new_dataset)
# td_error = td_error_scorer(DQN, test_episodes)
#
# results = DQN.fit(train_episodes,
#                   eval_episodes=test_episodes,
#                   n_epochs=100,
#                   scorers={
#                       'td_error': td_error_scorer,
#                       'value_scale': average_value_estimation_scorer,
#                   })
# DQN.save_model('./save_model/DQN_cerebellar.pt')


NFQ = NFQ(use_gpu=False)
NFQ.build_with_dataset(new_dataset)
td_error = td_error_scorer(NFQ, test_episodes)

results = NFQ.fit(train_episodes,
                  eval_episodes=test_episodes,
                  n_epochs=100,
                  scorers={
                      'td_error': td_error_scorer,
                      'value_scale': average_value_estimation_scorer,
                  })
NFQ.save_model('./save_model/NFQ_cerebellar.pt')

# critic_loss = np.zeros(100)
# actor_loss = np.zeros(100)
# td_error_save = np.zeros(100)
# value_scale = np.zeros(100)
# print(results)
#
#
# for i in range(100):
#     # critic_loss[i] = results[i][1]['critic_loss']
#     actor_loss[i] = results[i][1]['actor_loss']
#     td_error_save[i] = results[i][1]['td_error']
#     value_scale[i] = results[i][1]['value_scale']

# print(critic_loss, actor_loss)
# np.save('./save_data/critic_loss_discreteBCQ.npy', critic_loss)
# np.save('./save_data/actor_loss_NFQ.npy', actor_loss)
# np.save('./save_data/td_error_discreteSAC.npy', td_error_save)
# np.save('./save_data/value_scale_NFQ.npy', value_scale)
