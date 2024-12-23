import d3rlpy
from d3rlpy.algos import DiscreteSAC
from d3rlpy.dataset import MDPDataset
import numpy as np

new_dataset = MDPDataset.load('./cerebellar_dataset_RL.h5')
discreteSAC = DiscreteSAC()
discreteSAC.build_with_dataset(new_dataset)
discreteSAC.load_model('./DiscreteSAC_cerebellar.pt')

observations = np.load('./state.npy')
observations = observations[:, :]
observations = np.swapaxes(observations, 0, 1)
observations = observations[:, :] / [420, 420, 420, 420, 400, 400]
new_action = np.zeros(shape=observations.shape[0])
for i in range(len(observations)):
    # print(discreteSAC.predict([observations[i, :]]))
    # print(i, observations[i, :])
    new_action[i] = discreteSAC.predict([observations[i, :]])
    print([observations[i, :]])

np.save(file='./new_action_cerebellar.npy', arr=new_action)
