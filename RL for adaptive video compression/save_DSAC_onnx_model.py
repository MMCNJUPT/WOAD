import d3rlpy
from d3rlpy.algos import DiscreteSAC
from d3rlpy.dataset import MDPDataset
import numpy as np
import torch

new_dataset = MDPDataset.load('./RL_data/cerebellar_dataset_RL.h5')
discreteSAC = DiscreteSAC()
discreteSAC.build_with_dataset(new_dataset)
discreteSAC.load_model('./RL_data/DiscreteSAC_cerebellar.pt')
discreteSAC.save_policy('./RL_data/policy_cerebellar.onnx')





