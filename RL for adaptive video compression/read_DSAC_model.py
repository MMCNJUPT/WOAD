import d3rlpy
from d3rlpy.algos import DiscreteSAC
from d3rlpy.dataset import MDPDataset
import numpy as np
import torch

new_dataset = MDPDataset.load('./cerebellar_dataset_RL.h5')
discreteSAC = DiscreteSAC()
discreteSAC.build_with_dataset(new_dataset)
# discreteSAC.load_model('./DiscreteSAC_cerebellar.pt')
# discreteSAC.save_policy('./policy_cerebellar.onnx')


loaded = torch.jit.load('./policy.onnx')
print(loaded)



