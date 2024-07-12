### Environment:
python 3.8  
torch 1.13.0  
d3rlpy 1.1.1  

### Demo:
All the necessary data for the demo is included in the /RL_data folder.

1. python creat_MDPDataset.py  
   
2. python train_RL.py

3. python creat_MDPDataset_cerebellar_reward.py

4. python train_cerebellar_RL.py

5. python save_DSAC_onnx_model.py

6. python eval_discreteSAC.py

In the end, you will obtain policy_cerebellar.onnx, which will be deployed on the WOAD as C++ code for pre-processing.
