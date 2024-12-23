You can access all the data and source codes from [Google Drive Link](https://drive.google.com/file/d/11F03TIDk7saP8CxHhtUQ2NUzwRudHL46/view?usp=sharing)

### Environment

* Python 3.8  
* Torch 1.13.0  
* D3rlpy 1.1.1

1. Run `pip install -r requirements.txt` to install all dependencies required in your machine.

2. Import PyTorch with the correct CUDA version.

The installation time will take no longer than 20 minutes on a "normal" desktop computer with good Internet conditions.

### Data availability

All the necessary data for the demo is included in the `./RL_data folder`.

Download the data form [Google Drive Link](https://drive.google.com/file/d/100_sZQyzOolvUEu6isxiwMHsYrxk5sC5/view?usp=sharing) and put it into the `./RL_data folder`.

### Demo
* Run `python train_RL.py` to get policy model without cerebellar reward.
* Run `python creat_MDPDataset_cerebellar_reward.py` to get dataset with cerebellar reward.
* Run `python train_cerebellar_RL.py` to get policy model with cerebellar reward.
* Run `python save_DSAC_onnx_model.py` to save poliicy model.

In the end, you will obtain `policy_cerebellar.onnx`, which will be deployed on the WOAD as C++ code for pre-processing.

The expected run time of the demo will take no longer than 10 minutes on a "normal" desktop computer.
