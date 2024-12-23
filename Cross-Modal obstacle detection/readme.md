You can access all the data and source codes from [Google Drive Link](https://drive.google.com/file/d/1tJpmOu5DCxapU0XcoKPt76LHkW18s5xB/view?usp=sharing)

### Environment

* Python 3.8
* Torch 1.13.1

1. Run `pip install -r requirements.txt` to install all dependencies required in your machine.

2. Import PyTorch with the correct CUDA version.

The installation time will take no longer than 20 minutes on a "normal" desktop computer with good Internet conditions.

### Data availability

The training/validation/test datasets can be found in the [Google Drive Link](https://drive.google.com/file/d/1B9atHC74YXSZ89GcjtqVp_gJUezdxpPE/view?usp=sharing) and put them into the `./`.

### Demo
* Run `python train.py --data data/stairscoco_nc_4/6/8/10.yaml`.

After the training is complete, you can find the optimal weights in `/runs/train/best.pt`, which will be deployed on the mobile device.

* Run `python val.py --data data/stairscoco_nc_4/6/8/10.yaml` to get dataset with cerebellar reward.

You can determine the performance of the weights on the test set.

The installation time will take no longer than 120 minutes on a "normal" desktop computer with GPU device.
