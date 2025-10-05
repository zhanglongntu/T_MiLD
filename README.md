# T-MiLD: Combine Multi-Modal Features with Transfer Learning for Effective Long-Tailed Vulnerability Classification

This is the source code to the paper "T-MiLD: Combine Multi-Modal Features with Transfer Learning for Effective Long-Tailed Vulnerability Classification". Please refer to the paper for the experimental details.

## Approach
![](https://github.com/zhanglongntu/T_MiLD/blob/main/Fig/Framework.png)
## About dataset.
Due to the large size of the datasets, we have stored them in Google Drive: [Dataset Link](https://drive.google.com/file/d/1T7aeMu3scAEFCNFMIIPeFgrtUIv7_ZGz/view?usp=drive_link)

## Requirements
You can install the required dependency packages for our environment by using the following command: ``pip install - r requirements.txt``.

## Reproducing the experiments:

1.You can directly use the ``dataset`` we have processed: [Google Drive Link](https://drive.google.com/file/d/1T7aeMu3scAEFCNFMIIPeFgrtUIv7_ZGz/view?usp=drive_link)

2.Run ``train_sva.py`` and ``train_svc.py``. After running, you can retrain the ``model`` and obtain results.

3.You can find the implementation code for the ``RQ1-RQ4`` section and the ``Dis1, Dis2`` section experiments in the corresponding folders. 
