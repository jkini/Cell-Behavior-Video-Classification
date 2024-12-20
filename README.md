# Cell Behavior Video Classification

## Installations
````
conda create -n mm python=3.11.4
conda activate mm
conda install pytorch=2.0.1 torchvision=0.15.2 torchaudio=2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c anaconda numpy    
conda install -c conda-forge matplotlib
conda install -c conda-forge tqdm
pip install opencv-python
pip install fvcore
pip install timm
pip install mmcv==1.3.11
pip install einops
pip install scikit-learn
pip install focal-loss-torch
pip install pandas
pip install seaborn
````
## Dataset preparation
Download the following components of the Cell Behavior Video Classification Challenge (CBVCC) from the [official website](https://www.immunemap.org/index.php/challenges-menu/cbvcc): <br>
[Train](https://www.dp-lab.info/cbvcc/data/training.zip) <br>
[Test Phase1](https://www.dp-lab.info/cbvcc/data/test_phase1.zip) <br>
[Test Phase2](https://www.dp-lab.info/cbvcc/data/test_phase2.zip) <br> 

## Training
Run:
````
python -u train.py
````
## Testing
Run:
````
python -u test.py
````

## Pre-trained weights
We initialize our Swin3D-B backbone with pre-trained weights on the SomethingSomething v2 dataset.<br>
Swin3D-B with Something-Something v2 pre-training: [Google Drive](https://drive.google.com/uc?export=download&id=1B14MhWCYm9eEy8MW6DqKqioZWkCvs0A0) <br>

Our model for the Cell Behavior Video Classification Challenge (CBVCC) is fine-tuned on the training set provided by the challenge. <br>
**Ours**: [Google Drive]([https://drive.google.com/file/d/](https://drive.google.com/uc?export=download&id=1njaLiE6Ee_aedqg-y7kYlMuweko5Wd_w) <br>

## Contact
If you have any inquiries or require assistance, please reach out to Jyoti Kini (jyoti.kini@ucf.edu).
