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
We initiliaze the Swin3D-B backbone, which is pre-trained on the SomethingSomething v2 dataset.<br>
Swin3D-B with Something-Something v2 pre-training: [Google Drive](https://drive.google.com/uc?export=download&id=1B14MhWCYm9eEy8MW6DqKqioZWkCvs0A0) <br>

Our final scores for Cell Behavior Video Classification Challenge (CBVCC) yield action classes. <br>
**Ours**: [Google Drive]() <br>

## Contact
If you have any inquiries or require assistance, please reach out to Jyoti Kini (jyoti.kini@ucf.edu).
