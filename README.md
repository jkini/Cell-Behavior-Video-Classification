# CellBehaviorVideoClassification

# Ensemble Modeling for Multimodal Visual Action Recognition
Official code repo for Ensemble Modeling for Multimodal Visual Action Recognition [ICIAP-W 2023 ${\color{red}Competition~Winner}$]
[Project](https://www.crcv.ucf.edu/research/projects/ensemble-modeling-for-multimodal-visual-action-recognition/) and 
[Arxiv](https://arxiv.org/pdf/2308.05430.pdf)

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
Download the following components of the Meccano dataset from the [official website](https://iplab.dmi.unict.it/MECCANO/challenge.html): <br>
[RGB frames](https://iplab.dmi.unict.it/sharing/MECCANO/MECCANO_RGB_frames.zip) <br>
[Depth frames](https://iplab.dmi.unict.it/sharing/MECCANO/MECCANO_Depth_frames.zip) <br>
[Action annotations](https://iplab.dmi.unict.it/sharing/MECCANO/MECCANO_action_annotations.zip) <br> 
Update config.py [`data_dir`] to reflect the dataset location.

## Training
We train individual modalities RGB and Depth. <br>

Update config.py [`train_run_id`, `train_modality`, `train_weights_dir`, `train_ss_wt_file`] to reflect the relevant details.<br>

Run:
````
python -u train.py
````
## Testing ( individual modalities OR ensemble)
1. Test individual modalities RGB and Depth. <br><br>
Update config.py [`test_wt_file`, `test_modality`] to reflect the relevant details.<br><br>
Run:
````
python -u test.py
````
2. Obtain class probabilities averaged from RGB and Depth pathways (${\color{red}Competition~Result}$).<br><br>
Update config.py [`test_wt_file_1`, `test_wt_file_2`] to reflect the relevant details.<br><br>
Run:
````
python -u test_mm.py
````

## Pre-trained weights
We initiliaze the Swin3D-B backbone, which is pre-trained on the SomethingSomething v2 dataset.<br>
Swin3D-B with Something-Something v2 pre-training: [Google Drive](https://drive.google.com/uc?export=download&id=1B14MhWCYm9eEy8MW6DqKqioZWkCvs0A0) <br>

Our final scores for Cell Behavior Video Classification Challenge (CBVCC) yield action classes. <br>
**Ours**: [Google Drive]() <br>

````
## Contact
If you have any inquiries or require assistance, please reach out to Jyoti Kini (jyoti.kini@ucf.edu).
