import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from ss_swin import SwinTransformer3D
from collections import OrderedDict
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
import random
import numpy as np
import torch
import matplotlib as mpl
import csv

# Set DPI for better quality in Jupyter
mpl.rcParams['figure.dpi'] = 150

def seed_everything(seed=42):
    """
    Seed all random number generators for reproducibility.
    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)

# Dataset class
class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, deterministic=False, num_frames=16):
        self.root_dir = root_dir
        self.transform = transform
        self.deterministic = deterministic
        self.num_frames = num_frames
        self.video_paths = []
        self.labels = []

        # Traverse the folder structure and store paths and labels
        for subject in os.listdir(root_dir):
            subject_path = os.path.join(root_dir, subject)
            if not os.path.isdir(subject_path):
                continue
            class_path = os.path.join(subject_path)
            if not os.path.isdir(class_path):
                continue
            for video in os.listdir(class_path):
                if video.endswith(".avi"):
                    self.video_paths.append(os.path.join(class_path, video))

    def __len__(self):
        return len(self.video_paths)

    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]

        # Read video and extract frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Ensure at least 'num_frames' frames are selected
        max_start_frame = total_frames - self.num_frames
        if self.deterministic:
            start_frame = idx % (max_start_frame + 1)  # Deterministic start frame based on index
        else:
            start_frame = random.randint(0, max_start_frame)

        # Calculate step size to evenly sample frames
        step_size = max(total_frames // self.num_frames, 1)

        for i in range(self.num_frames):
            frame_idx = start_frame + i * step_size
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # Explicitly seek to the frame
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()

        # Handle cases with fewer frames than num_frames
        if len(frames) < self.num_frames:
            frames.extend([frames[-1]] * (self.num_frames - len(frames)))

        video_tensor = torch.stack(frames, dim=0)
        return video_tensor, video_path
 
class SSModel(nn.Module):
    def __init__(self, num_classes, pretrained_weights_path=None):
        super(SSModel, self).__init__()
        self.swin3d_b_ss = SwinTransformer3D(embed_dim=128, 
                          depths=[2, 2, 18, 2], 
                          num_heads=[4, 8, 16, 32], 
                          patch_size=(2,4,4), 
                          window_size=(16,7,7), 
                          drop_path_rate=0.4, 
                          patch_norm=True)
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=1)
        self.linear = nn.Linear(self.swin3d_b_ss.layers[-1].blocks[-1].mlp.fc2.out_features, num_classes)

        # Train the earlier layers
        for param in self.swin3d_b_ss.parameters():
            param.requires_grad = True
        
        # Only the final layer will be trained
        for param in self.linear.parameters():
            param.requires_grad = True

        # Load external weights if specified
        if pretrained_weights_path:
            self.load_external_weights(pretrained_weights_path)

    def load_external_weights(self, weights_path):
        print(f"Loading weights from: {weights_path}")
    
        # Load the checkpoint
        checkpoint = torch.load(weights_path, map_location="cuda")
        
        # Extract the model state dict from the checkpoint
        external_state_dict = checkpoint['state_dict']  # Assuming the checkpoint contains 'model' key
        
        # Prepare a new state dict for the model
        new_state_dict = OrderedDict()

        # Fill the new_state_dict with the checkpoint weights, skipping the final 'linear' layer
        for k, v in external_state_dict.items():
            if 'backbone' in k:
                name = 'swin3d_b_ss.' + k[9:]
                new_state_dict[name] = v

        # Load the state dict into the model
        model_keys_before = set(self.state_dict().keys())
        missing_keys, unexpected_keys = self.load_state_dict(new_state_dict, strict=False)

        # Keys in the new state dict
        model_keys_after = set(new_state_dict.keys())

        # Count matched and unmatched keys
        matched_keys = model_keys_before.intersection(model_keys_after)
        total_keys_in_checkpoint = len(new_state_dict)
        total_keys_in_model = len(model_keys_before)
        matched_count = len(matched_keys)
        missing_count = len(missing_keys)
        unexpected_count = len(unexpected_keys)

        print("=> Weights loaded successfully!")
        print(f"Total weights in checkpoint: {total_keys_in_checkpoint}")
        print(f"Total weights in model: {total_keys_in_model}")
        print(f"Matched weights: {matched_count}")
        print(f"Missing weights: {missing_count}")
        print(f"Unexpected weights: {unexpected_count}")    

    def forward(self, x):
        mm_model = self.swin3d_b_ss(x)
        mm_pool = self.avgpool(mm_model)
        mm_pool = mm_pool.view(mm_pool.size(0), -1)
        mm_fc = self.linear(mm_pool)
        return mm_fc

def write_results_to_csv(output_csv, results):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(results)
    print(f"Results saved to {output_csv}")
           
# Make predictions
def predict(model, dataset, dataloader, criterion, checkpoint_path, output_csv):
    model.eval()
    
    # Load the best checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading best model checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        assert checkpoint['signature'] == "best_model_epoch", "Checkpoint signature mismatch!"
        # Load the saved checkpoint directly for comparison
        saved_model_state = checkpoint['model_state_dict']

        # Compare state dictionaries
        for param_tensor in model.state_dict():
            assert torch.equal(model.state_dict()[param_tensor], saved_model_state[param_tensor]), \
                f"Mismatch in parameter: {param_tensor}"
        print("Loaded model matches the saved checkpoint.")
    else:
        print(f"No checkpoint found at {checkpoint_path}. Proceeding with the current model.")

    epoch_loss, correct, total = 0, 0, 0
    all_probs, all_predictions, all_labels = [], [], []
    video_indices = []
    results = []

    with torch.no_grad():
        for i, (videos, video_paths) in enumerate(tqdm(dataloader, desc="Testing")):
            videos = videos.permute(0, 2, 1, 3, 4).cuda()

            outputs = model(videos)
            
            # Compute predictions and probabilities
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            predicted = outputs.argmax(dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_predictions.extend(predicted)
            
            video_indices.extend(range(i * dataloader.batch_size, (i + 1) * dataloader.batch_size))
            
            # Collect data for CSV
            for j, prob in enumerate(probs):
                video_id = os.path.basename(video_paths[j])
                results.append((video_id, round(prob, 2)))
    
    # Save to CSV
    write_results_to_csv(output_csv, results)

# Call the function at the start of your script
seed_everything(seed=42)

# Data transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets and loaders
root_dir = '/home/datasets/CBVCC'
test_dir = 'test_phase2'
test_dataset = VideoDataset(os.path.join(root_dir, test_dir), transform=transform, deterministic=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

num_classes = 2

train_lr = 3e-4
train_betas=(0.9, 0.999)
train_weight_decay = 0.05
epochs = 100

# Initialize model, optimizer, and loss
checkpoint_path = '/home/projects/CBVCC/checkpoints/best_model.pt'
output_csv = '/home/projects/CBVCC/outputs/submission_phase2.csv'

model = SSModel(num_classes).cuda()
optimizer = optim.Adam(model.parameters(), train_lr, train_betas, train_weight_decay)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=epochs)
criterion = nn.CrossEntropyLoss() 
predict(model, test_dataset, test_loader, criterion, checkpoint_path, output_csv)
    
