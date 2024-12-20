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
import hashlib

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

def hash_data(data):
    """
    Generate a hash for the dataset.
    Args:
        data: Dataset or dataloader.
    Returns:
        A hash string for the data.
    """
    data_hash = hashlib.md5()
    for inputs, labels, video_paths in data:
        data_hash.update(torch.cat([inputs.view(-1), labels.view(-1)]).numpy().tobytes())
    return data_hash.hexdigest()

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
            for label in os.listdir(subject_path):
                class_path = os.path.join(subject_path, label)
                if not os.path.isdir(class_path):
                    continue
                for video in os.listdir(class_path):
                    if video.endswith(".avi"):
                        self.video_paths.append(os.path.join(class_path, video))
                        self.labels.append(int(label))

    def __len__(self):
        return len(self.video_paths)

    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

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
        return video_tensor, label, video_path

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
    
# Validation function
def validate(model, dataloader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for videos, labels, _ in tqdm(dataloader):
            videos = videos.permute(0, 2, 1, 3, 4).cuda()  # Convert to (B, C, T, H, W)
            labels = labels.cuda()

            outputs = model(videos)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            # Compute predictions and probabilities
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            predicted = outputs.argmax(dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()

            correct += (predicted == labels_np).sum()
            total += labels.size(0)

    avg_loss = val_loss / len(dataloader)
    accuracy = 100. * correct / total
    print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy
        
def train_with_validation(
    model, train_loader, val_loader, optimizer, scheduler, criterion, epochs=100, patience=100, delta=0.01, save_dir="./checkpoints"
):
    os.makedirs(save_dir, exist_ok=True)  # Ensure checkpoint directory exists
    best_val_accuracy = 0  # Initialize best validation accuracy
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for videos, labels, _ in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            videos = videos.permute(0, 2, 1, 3, 4).cuda()  # Convert to (B, C, T, H, W)
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = epoch_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Validation phase
        val_loss, val_accuracy = validate(model, val_loader, criterion)

        # Checkpointing every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f"epoch_{epoch+1}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'signature': "model_epoch"
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}: {checkpoint_path}")

        # Save best model based on validation accuracy improvement
        if val_accuracy > best_val_accuracy + delta:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_epoch = epoch + 1  # Update the best epoch
            # Save a signature
            torch.save({
                'model_state_dict': model.state_dict(),
                'signature': "best_model_epoch"
            }, os.path.join(save_dir, "best_model.pt"))
            print(f"Best model updated at Epoch {best_epoch} with Validation Accuracy: {val_accuracy:.2f}% and Loss: {val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"Validation accuracy did not improve by at least {delta:.2f}. Patience counter: {patience_counter}"
            )

        # Early stopping
        if patience_counter >= patience:
            print("Early stopping triggered. Stopping training.")
            break
    
    print(
        f"Training completed. Best model was saved at epoch {best_epoch} with Validation Accuracy: {best_val_accuracy:.2f}% and Loss: {best_val_loss:.4f}"
    )

# Call the function at the start of your script
seed_everything(seed=42)

# Data transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets and loaders
root_dir = '/home/datasets/CBVCC'
train_dir = 'training'
train_dataset = VideoDataset(os.path.join(root_dir, train_dir) , transform=transform, deterministic=False)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)

val_dir = 'test_phase1'
val_dataset = VideoDataset(os.path.join(root_dir, val_dir), transform=transform, deterministic=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

num_classes = len(set(train_dataset.labels))

train_lr = 3e-4
train_betas=(0.9, 0.999)
train_weight_decay = 0.05
epochs = 100

# Initialize model, optimizer, and loss
checkpoint_path = '/home/projects/CBVCC/checkpoints/best_model.pt'
output_csv = '/home/projects/CBVCC/outputs/submission.csv'
ss_wt_file = '/home/projects/CBVCC/checkpoints/swin_base_patch244_window1677_sthv2.pth'

model = SSModel(num_classes, pretrained_weights_path=ss_wt_file).cuda()
optimizer = optim.Adam(model.parameters(), train_lr, train_betas, train_weight_decay)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=epochs)
criterion = nn.CrossEntropyLoss() 

# Train with validation
train_with_validation(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs)
    
