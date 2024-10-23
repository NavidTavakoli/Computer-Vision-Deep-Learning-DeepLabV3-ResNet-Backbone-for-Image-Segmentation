import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim import Adam
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.optim.lr_scheduler import CyclicLR


    

# Function to check if a directory exists
def directory_exists(path):
    return os.path.exists(path)

# Function to get sorted lists of image and mask paths
def get_files(directory):
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) if not f.startswith('.')], key=str.lower)

# Assuming all images and masks are stored in 'images_dir' and 'masks_dir' respectively
images_dir = "train/images"
masks_dir = "train/masks"

# Check if the folders exist
train_exists = directory_exists("train_split")
val_exists = directory_exists("val_split")
test_exists = directory_exists("test_split")

# If the folders do not exist, split and save the data
if not (train_exists and val_exists and test_exists):
    from sklearn.model_selection import train_test_split
    
    # Split data into train+val and test sets
    images = get_files(images_dir)
    masks = get_files(masks_dir)
    train_val_images, test_images, train_val_masks, test_masks = train_test_split(images, masks, test_size=0.15, random_state=42)

    # Split train+val into train and val sets
    train_images, val_images, train_masks, val_masks = train_test_split(train_val_images, train_val_masks, test_size=0.15, random_state=42) 

    # Print the number of images in each set
    print(f"Training images: {len(train_images)}, Validation images: {len(val_images)}, Test images: {len(test_images)}")

    # Function to save split folders with progress bar
    def save_split_folders(image_paths, mask_paths, folder_name):
        os.makedirs(os.path.join(folder_name, "images"), exist_ok=True)
        os.makedirs(os.path.join(folder_name, "masks"), exist_ok=True)

        for src_img, src_mask in tqdm(zip(image_paths, mask_paths), total=len(image_paths), desc=f"Copying {folder_name}"):
            shutil.copy(src_img, os.path.join(folder_name, "images", os.path.basename(src_img)))
            shutil.copy(src_mask, os.path.join(folder_name, "masks", os.path.basename(src_mask)))

    # Save the train, validation, and test splits
    save_split_folders(train_images, train_masks, "train_split")
    save_split_folders(val_images, val_masks, "val_split")
    save_split_folders(test_images, test_masks, "test_split")
else:
    print("Split folders already exist. Skipping data splitting.")

# Load data from existing split folders
train_images = get_files("train_split/images")
train_masks = get_files("train_split/masks")
val_images = get_files("val_split/images")
val_masks = get_files("val_split/masks")
test_images = get_files("test_split/images")
test_masks = get_files("test_split/masks")

class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)  # Ensuring inputs are in [0,1]
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')
        return dice_loss + BCE_loss
    
    
train_dataset = CustomDataset(train_images, train_masks, transform=transform)
val_dataset = CustomDataset(val_images, val_masks, transform=transform)
test_dataset = CustomDataset(test_images, test_masks, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

class TransformerSegmentationModel(nn.Module):
    def __init__(self, n_classes):
        super(TransformerSegmentationModel, self).__init__()
        # Initialize the backbone model, for example, DeepLabV3 with a ResNet50 backbone.
        self.model = deeplabv3_resnet50(pretrained=True)
        # Replace the classifier with a new one for the number of classes in the dataset.
        self.model.classifier[4] = nn.Conv2d(256, n_classes, kernel_size=(1, 1))

    def forward(self, x):
        return self.model(x)['out']
    





def train_one_epoch(epoch_index, dataloader, model, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(dataloader, desc=f'Epoch {epoch_index+1} [Training]'):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item()
    average_loss = total_loss / len(dataloader)
    return average_loss

def validate(epoch_index, dataloader, model, criterion, device):
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc=f'Epoch {epoch_index+1} [Validation]'):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_val_loss += loss.item()
    average_val_loss = total_val_loss / len(dataloader)
    return average_val_loss

# Initialize device, model, optimizers, and scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerSegmentationModel(n_classes=1).to(device)
optimizer = Adam(model.parameters(), lr=0.001)  # Higher learning rate for initial training
criterion = DiceBCELoss().to(device)
scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_up=5*len(train_loader), mode='triangular', cycle_momentum=False)

best_val_loss = float('inf')
early_stopping_patience = 10
patience_counter = 0

# Training and validation cycles
for epoch in range(100):
    train_loss = train_one_epoch(epoch, train_loader, model, optimizer, criterion, device, scheduler)
    val_loss = validate(epoch, val_loader, model, criterion, device)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Checkpoint the best model
    if val_loss < best_val_loss:
        print(f'Epoch {epoch+1}: New best model saved.')
        torch.save(model.state_dict(), 'DeepLabV3_Res50_New_Learning_rate_256.pth')
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(f'No improvement in {early_stopping_patience} epochs, stopping early.')
            break
