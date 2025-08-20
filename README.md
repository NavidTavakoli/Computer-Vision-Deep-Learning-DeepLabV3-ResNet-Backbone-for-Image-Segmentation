[![Preprint DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16906917.svg)](https://doi.org/10.5281/zenodo.16906917)


# DeepLabV3 with ResNet50 Backbone for Image Segmentation


This repository implements a semantic segmentation pipeline using a pretrained DeepLabV3 model with a ResNet50 backbone, customized for a binary segmentation task on RGB images.

## DeepLabV3 Architecture Overview
DeepLabV3 is a powerful semantic segmentation model designed to accurately label each pixel in an image. It is particularly well-known for its ability to capture multi-scale context using Atrous Spatial Pyramid Pooling (ASPP), which employs multiple parallel atrous (dilated) convolutions with different dilation rates to capture features at multiple scales without reducing resolution.

**Key components of DeepLabV3:**
- Backbone CNN (e.g., ResNet50): extracts hierarchical feature maps from the input image.
- Atrous Convolutions: dilated convolutions applied in the backbone and ASPP to enlarge receptive fields without losing spatial resolution.
- ASPP Module: parallel atrous convolutions with different dilation rates combined to capture multi-scale information.
- Decoder / Classifier Head: processes the features and upsamples to generate the final pixel-wise class predictions (segmentation map).

## Model Setup: DeepLabV3 with ResNet50 Backbone

I used the pretrained DeepLabV3 model with a ResNet50 backbone from torchvision. This means:

- ResNet50 Backbone:
  - A residual network with 50 layers deep.
  - Consists mainly of convolutional layers organized into 4 main blocks (also called stages), each containing multiple bottleneck residual blocks.
  - Each bottleneck block has 3 convolutional layers: 1×1 conv (reducing channels), 3×3 conv, and 1×1 conv (restoring channels).
  - The residual connections help gradients flow deeper, enabling training of very deep networks.
  - Total number of convolutional layers in ResNet50 is 49 conv layers + 1 fully connected layer at the end (which is discarded when used as a backbone).

- Adaptations for Segmentation:
  - The fully connected (FC) layer at the end of ResNet50 is removed.
  - Atrous convolutions replace some strides to keep spatial resolution higher.
  - The ASPP module is attached on top of the backbone output feature maps.
  - The classifier head is a few convolutional layers to convert the ASPP output into segmentation logits.
 
## Technical Architecture Details

- Backbone: ResNet50 pretrained on ImageNet, truncated before the classification head.
- ASPP Module: applies multiple parallel atrous convolutions with different dilation rates (usually 6, 12, 18), plus global average pooling, concatenates results.
- Final Classifier Layer:
  - In torchvision’s DeepLabV3, the classifier is a small sequential module ending with a `nn.Conv2d` layer outputting `n_classes` channels.
  - In your code, this last `nn.Conv2d` layer is replaced to output 1 channel for binary segmentation.

- Number of layers in your model:
  - ResNet50 backbone: 50 layers total.
  - ASPP module: 5 parallel atrous conv layers + global average pooling.
  - Final classifier conv layer: 1 convolutional layer with 1×1 kernel producing the final output.
 
- Loss function:
  - You used `BCEWithLogitsLoss`, a combination of `sigmoid` activation and `binary cross-entropy` loss.
  - This is appropriate for binary segmentation and is numerically stable because it applies sigmoid inside the loss.
 
- No fully connected layers in my segmentation head because fully connected layers are removed to maintain spatial structure for pixel-wise predictions.

**Notice**

In semantic segmentation models like DeepLabV3, the goal is to produce a prediction for each pixel, so the spatial structure of the feature maps must be preserved throughout the network.
Fully connected (FC) layers—like those in typical classification networks—flatten the spatial dimensions into a 1D vector, which discards all spatial information. This makes FC layers unsuitable for segmentation heads, where you need to keep the 2D spatial layout.
Instead, segmentation models use convolutional layers only (often 1×1 convolutions at the end) so that the output remains a 2D feature map with a channel dimension representing class scores per pixel.

## Dataset Preparation and Splitting

```bash 
def get_files(directory):
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) if not f.startswith('.')], key=str.lower)
```
- This helper function lists all files in a directory excluding hidden files, and sorts them alphabetically. It ensures consistent order of images and masks.

```bash 
images_dir = "train/images"
masks_dir = "train/masks"

images = get_files(images_dir)
masks = get_files(masks_dir)
```
- Input images and masks are stored separately, paths are loaded using the above function.
```bash 
train_val_images, test_images, train_val_masks, test_masks = train_test_split(images, masks, test_size=0.15, random_state=42)
train_images, val_images, train_masks, val_masks = train_test_split(train_val_images, train_val_masks, test_size=0.15, random_state=42)
```
- The dataset is split into `train/validation/test` sets with 70% train, 15% validation, and 15% test sizes using train_test_split from scikit-learn. Random seed ensures reproducibility.

```bash 
def save_split_folders(image_paths, mask_paths, folder_name):
    os.makedirs(os.path.join(folder_name, "images"), exist_ok=True)
    os.makedirs(os.path.join(folder_name, "masks"), exist_ok=True)

    for src_img, src_mask in tqdm(zip(image_paths, mask_paths), total=len(image_paths), desc=f"Copying {folder_name}"):
        shutil.copy(src_img, os.path.join(folder_name, "images", os.path.basename(src_img)))
        shutil.copy(src_mask, os.path.join(folder_name, "masks", os.path.basename(src_mask)))
```
This function saves the splits into separate directories (`train_split/`, `val_split/`, `test_split/`) by copying images and masks accordingly, showing a progress bar.


## Custom Dataset Class and Transforms

```bash 
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
```
- A PyTorch Dataset class that loads images (as RGB) and masks (as grayscale), applying the specified transforms.
- The transform converts PIL images to tensors and resizes them to 256×256.

```bash 
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

```

- Resize and tensor conversion is applied to both images and masks for consistency and batch processing.

## Model Definition: DeepLabV3 with ResNet50 Backbone

```bash 
from torchvision.models.segmentation import deeplabv3_resnet50

class TransformerSegmentationModel(nn.Module):
    def __init__(self, n_classes):
        super(TransformerSegmentationModel, self).__init__()
        self.model = deeplabv3_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(256, n_classes, kernel_size=(1, 1))

    def forward(self, x):
        return self.model(x)['out']
```

- Uses torchvision’s pretrained DeepLabV3 with ResNet50 backbone.
- The final classifier’s last convolution layer (index 4) is replaced to output the correct number of classes (`n_classes=1` for binary segmentation).
- The forward method returns the segmentation map logits from the `out` key.

## Training and Validation Loops

```bash 
def train_one_epoch(epoch_index, dataloader, model, optimizer, criterion, device):
    model.train()
    train_loss = 0.0
    for images, masks in tqdm(dataloader, desc=f'Epoch {epoch_index+1} [Training]'):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(dataloader)
```

- Sets the model to training mode.
- Loops over batches: moves data to GPU if available, computes outputs, calculates loss, backpropagates, and updates model weights.
- Accumulates loss to compute average training loss per epoch.

```bash 
def validate(epoch_index, dataloader, model, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc=f'Epoch {epoch_index+1} [Validation]'):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    return val_loss / len(dataloader)
```

- Switches model to evaluation mode.
- Disables gradient calculations for speed and memory efficiency.
- Computes and accumulates validation loss.

## Training Setup and Early Stopping

```bash 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerSegmentationModel(n_classes=1).to(device)
optimizer = Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
```

- Device is set dynamically (GPU if available).
- Adam optimizer is used with an initial learning rate of 1e-4.
- Binary cross-entropy loss with logits is chosen for binary mask prediction.
- Cosine annealing learning rate scheduler reduces learning rate gradually to improve convergence.

```bash 
best_val_loss = float('inf')
early_stopping_patience = 20
patience_counter = 0

for epoch in range(100):
    train_loss = train_one_epoch(epoch, train_loader, model, optimizer, criterion, device)
    val_loss = validate(epoch, val_loader, model, criterion, device)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    scheduler.step()

    if val_loss < best_val_loss:
        print(f'Epoch {epoch+1}: New best model saved.')
        torch.save(model.state_dict(), 'DeepLabV3_Res50_Enhanced_New_256.pth')
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(f'No improvement in {early_stopping_patience} epochs, stopping early.')
            break
```

- Runs up to 100 epochs with early stopping if no validation loss improvement after 20 epochs.
- Saves the model weights whenever a new best validation loss is found.
- Scheduler steps after each epoch to adjust learning rate.
- Training and validation loss values are printed every epoch for monitoring.



---

# Model Evaluation Script for DeepLabV3-ResNet50 Segmentation

This script evaluates the performance of a binary semantic segmentation model based on DeepLabV3 with a ResNet50 backbone. It computes multiple metrics on the test dataset, saves predicted masks as images, and plots ROC and Precision-Recall curves.

---

## Features

- Computes pixel-wise **Accuracy**, **Dice Score**, **Intersection over Union (IoU)**, **Mean Squared Error (MSE)**, **Precision**, **Recall**, and **F1 Score**.
- Generates and saves **predicted masks** as PNG images.
- Calculates and plots **ROC Curve** and **Precision-Recall Curve** along with their AUC scores.
- Supports GPU acceleration if available.

---

## Functions

### Metrics

- `binary_accuracy(preds, target)`:  
  Calculates pixel-wise accuracy for binary predictions.

- `precision_recall_f1(preds, target)`:  
  Calculates precision, recall, and F1 score based on thresholded predictions.

- `dice_score(preds, target)`:  
  Computes Dice coefficient (overlap measure) between prediction and ground truth.

- `iou_score(preds, target)`:  
  Computes Intersection over Union (IoU) between prediction and ground truth.

- `mse_loss(preds, target)`:  
  Calculates Mean Squared Error between predicted probabilities and targets.

---

## Main Evaluation Function

### `evaluate_model(model, dataloader, device, save_path='Test/Output')`

- Runs model in evaluation mode without gradients.
- Iterates over batches from the dataloader.
- Calculates all metrics per batch.
- Saves predicted binary masks to disk.
- Collects prediction probabilities and true labels for ROC and PR curve computation.
- Plots and saves ROC and Precision-Recall curves.
- Returns average metrics over the entire dataset:
  - Accuracy, Dice, IoU, MSE, Precision, Recall, F1, ROC AUC, PR AUC.

---

## Usage Example

```python
import torch
from Model_DeepLabV3_Resnet50_Enhanced_New import test_loader, TransformerSegmentationModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerSegmentationModel(n_classes=1).to(device)
model.load_state_dict(torch.load('DeepLabV3_Res50_Enhanced_New_256.pth', map_location=device))
save_path = 'Test/Output'

avg_acc, avg_dice, avg_iou, avg_mse, avg_precision, avg_recall, avg_f1, roc_auc, pr_auc = evaluate_model(model, test_loader, device, save_path)

print(f"Accuracy: {avg_acc:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}, MSE: {avg_mse:.4f}")
print(f"Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
```


## Requirements

- Python 3.x  
- PyTorch  
- NumPy  
- Pillow  
- Matplotlib  
- scikit-learn  
- tqdm  


## Notes

- Predictions are thresholded at **0.5** after applying sigmoid activation.  
- Masks are resized to match output spatial dimensions before metric calculation.  
- Predicted masks are saved as **8-bit PNG images** for visualization.  
- ROC and Precision-Recall curves are saved as **roc_pr_curves.png** in the output folder.  


## Evaluation Metrics

- **Accuracy:**  
  Measures the ratio of correctly predicted pixels (both positives and negatives) to the total number of pixels.

- **Precision:**  
  Proportion of true positive pixels among all pixels predicted as positive.

- **Recall:**  
  Proportion of true positive pixels detected out of all actual positive pixels.

- **F1 Score:**  
  Harmonic mean of precision and recall, balancing both metrics.

- **Dice Score:**  
  Measures overlap between predicted mask and ground truth, focusing on positive prediction accuracy.

- **Intersection over Union (IoU):**  
  Ratio of the intersection area to the union area between predicted and ground truth masks.

- **Mean Squared Error (MSE):**  
  Average squared difference between predicted probabilities and actual labels, reflecting prediction accuracy.



  ### Note on Model Version

This implementation uses the base **DeepLabV3** model with a ResNet-50 backbone from torchvision. While effective, other versions like **DeepLabV3+** or newer segmentation architectures may potentially yield better results. Users are encouraged to experiment with these alternative models to improve performance depending on their specific datasets and tasks.

Author: Navid
