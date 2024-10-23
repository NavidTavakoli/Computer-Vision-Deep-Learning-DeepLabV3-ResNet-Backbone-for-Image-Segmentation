import torch
import torch.nn.functional as F
import os
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from Model_DeepLabV3_Resnet50_Enhanced_New import test_loader, TransformerSegmentationModel
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score



def binary_accuracy(preds, target):
    """Calculates accuracy for binary classification"""
    preds = torch.sigmoid(preds) > 0.5  # Apply sigmoid and threshold
    correct = (preds == target).float()  # Convert into float for division 
    acc = correct.sum() / correct.numel()
    return acc


def precision_recall_f1(preds, target):
    """Calculate precision, recall, and F1 score for binary classification."""
    preds = torch.sigmoid(preds) > 0.5
    target = target > 0.5
    
    TP = (preds & target).sum().float()
    FP = (preds & ~target).sum().float()
    FN = (~preds & target).sum().float()
    
    precision = TP / (TP + FP + 1e-7)
    recall = TP / (TP + FN + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    return precision.item(), recall.item(), f1.item()


def dice_score(preds, target, smooth=1e-7):
    """Calculates the Dice Score"""
    preds_bool = torch.sigmoid(preds) > 0.5  # Sigmoid to get probabilities, then threshold to get a binary mask
    target_bool = target > 0.5  # Assuming target is already a probability that needs to be thresholded
    
    # Convert to boolean tensors
    preds_bool = preds_bool.bool()
    target_bool = target_bool.bool()
    
    intersection = (preds_bool & target_bool).float().sum((1, 2))  # Perform bitwise AND on boolean tensors, then convert to float for summation
    union = preds_bool.float().sum((1, 2)) + target_bool.float().sum((1, 2))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()


def iou_score(preds, target, smooth=1e-7):
    """Calculates Intersection over Union (IoU)"""
    preds_bool = torch.sigmoid(preds) > 0.5  # Convert predictions to boolean based on threshold
    target_bool = target > 0.5  # Ensure target is also boolean
    
    # Convert to boolean tensors
    preds_bool = preds_bool.bool()
    target_bool = target_bool.bool()
    
    intersection = (preds_bool & target_bool).float().sum((1, 2))  # Perform bitwise AND on boolean tensors
    union = (preds_bool | target_bool).float().sum((1, 2))  # Perform bitwise OR on boolean tensors
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def mse_loss(preds, target):
    """Calculates Mean Squared Error"""
    preds = torch.sigmoid(preds)  # Apply sigmoid to get probabilities
    mse = F.mse_loss(preds, target, reduction='mean')
    return mse

def evaluate_model(model, dataloader, device, save_path='Test/Output'):
    model.eval()
    total_acc, total_dice, total_iou, total_mse, total_precision, total_recall, total_f1 = 0, 0, 0, 0, 0, 0, 0
    all_preds, all_targets = [], []
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    with torch.no_grad():
        for images, masks, image_names in tqdm(dataloader, desc="Evaluation Progress", unit="batch", leave=False):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            # Ensure masks are binary for all_targets collection
            masks_binary = (masks > 0.5).float()  # Convert to binary labels
            true_labels = masks_binary.cpu().numpy().flatten()  # Flatten for consistency with all_preds
            all_targets.extend(true_labels)  # Collect binary true labels for ROC and PR curves
            
            # Collect probabilities for all_preds
            preds_probs = torch.sigmoid(outputs).cpu().numpy().flatten()  # Probabilities
            all_preds.extend(preds_probs)
            
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            
            masks_resized = F.interpolate(masks, size=outputs.shape[2:], mode='bilinear', align_corners=False)
            
            acc = binary_accuracy(outputs, masks_resized)
            dice = dice_score(outputs, masks_resized)
            iou = iou_score(outputs, masks_resized)
            mse = mse_loss(outputs, masks_resized)
            precision, recall, f1 = precision_recall_f1(outputs, masks_resized)
            
            total_acc += acc.item()
            total_dice += dice.item()
            total_iou += iou.item()
            total_mse += mse.item()
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            
            # Save predicted masks
            preds_thresh = (torch.sigmoid(outputs) > 0.5).float()
            for j, pred in enumerate(preds_thresh):
                pred_img = pred.squeeze().cpu().numpy()  # Convert to numpy array
                pred_img = (pred_img * 255).astype(np.uint8)  # Convert to an 8-bit image
                original_image_name = image_names[j]
                img_path = os.path.join(save_path, f'mask_{original_image_name}.png')
                Image.fromarray(pred_img).save(img_path)
    
    # Calculate and save ROC and PR curves
    print("Unique labels (before ROC calculation):", np.unique(all_targets))  # Debug print to check labels
    fpr, tpr, _ = roc_curve(all_targets, all_preds)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(all_targets, all_preds)
    pr_auc = average_precision_score(all_targets, all_preds)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower left")
    
    plt.savefig(os.path.join(save_path, 'roc_pr_curves.png'))
    plt.close()
    
    # Calculating averages
    avg_acc = total_acc / len(dataloader)
    avg_dice = total_dice / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    avg_mse = total_mse / len(dataloader)
    avg_precision = total_precision / len(dataloader)
    avg_recall = total_recall / len(dataloader)
    avg_f1 = total_f1 / len(dataloader)
    
    return avg_acc, avg_dice, avg_iou, avg_mse, avg_precision, avg_recall, avg_f1, roc_auc, pr_auc

 


# Initialize and load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerSegmentationModel(n_classes=1).to(device)
model.load_state_dict(torch.load('DeepLabV3_Res50_Enhanced_New_256.pth', map_location=device))
save_path = 'Test/Output'  # Ensure this path is what you want, adjusted for clarity

# Run evaluation
avg_acc, avg_dice, avg_iou, avg_mse, avg_precision, avg_recall, avg_f1, roc_auc, pr_auc = evaluate_model(model, test_loader, device, save_path)
# Print results
print(f"Accuracy: {avg_acc:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}, MSE: {avg_mse:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")