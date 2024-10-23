import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
import rasterio
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet50



class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        """
        Initialize the dataset with paths to the images and masks, and any transformations.

        Parameters:
        - image_paths: A list of paths to the images.
        - mask_paths: A list of paths to the corresponding masks.
        - transform: A torchvision.transforms composition for image transformations.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get the item at the provided index.

        Parameters:
        - idx: The index of the item.

        Returns:
        - A tuple of the transformed image and mask tensors, and the image name.
        """
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Open the image file with rasterio
        with rasterio.open(image_path) as image_file:
            image_array = image_file.read([1, 2, 3])  # Read the first three bands (RGB)
            image_array = np.moveaxis(image_array, 0, -1)  # Reorder dimensions to HWC
        
        # Convert the Numpy array to a PIL Image for compatibility with torchvision transforms
        image = Image.fromarray(image_array)
        
        # Open the mask as a grayscale image
        mask = Image.open(mask_path).convert("L")
        
        # Apply the provided transformations to both the image and the mask
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)  # Ensure this transform does not alter spatial dimensions
        
        image_name = os.path.basename(image_path).split('.')[0]  # Extract the base name without file extension

        return image, mask, image_name

# Define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])



class TransformerSegmentationModel(nn.Module):
    def __init__(self, n_classes):
        super(TransformerSegmentationModel, self).__init__()
        # Initialize the backbone model, for example, DeepLabV3 with a ResNet50 backbone.
        self.model = deeplabv3_resnet50(pretrained=True)
        # Replace the classifier with a new one for the number of classes in the dataset.
        self.model.classifier[4] = nn.Conv2d(256, n_classes, kernel_size=(1, 1))

    def forward(self, x):
        return self.model(x)['out']






# Function to get sorted lists of image and mask paths
def get_files(directory):
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) if not f.startswith('.')], key=str.lower)

test_images = get_files("test_split/images")
test_masks = get_files("test_split/masks")
test_dataset = CustomDataset(test_images, test_masks, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
