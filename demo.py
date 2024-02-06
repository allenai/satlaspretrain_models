# This demostrates how to finetune a SatlasPretrain Sentinel-2 model on the EuroSAT classification task.
import io
import os
import torch
import zipfile
import requests
import torch.nn
import torchvision
from torch.utils.data import Dataset, DataLoader

import satlaspretrain_models


# Only go through the downloading and unzipping process if it hasn't been done before.
if not os.path.exists('EuroSAT_RGB/'):
    # Download the EuroSAT_RGB dataset. This is a zip file.
    zip_file_url = 'https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1'

    # Send a GET request to the EuroSAT_RGB Zenodo download URL.
    response = requests.get(zip_file_url)

    # Check if the request was successful.
    if response.status_code == 200:
        # Use BytesIO for the zip file content.
        zip_file = io.BytesIO(response.content)
        
        # Open the zip file. You should see the EuroSAT_RGB/ folder in this directory.
        with zipfile.ZipFile(zip_file) as zfile:
            # Extract all the contents into the current directory
            zfile.extractall('.')
            print("Zip file extracted successfully.")
    else:
        print(f"Failed to download the zip file. Status code: {response.status_code}")

# Define a dataset class that takes in the path to the EuroSAT_RGB path and returns datapoints
# with an image and a target class. 
class Dataset(Dataset):
    def __init__(self, dataset_path, val=False):

        self.datapoints = []

        # The subdirectories in the dataset folder are named by class.
        classes = os.listdir(dataset_path)
        print("EuroSAT classes:", classes)

        # Create a mapping from class label to a unique integer
        cls_to_int = {label: idx for idx, label in enumerate(set(classes))}

        # For each class, use 80% of the images for training and the rest for validation.
        for cls in classes:
            cls_int = cls_to_int[cls]
            cls_imgs = os.listdir(dataset_path + '/' + cls + '/')

            if val:
                cls_datapoints = [(dataset_path + '/' + cls + '/' + img, cls_int) for img in cls_imgs[2400:]]
            else:
                cls_datapoints = [(dataset_path + '/' + cls + '/' + img, cls_int) for img in cls_imgs[:2400]]
            
            self.datapoints.extend(cls_datapoints)
        print("Loaded ", len(self.datapoints), " datapoints.")

    def __getitem__(self, idx):
        img_path, cls_int = self.datapoints[idx]
        img = torchvision.io.read_image(img_path)  # load image directly into a [3, 64, 64] tensor
        img = img.float() / 255  # normalize input to be between 0-1
        target = torch.tensor(cls_int)  # convert class index into a torch tensor
        return img, target
    
    def __len__(self):
        return len(self.datapoints)


# Experiment arguments.
device = torch.device('cuda')
num_epochs = 1000
criterion = torch.nn.CrossEntropyLoss()
val_step = 10  # evalaute every val_step epochs

# Initialize the train and validation datasets.
train_dataset = Dataset('EuroSAT_RGB/')
val_dataset = Dataset('EuroSAT_RGB/', val=True)

# Dataloaders.
train_dataloader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=16
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=16
)

# Initialize a pretrained model, using the SatlasPretrain single-image Swin-v2-Base Sentinel-2 image model weights
# with a classification head with num_categories=10, since EuroSAT has 10 classes.
weights_manager = satlaspretrain_models.Weights()
model = weights_manager.get_pretrained_model("Sentinel2_SwinB_SI_RGB", fpn=True, head=satlaspretrain_models.Head.CLASSIFY, num_categories=10)
model = model.to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training loop.
for epoch in range(num_epochs):
    print("Starting Epoch...", epoch)

    for data, target in train_dataloader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)

        loss = criterion(output, target)
        print("Train Loss = ", loss)

        loss.backward()
        optimizer.step()

    if epoch % val_step == 0:
        model.eval()

        for val_data, val_target in val_dataloader:
            val_data = val_data.to(device)
            val_target = val_target.to(device)

            val_output = model(val_data)

            val_loss = criterion(val_output, val_target)
            val_accuracy = (val_output.argmax(dim=1) == val_target).float().mean().item()

            print("Validation accuracy = ", val_accuracy)

