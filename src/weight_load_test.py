import torch
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from torch.utils.data import DataLoader

from models.model import Yolov1
from utils.mAP import mean_average_percision
from datasets.VOC_dataset import VOCDataset


BATCH_SIZE = 4
NUM_WORKERS = 1
PIN_MEMORY=True
DEVICE = torch.device("mps" if torch.has_mps else "cpu")
# DEVICE = torch.device("cpu")


steps, model_state, optimizer_state = torch.load("./save_file.pth").values()

model = Yolov1(
    in_channels=3,
    split_size = 7,
    num_boxes=2,
    num_classes=20
).to(DEVICE)
model.load_state_dict(model_state)


transforms = Compose([
    ToTensor(),
    Resize((448, 448), antialias=True)
])

dataset =VOCDataset(
    "./data/VOC_dataset/8samples.txt", 
    "./data/VOC_dataset/images",
    "./data/VOC_dataset/labels",
    transform=transforms
)

loader = DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    pin_memory=PIN_MEMORY,
    shuffle=True,
    drop_last=True
)



for images, targets in loader:
    images, targets = images.to(DEVICE), targets.to(DEVICE)
    pred = model(images)

    print(pred.reshape(-1, 7, 7, 30))
    print(targets.shape)