from typing import Any
import numpy as np
import torch
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.model import Yolov1
from datasets.VOC_dataset import VOCDataset
# from utils.iou import intersection_over_union
# from utils.nms import non_maximum_suppression
# from utils.mAP import mean_average_percision
from losses.yolo_loss import YoloLoss


seed = 1004
torch.manual_seed(seed=seed)
torch.cuda.manual_seed(seed=seed)

LEARNING_RATE = 2e-5
DEVICE = torch.device("mps" if torch.has_mps else "cpu")
BATCH_SIZE = 32
STEPS = 10000
WEIGHT_DECAY = 0
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/lables"


def train_fn(train_loader, model, optimizer, loss_fn, steps=0):
    # loop = tqdm(train_loader, leave=True)
    mean_loss = []

    # for step, (x, y) in enumerate(train_loader):
        # x, y = x.to(DEVICE), y.to(DEVICE)
        # p = model(x)
        # loss = loss_fn(p, y)
        # mean_loss.append(loss.item())
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # print("Train Step: {}, Loss: {}".format(step, loss.item()))
        # loop.set_postfix(loss=loss.item())

    steps=steps
    iters = iter(train_loader)

    while True:
        try:
            image, target = next(iters)
        except StopIteration:
            iters = iter(train_loader)
            mean_loss = []


        image, target = image.to(DEVICE), target.to(DEVICE)
        pred = model(image)
        loss = loss_fn(pred, target)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Train Step: {}, Loss: {}".format(steps, np.mean(mean_loss)))
        steps+=1
        if steps == 10:
            state_dicts = {"steps": steps,
                            "parameters": model.state_dict(),
                            "optimizer": optimizer.state_dict()}
            torch.save(state_dicts, "./save_file.pth")
            break

    # print(f"Mean loss was {sum(mean_loss)/ len(mean_loss)}")

def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(device=DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    # if LOAD_MODEL:
        # load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    transforms = Compose([
        ToTensor(),
        Resize((448, 448), antialias=True)
    ])

    train_dataset =VOCDataset(
        "./data/VOC_dataset/trainval.txt", 
        "./data/VOC_dataset/images",
        "./data/VOC_dataset/labels",
        transform=transforms,
        steps=BATCH_SIZE*STEPS
    )
    test_dataset =VOCDataset(
        "./data/VOC_dataset/test.txt", 
        "./data/VOC_dataset/images",
        "./data/VOC_dataset/labels",
        transform=transforms
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True
    )

    test_dataset = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False
    )


    train_fn(
        train_loader=train_loader,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn)


if __name__ == "__main__":
    main()


