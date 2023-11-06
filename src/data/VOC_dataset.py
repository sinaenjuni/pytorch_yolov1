import torch
import os
# import pandas as pd
from PIL import Image
import cv2
import numpy as np

import sys

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, txt_file, image_dir, label_dir, S=7, B=2, C=20, transform=None):
        self.file_list = []
        with open(txt_file, "r") as f:
            for line in f.readlines():
                self.file_list += [line.strip()]

        self.image_dir = image_dir
        self.label_dir = label_dir

        # self.images = []
        # self.labels = []
        # for file_id in self.file_list:
        #     self.images += [Image.open("{}/{}.jpg".format(self.image_dir, file_id))]
        #     self.labels += [open("{}/{}.txt".format(self.label_dir, file_id)).readlines()]
        #     print(self.labels)
        #     exit()


        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        target_file = self.file_list[index]
        boxes = None
        with open("{}/{}.txt".format(self.label_dir, target_file), 'r') as f:
            boxes = [
                list(map(np.float32, labels.replace("\n", "").split())) 
                for labels in f.readlines()
            ]

        # img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        img_path = "{}/{}.jpg".format(self.image_dir, target_file)
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)
        
        if self.transform is not None:
            image = self.transform(image)

        #[..., 20] objectness, 2
        # cx, cy: 0.5, 0.6 => 3.5, 4.2 => 0.5, 0.2
        label_matrix = np.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            cls, cx, cy, w, h = box.tolist()
            cls = int(cls)

            i = int(self.S * cy)
            j = int(self.S * cx)

            x_cell = self.S * cx - j
            y_cell = self.S * cy - i

            width_cell, height_cell = (
                self.S * w,
                self.S * h,
            )

            if label_matrix[i, j, 20] == 0:
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                label_matrix[i, j, 20] = 1 # prob is 1
                label_matrix[i, j, cls] = 1
                label_matrix[i, j, 21:25] = box_coordinates
        return image, label_matrix 


if __name__ == "__main__":
    from torchvision.transforms import Compose, Resize, Normalize, ToTensor
    transform = Compose([
        ToTensor(),
        Resize((448,448)),
        # Normalize(
        #     (.5,.5,.5),
        #     (.5,.5,.5))
    ])
    dataset = VOCDataset(
        "./data/VOC_dataset/8samples.txt", 
        "./data/VOC_dataset/images",
        "./data/VOC_dataset/labels",
        transform=transform)
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=4,
        num_workers=1,
        pin_memory=True,
        shuffle=False,
        drop_last=True
    )
    imgs, targets = next(iter(train_loader))
    print(imgs)

    # img, target = dataset[0]
    # print(img)
#     print(img.shape, target.shape)

#     import numpy as np
#     np_img = img.permute(1,2,0).numpy()
#     np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

#     np_target = target.cpu().numpy()
    
#     S = 7
#     WIDTH, HEIGHT, _ = np_img.shape
#     i, j = np.nonzero(np_target[...,20])
    
#     bboxes = np_target[i, j, 21:25]
#     bboxes[:, 0] = (bboxes[:, 0] + j) / S
#     bboxes[:, 1] = (bboxes[:, 1] + i) / S
#     bboxes[:, 2] = bboxes[:, 2] / S
#     bboxes[:, 3] = bboxes[:, 3] / S
#     print(bboxes)

#     bboxes = bboxes * np.array((WIDTH, HEIGHT, WIDTH, HEIGHT))
#     bboxes = bboxes.astype(np.uint16)
#     # bboxes[:, 1] *= HEIGHT
#     print(bboxes)

#     for cx, cy, w, h in bboxes:
#         # print(cx, cy)
#         cv2.circle(np_img, tuple(map(int, (cx, cy))), 15, (128,255,0), -1, cv2.LINE_AA)


#     ret = np.zeros_like(bboxes)
#     ret[:,0] = bboxes[:,0] - bboxes[:,2]/2
#     print(bboxes[:,2]/2)
#     ret[:,1] = bboxes[:,1] - bboxes[:,3]/2
#     ret[:,2] = bboxes[:,0] + bboxes[:,2]/2
#     ret[:,3] = bboxes[:,1] + bboxes[:,3]/2
#     # ret[:,2] = bboxes[:,2]
#     # ret[:,3] = bboxes[:,3]
#     print(ret)
#     for minx, miny, maxx, maxy in ret:
#         cv2.rectangle(np_img, (minx, miny), (maxx, maxy), (255,0,0), 2, cv2.LINE_AA)


#     cv2.imshow("np_img", np_img)
#     cv2.waitKey()

#     sys.exit()
    # bboxes = np_target[x, y, 21:25] 
    
    
    # ret = np.zeros_like(bboxes)
    # ret[:,0] = bboxes[:,0] - bboxes[:,2]/2
    # ret[:,1] = bboxes[:,1] - bboxes[:,3]/2
    # # ret[:,2] = bboxes[:,0] + bboxes[:,2]/2
    # # ret[:,3] = bboxes[:,1] + bboxes[:,3]/2
    # ret[:,2] = bboxes[:,2]
    # ret[:,3] = bboxes[:,3]

    # ret = ret * np.array([w, h, w, h])
    # ret = ret.astype(np.uint16)
    # print(ret)

    # for minx, miny, maxx, maxy in ret:
    #     pass
        # cv2.rectangle(np_img, (minx, miny), (maxx, maxy), (255,0,0), 2, cv2.LINE_AA)
    # cv2.imshow("np_img", np_img)
    # cv2.waitKey()

