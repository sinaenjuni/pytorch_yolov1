import torch
import torch.nn as nn
from utils.iou import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20): 
        # S: Size of grids, B: Number of boxes, C: Number of classes
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduce="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, targets):
        # (Batch, 7, 7, 20 + 5(prob, cx, cy, w, h) * 2)
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        iou_b1 = intersection_over_union(predictions[..., 21:25], targets[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], targets[..., 21:25])
        
        # ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        ious = torch.stack([iou_b1, iou_b2])
        iou_maxes, best_box = torch.max(ious, dim=0) # index of best iou: [0, 1]
        best_box = best_box.unsqueeze(-1)
        # Iobj_i, 0~19: classes idx, 
        exists_box = targets[..., 20].unsqueeze(-1) # (N, S, S, 1)

        # FOR BOX COORDINATES
        # cx, cy, w, h
        # exists_box 박스가 존재하는 경우에만 손실 함수를 계산하기 위해서
        # 두 박스 중 하나가 높은 iou라면, 높은 iou
        box_predictions = exists_box * (
            (
                best_box * predictions[..., 26:30] +
                (1 - best_box) * predictions[..., 21:25]
            )
        )
        box_targets = exists_box * targets[..., 21:25]
        
        # sqrt()제곱근을 구한 후, 원래 부호를 유지하기 위해 torch.sign()을 사용
        # 음수를 예측하는 경우와 0이 되어 무한대가 되는 상황을 막기 위해
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * \
                                    torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6 )) 
    
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # (N, S, S, 4) -> (N * S * S, 4)
        box_loss = self.mse(torch.flatten(box_predictions, end_dim=-2),
                            torch.flatten(box_targets, end_dim=-2))
        
        # FOR OBJECT LOSS
        pred_box = (best_box * predictions[..., 25:26] +
                    (1-best_box) * predictions[..., 20:21])
        # [N*S*S]
        object_loss = self.mse(torch.flatten(exists_box * pred_box),
                               torch.flatten(exists_box * targets[..., 20:21]))

        # FOR NO OBJECTLOSS
        no_object_loss = self.mse(torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
                                  torch.flatten((1 - exists_box) * targets[..., 20:21], start_dim=1))
        no_object_loss += self.mse(torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
                                   torch.flatten((1 - exists_box) * targets[..., 20:21], start_dim=1))

        # FOR CLASS LOSS
        # (N * S * S, 20)
        class_loss = self.mse(torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
                              torch.flatten(exists_box * targets[..., :20], end_dim=-2))

        loss = (self.lambda_coord * box_loss # First two rows of loss in paper
                + object_loss
                + self.lambda_noobj * no_object_loss
                + class_loss)

        return loss
    

if __name__ == "__main__":
    from datasets.VOC_dataset import VOCDataset
    from torchvision.transforms import Compose, Resize, Normalize, ToTensor
    transform = Compose([
        ToTensor(),
        Resize((448,448), antialias=True),
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