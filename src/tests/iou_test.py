import torch
import torchvision
from torchvision.utils import draw_bounding_boxes

from utils.misc import xywh2minmax, minmax2xywh
from utils.iou import intersection_over_union

if __name__ == "__main__":
    pred_bboxes = torch.tensor([
        [0, 0, 100, 100],
        [200, 200, 300, 300]
        ])
    target_bboxes = torch.tensor([
        [50, 50, 100, 100],
        [225, 200, 300, 300],
        ])
    
    print(pred_bboxes)
    print(minmax2xywh(pred_bboxes))

    # pred_bboxes = xywh2minmax(minmax2xywh(pred_bboxes))
    print(pred_bboxes)

    # img=torch.zeros((3, 300, 300), dtype=torch.uint8)
    # img=draw_bounding_boxes(img, pred_bboxes, width=3, labels=['0','0'], colors=(255,255,0))
    # img=draw_bounding_boxes(img, target_bboxes, width=3, labels=['1','1'], colors=(255,0,255))
    # img = torchvision.transforms.ToPILImage()(img)
    # img.show()

    # iou = intersection_over_union(pred_bboxes, target_bboxes)
    # print(iou)
