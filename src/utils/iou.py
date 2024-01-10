import torch
from utils.misc import xywh2minmax, minmax2xywh



def intersection_over_union(pred_bboxes, target_bboxes, box_format="minmax"):
    if box_format != "minmax":
        pred_bboxes = xywh2minmax(pred_bboxes)

    # minx, miny, maxx, maxy    
    minx = torch.max(pred_bboxes[..., 0], target_bboxes[..., 0])
    miny = torch.max(pred_bboxes[..., 1], target_bboxes[..., 1])
    maxx = torch.min(pred_bboxes[..., 2], target_bboxes[..., 2])
    maxy = torch.min(pred_bboxes[..., 3], target_bboxes[..., 3])
    intersection = (maxx - minx).clamp(0) * (maxy - miny).clamp(0)
    # print(intersection)

    # w: maxx - minx, h:maxy - miny
    preds_area =   torch.abs((pred_bboxes[..., 2] - pred_bboxes[..., 0]) * (pred_bboxes[..., 3] - pred_bboxes[..., 1]))
    targets_area = torch.abs((target_bboxes[..., 2] - target_bboxes[..., 0]) * (target_bboxes[..., 3] - target_bboxes[..., 1]))
    union = (preds_area + targets_area - intersection + 1e-6)
    # print(union)
    return intersection / union

if __name__ == "__main__":
    pred_bboxes = torch.tensor([
        [0, 0, 100, 100],
        [200, 200, 300, 300]
        ])
    target_bboxes = torch.tensor([
        [50, 50, 100, 100],
        [225, 200, 300, 300],
        ])
    
    pred_bboxes = xywh2minmax(minmax2xywh(pred_bboxes))
    print(pred_bboxes)
    import torchvision
    from torchvision.utils import draw_bounding_boxes
    img=torch.zeros((3, 300, 300), dtype=torch.uint8)
    img=draw_bounding_boxes(img, pred_bboxes, width=3, labels=['0','0'], colors=(255,255,0))
    img=draw_bounding_boxes(img, target_bboxes, width=3, labels=['1','1'], colors=(255,0,255))
    img = torchvision.transforms.ToPILImage()(img)
    img.show()


    iou = intersection_over_union(pred_bboxes, target_bboxes)
    print(iou)