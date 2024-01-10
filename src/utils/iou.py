import torch

def intersection_over_union(pred_bboxes, target_bboxes):
    if not torch.all(pred_bboxes[..., 0] < pred_bboxes[..., 2]) and \
        not torch.all(pred_bboxes[..., 1] < pred_bboxes[..., 3]):
        raise Exception("pred_bboxes box is not minmax type")

    if not torch.all(target_bboxes[..., 1] < target_bboxes[..., 3]) and \
        not torch.all(target_bboxes[..., 1] < target_bboxes[..., 3]):
        raise Exception("target_bboxes box is not minmax type")

    # minx, miny, maxx, maxy    
    minx = torch.max(pred_bboxes[..., 0], target_bboxes[..., 0])
    miny = torch.max(pred_bboxes[..., 1], target_bboxes[..., 1])
    maxx = torch.min(pred_bboxes[..., 2], target_bboxes[..., 2])
    maxy = torch.min(pred_bboxes[..., 3], target_bboxes[..., 3])
    intersection = (maxx - minx).clamp(0) * (maxy - miny).clamp(0)

    # w: maxx - minx, h:maxy - miny
    preds_area =   torch.abs((pred_bboxes[..., 2] - pred_bboxes[..., 0]) * (pred_bboxes[..., 3] - pred_bboxes[..., 1]))
    targets_area = torch.abs((target_bboxes[..., 2] - target_bboxes[..., 0]) * (target_bboxes[..., 3] - target_bboxes[..., 1]))
    union = (preds_area + targets_area - intersection + 1e-6)
    return intersection / union