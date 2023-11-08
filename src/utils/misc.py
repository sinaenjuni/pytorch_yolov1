import torch

# minx = cx-w/2
# miny = cy-h/2
# maxx = cx+w/2
# maxy = cy+y/2
def xywh2minmax(bboxes):
    ret = torch.zeros_like(bboxes).to(torch.float32)
    ret[..., 0] = bboxes[..., 0] - bboxes[..., 2] /2 # minx
    ret[..., 1] = bboxes[..., 1] - bboxes[..., 3] /2 # miny
    ret[..., 2] = bboxes[..., 0] + bboxes[..., 2] /2 # maxx
    ret[..., 3] = bboxes[..., 1] + bboxes[..., 3] /2 # maxy
    return ret

# cx = (maxx - minx) / 2
# cy = (maxy - miny) / 2
# w = maxx - minx
# h = maxy - miny
def minmax2xywh(bboxes):
    ret = torch.zeros_like(bboxes).to(torch.float32)
    ret[..., 0] = (bboxes[..., 0] + bboxes[..., 2]) /2 # cx
    ret[..., 1] = (bboxes[..., 1] + bboxes[..., 3]) /2 # cy
    ret[..., 2] = bboxes[..., 2] - bboxes[..., 0]      # w
    ret[..., 3] = bboxes[..., 3] - bboxes[..., 1]      # h
    return ret



def get_cellbox_to_bboxes(cell_tensor, S=7):
    classes = torch.argmax(cell_tensor[...,:20], dim=-1, keepdim=True)

    bboxes_1, bboxes_2 = cell_tensor[...,21:25], cell_tensor[...,26:]
    best_bbox_conf, best_bbox_ind = torch.max(cell_tensor[...,[20, 25]], dim=3, keepdim=True)

    best_bboxes = (1-best_bbox_ind) * bboxes_1 + best_bbox_ind * bboxes_2
    cell_indices = torch.stack(torch.meshgrid(torch.arange(S), torch.arange(S), indexing='xy')).permute(1,2,0)[None, ...]
    best_bboxes[..., :2] = best_bboxes[..., :2] + cell_indices
    best_bboxes /= S
    return torch.cat([classes, best_bbox_conf, best_bboxes], dim=-1)