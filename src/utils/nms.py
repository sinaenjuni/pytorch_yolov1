import torch
from iou import intersection_over_union

def non_maximum_suppression(
        bboxes,
        prob_threshold,
        iou_threshhold
):
    # prediction: [[cls, prob, cx, cy, w, h]] -> (N, 5)

    bboxes = [bbox for bbox in bboxes if bbox[1] > prob_threshold]
    # remove all bbox < probability threshold
    bboxes = sorted(bboxes, key=lambda bbox:bbox[1], reverse=True)
    # sort descending order

    result = []
    while bboxes:
        # select largest probaility bbox
        target_bbox = bboxes.pop(0) 

        bboxes = [
            bbox for bbox in bboxes
            if (bbox[0] != target_bbox[0]) or # ignore another class
                (intersection_over_union( torch.tensor(target_bbox[2:]), torch.tensor(bbox[2:])) < iou_threshhold) # ignore bbox that is lower that iou_threshold
        ]
        result.append(target_bbox)
    
    return result


if __name__ == "__main__":
    test_predictions = torch.tensor([
        [0, .6, 0, 0, 100, 100],
        [0, .7, 0, 0, 200, 100],
    ])

    print(non_maximum_suppression(test_predictions, .5, .5))