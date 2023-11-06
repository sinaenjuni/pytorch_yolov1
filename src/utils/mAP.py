import torch
from collections import Counter
from utils import intersection_over_union

def mean_average_percision(
        pred_bboxes,
        target_bboxes,
        iou_threshold,
        num_classes
):
    # pred_bboxes = [[img_idx, cls, prob_score, cx, cy, w, h],
    #                 ...]

    
    epsilon=1e-6
    average_precisions = []
    predictions = []
    targets = []

    for cls in range(num_classes):
        # 동일한 class에 대한 예측 박스와 정답 박스를 저장
        for pred_bbox in pred_bboxes:
            if pred_bbox[1] == cls:
                predictions.append(pred_bbox)
        for target_bbox in target_bboxes:
            if target_bbox[1] == cls:
                targets.append(target_bbox)

        # img0 has 3 bboxes
        # img1 has 5 bboxes
        # num_bbox_per_cls = {0:3, 1:5} 클래스 별 파일의 개수
        num_bbox_per_cls = Counter([target[0] for target in targets]) 

        for k, v in num_bbox_per_cls.items():
            num_bbox_per_cls[k] = torch.zeros(v)
        # num_bbox_per_cls = {0:torch.tensor([0,0,0]), 1:torch.tensor([0,0,0,0,0])}
        
        predictions.sort(key=lambda x: x[2], reverse=True)
        # probability로 정렬
        TP = torch.zeros((len(predictions)))
        FP = torch.zeros((len(predictions)))
        num_targets = len(targets)

        for prediction_idx, prediction in enumerate(predictions):
            target_imgs = [
                target_bbox for target_bbox in targets
                if (target_bbox[0] == prediction[0])
            ]

            num_targets = len(target_imgs)
            best_iou = 0

            for target_idx, target_img in enumerate(target_imgs):
                iou = intersection_over_union(
                    torch.tensor(prediction[3:]),
                    torch.tensor(target_img[3:])
                )

                if(iou > best_iou):
                    best_iou = iou
                    best_target_idx = target_idx
        
            if (best_iou > iou_threshold): # iou_threshold를 기준으로 TP, FP 선정
                if num_bbox_per_cls[prediction[0]][best_target_idx] == 0:
                    TP[prediction_idx] = 1
                    num_bbox_per_cls[prediction[0]][best_target_idx] = 1
                else:
                    # 이미 처리된 bbox의 경우 FP
                    FP[prediction_idx] = 1 
            else:
                # iou_threhold를 넘지 못하는 경우
                FP[prediction_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (num_targets + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        # recall에 따른 precision 그래프를 그릴 때, 0과 1에서 부터 시작하기 위함
        precisions = torch.cat(torch.tensor([1]), precisions)
        recalls = torch.cat(torch.tensor([0]), recalls)
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)










