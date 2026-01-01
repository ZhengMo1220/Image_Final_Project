import cv2
import numpy as np
import os
from main import predict_mask_mn, dice_coef

BASE = os.path.join(os.path.dirname(__file__), "MRIsample")
FILES = sorted(
    [f for f in os.listdir(os.path.join(BASE, "MN")) if f.endswith(".jpg")],
    key=lambda x: int(os.path.splitext(x)[0]),
)


def load_data():
    data = []
    for fname in FILES:
        t1 = cv2.imread(os.path.join(BASE, "T1", fname), cv2.IMREAD_GRAYSCALE)
        t2 = cv2.imread(os.path.join(BASE, "T2", fname), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(os.path.join(BASE, "MN", fname), cv2.IMREAD_GRAYSCALE)
        gt = (gt > 0).astype(np.uint8)
        data.append((fname, t1, t2, gt))
    return data


def evaluate():
    rows = []
    for fname, t1, t2, gt in load_data():
        pred = predict_mask_mn(t1, t2)
        score = dice_coef(gt, pred)
        rows.append((fname, score, gt.sum(), pred.sum()))
    return rows


if __name__ == "__main__":
    rows = evaluate()
    mean_dice = np.mean([r[1] for r in rows])
    print(f"MN Dice mean: {mean_dice:.3f}")
    for fname, score, gt_area, pred_area in rows:
        print(f"{fname}: dice={score:.3f}, gt_area={gt_area}, pred_area={pred_area}")
