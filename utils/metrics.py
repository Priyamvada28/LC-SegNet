import torch


def get_stats(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    TP = (preds * targets).sum()
    FP = (preds * (1 - targets)).sum()
    FN = ((1 - preds) * targets).sum()
    TN = ((1 - preds) * (1 - targets)).sum()

    return TP, FP, FN, TN


def precision(preds, targets):
    TP, FP, _, _ = get_stats(preds, targets)
    return TP / (TP + FP + 1e-6)


def recall(preds, targets):
    TP, _, FN, _ = get_stats(preds, targets)
    return TP / (TP + FN + 1e-6)


def accuracy(preds, targets):
    TP, FP, FN, TN = get_stats(preds, targets)
    return (TP + TN) / (TP + TN + FP + FN + 1e-6)


def dice_score(preds, targets):
    TP, FP, FN, _ = get_stats(preds, targets)
    return (2 * TP) / (2 * TP + FP + FN + 1e-6)


def iou_score(preds, targets):
    TP, FP, FN, _ = get_stats(preds, targets)
    return TP / (TP + FP + FN + 1e-6)


def compute_all_metrics(preds, targets):
    return {
        "accuracy": accuracy(preds, targets).item(),
        "dice": dice_score(preds, targets).item(),
        "iou": iou_score(preds, targets).item(),
        "precision": precision(preds, targets).item(),
        "recall": recall(preds, targets).item(),
    }