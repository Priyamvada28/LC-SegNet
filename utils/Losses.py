import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


# -------------------- Dice Loss --------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)

        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


# -------------------- Boundary Loss --------------------
class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def get_boundary(self, mask):
        boundaries = []

        for m in mask:
            m_np = m.squeeze().cpu().numpy().astype(np.uint8)

            edge = cv2.Canny(m_np * 255, 100, 200)
            edge = edge / 255.0

            boundaries.append(edge)

        boundaries = np.stack(boundaries)
        boundaries = torch.tensor(boundaries).unsqueeze(1).float()

        return boundaries.to(mask.device)

    def forward(self, pred_boundary, gt_mask):
        gt_boundary = self.get_boundary(gt_mask)
        return self.bce(pred_boundary, gt_boundary)


# -------------------- Hybrid Loss --------------------
class HybridLoss(nn.Module):
    def __init__(self, lambda_dice=1.0, lambda_boundary=1.0):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()  # Cross-Entropy (binary case)
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()

        self.lambda_dice = lambda_dice
        self.lambda_boundary = lambda_boundary

    def forward(self, pred_mask, gt_mask, pred_boundary=None):

        # ----- Cross Entropy -----
        loss_ce = self.bce(pred_mask, gt_mask)

        # ----- Dice -----
        loss_dice = self.dice(pred_mask, gt_mask)

        # ----- Boundary -----
        if pred_boundary is not None:
            loss_boundary = self.boundary(pred_boundary, gt_mask)
        else:
            loss_boundary = 0.0

        # ----- Total Loss -----
        total_loss = loss_ce + self.lambda_dice * loss_dice + self.lambda_boundary * loss_boundary

        return total_loss, {
            "ce": loss_ce.item(),
            "dice": loss_dice.item(),
            "boundary": loss_boundary if isinstance(loss_boundary, float) else loss_boundary.item()
        }