import os
import cv2
import torch
import numpy as np
from models.Lcsegnet import LCSegNet
from utils.metrics import compute_all_metrics
from PIL import Image

# ---------------- LOAD MODEL ----------------
def load_model(model_path, device):
    model = LCSegNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# ---------------- PREPROCESS ----------------
def preprocess_image(img_path):
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Image not found or corrupted: {img_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    image = torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0)
    return image, original_size

# ---------------- LOAD MASK ----------------
def load_mask(mask_path):
    if not os.path.exists(mask_path):
        return None  # return None if mask is missing
    mask = cv2.imread(mask_path, 0)
    if mask is None:
        return None
    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype("float32")
    mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float()
    return mask

# ---------------- POSTPROCESS ----------------
def postprocess_mask(pred_mask, original_size):
    pred_mask = torch.sigmoid(pred_mask)
    pred_mask = (pred_mask > 0.5).float()
    pred_mask = pred_mask.squeeze().cpu().numpy()
    pred_mask = cv2.resize(pred_mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    return pred_mask

# ---------------- MAIN ----------------
def predict_and_evaluate(image_dir, mask_dir, output_dir, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    os.makedirs(output_dir, exist_ok=True)

    image_list = sorted(os.listdir(image_dir))

    total_metrics = {"accuracy": 0, "dice": 0, "iou": 0, "precision": 0, "recall": 0}
    count = 0

    for img_name in image_list:
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(image_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + ".jpg"
        mask_path = os.path.join(mask_dir, mask_name)

        # ---- preprocess ----
        try:
            image, original_size = preprocess_image(img_path)
        except ValueError as e:
            print(f"⚠️ {e}, skipping...")
            continue

        image = image.to(device)
        gt_mask = load_mask(mask_path)
        if gt_mask is not None:
            gt_mask = gt_mask.to(device)

        # ---- prediction ----
        with torch.no_grad():
            pred_mask = model(image)

        # ---- metrics ----
        if gt_mask is not None:
            metrics = compute_all_metrics(pred_mask, gt_mask)
            for key in total_metrics:
                total_metrics[key] += metrics[key]
            count += 1

        # ---- save prediction ----
        pred_mask_np = postprocess_mask(pred_mask, original_size)
        save_path = os.path.join(output_dir, img_name)
        cv2.imwrite(save_path, (pred_mask_np * 255).astype(np.uint8))

        print(f"✅ Processed: {img_name}")

    if count > 0:
        for key in total_metrics:
            total_metrics[key] /= count
        print("\n🔥 FINAL TEST METRICS 🔥")
        print(f"Accuracy : {total_metrics['accuracy']:.4f}")
        print(f"Dice     : {total_metrics['dice']:.4f}")
        print(f"IoU      : {total_metrics['iou']:.4f}")
        print(f"Precision: {total_metrics['precision']:.4f}")
        print(f"Recall   : {total_metrics['recall']:.4f}")
    else:
        print("⚠️ No valid masks found. Metrics not computed.")

# ---------------- RUN ----------------
if __name__ == "__main__":
    predict_and_evaluate(
        image_dir="data/test/images",
        mask_dir="data/test/masks",
        output_dir="predictions",
        model_path="checkpoints/best_model.pth"
    )