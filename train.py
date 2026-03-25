import ssl
import os
import shutil
import torch
torch.autograd.set_detect_anomaly(True)
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context

import torch
from torch.utils.data import DataLoader, random_split
from models.Lcsegnet import LCSegNet
from dataset import SegmentationDataset
from utils.Losses import HybridLoss
from utils.metrics import compute_all_metrics
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.GaussNoise(p=0.3),
    A.Normalize(),            # optional: standardizes input
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(),
    ToTensorV2()
])

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- Model --------
    model = LCSegNet().to(device)

    # -------- Dataset --------
    # dataset = SegmentationDataset("data/train/images", "data/train/masks")

    # # 80-10-10 split
    # train_size = int(0.8 * len(dataset))
    # val_size = int(0.1 * len(dataset))
    # test_size = len(dataset) - train_size - val_size

    train_dataset = SegmentationDataset("data/train/images", "data/train/masks", transform=train_transform)
    val_dataset   = SegmentationDataset("data/val/images", "data/val/masks", transform=val_transform)
    

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)    
    val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)
    

    # train_dataset, val_dataset, _ = random_split(dataset, [train_size, val_size, test_size])

    # train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # -------- Loss & Optimizer --------
    criterion = HybridLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # -------- Folders --------
    if os.path.exists("results"):
        shutil.rmtree("results")
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # -------- Training config -------- 
    NUM_EPOCHS = 50
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):

        # ================= TRAIN =================
        model.train()
        train_loss = 0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            # ---- Forward ----
            pred_mask, pred_boundary = model(imgs)

            # ---- Loss ----
            loss, logs = criterion(pred_mask, masks, pred_boundary)

            # ---- Backprop ----
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ================= VALIDATION =================
        model.eval()
        val_loss = 0

# 🔥 add metric accumulators
        total_metrics = {
            "accuracy": 0,
            "dice": 0,
            "iou": 0,
            "precision": 0,
            "recall": 0,
        }

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)

                pred_mask = model(imgs)

                # ---- save predicted masks ----
    #             for i in range(pred_mask.shape[0]):
    # # Define save path first
    #                 save_path = os.path.join("results", f"epoch{epoch}_val_sample{i}.png")
    
    # # Convert logits to [0,1]
    #                 mask_img = torch.sigmoid(pred_mask[i])   # [1,H,W], float
    #                 mask_img = (mask_img > 0.5).float()      # Binarize mask

    # # Option 1: Save using PIL (scaled to 0-255)
    #                 from PIL import Image
    #                 mask_img_pil = (mask_img * 255).byte()   # scale to 0-255 and convert to byte
    #                 Image.fromarray(mask_img_pil.cpu().squeeze(0).numpy()).save(save_path)

    # Option 2: Save using torchvision (expects float 0-1)
    # mask_img = mask_img.cpu().unsqueeze(0)  # [1,H,W]
    # torchvision.utils.save_image(mask_img, save_path)
            #     for i in range(pred_mask.shape[0]):
            # # Convert logits to [0,1]
            #         mask_img = torch.sigmoid(pred_mask[i])  # [1,H,W]
            #         mask_img = (mask_img > 0.5).float()
            #         mask_img = mask_img * 255
            #         mask_img = mask_img.byte()
            #         from PIL import Image
            #         Image.fromarray(mask_img.cpu().squeeze(0).numpy()).save(save_path)
            #         mask_img = mask_img.cpu().squeeze(0)    # remove channel dim

            #         save_path = os.path.join("results", f"epoch{epoch}_val_sample{i}.png")
            #         torchvision.utils.save_image(mask_img.unsqueeze(0), save_path)

        # ---- loss ----
                loss, logs = criterion(pred_mask, masks)
                val_loss += loss.item()

        # ---- metrics ----
                metrics = compute_all_metrics(pred_mask, masks)

                for key in total_metrics:
                    total_metrics[key] += metrics[key]

# average loss
        if len(val_loader) >0:

            val_loss /= len(val_loader)
        else:
            val_loss=0

# 🔥 average metrics
        for key in total_metrics:
            if len(val_loader) >0:
                total_metrics[key] /= len(val_loader)
        
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k,v in total_metrics.items()])

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | | {metrics_str}")

        # ================= SAVE BEST MODEL =================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print("✅ Saved Best Model")



             # Clear previous saved results to save space
            if os.path.exists("results"):
                shutil.rmtree("results")
            os.makedirs("results", exist_ok=True)

            # Save validation masks for this best epoch
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    pred_mask = model(imgs)
                    for i in range(pred_mask.shape[0]):
                        save_path = os.path.join("results", f"best_val_sample{i}.png")
                        mask_img = torch.sigmoid(pred_mask[i])
                        mask_img = (mask_img > 0.5).float()
                        from PIL import Image
                        mask_img_pil = (mask_img * 255).byte()
                        Image.fromarray(mask_img_pil.cpu().squeeze(0).numpy()).save(save_path)

            patience_counter = 0
        else:
            patience_counter += 1

        # ================= EARLY STOPPING =================
        if patience_counter >= patience:
            print("⛔ Early stopping triggered")
            break

    print("🎉 Training Complete")


if __name__ == "__main__":
    train()