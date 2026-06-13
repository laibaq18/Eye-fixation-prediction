import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EyeFixationDataset
from model import EyeFixationFCN


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        images = batch["image"].to(device)
        fixations = batch["fixation"].to(device)

        preds = model(images)

        loss = F.binary_cross_entropy_with_logits(preds, fixations)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Validation"):
        images = batch["image"].to(device)
        fixations = batch["fixation"].to(device)

        preds = model(images)
        loss = F.binary_cross_entropy_with_logits(preds, fixations)

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    train_dataset = EyeFixationDataset(
        data_root=data_root,
        image_list_file="train_images.txt",
        fixation_list_file="train_fixations.txt",
        image_size=(224, 224),
    )

    val_dataset = EyeFixationDataset(
        data_root=data_root,
        image_list_file="val_images.txt",
        fixation_list_file="val_fixations.txt",
        image_size=(224, 224),
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model = EyeFixationFCN(
        center_bias_path=data_root / "center_bias_density.npy",
        freeze_backbone=args.freeze_backbone,
        image_size=(224, 224),
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    # checking everything before running epochs:
    sample = next(iter(train_loader))
    images = sample["image"].to(device)
    fixations = sample["fixation"].to(device)

    with torch.no_grad():
        preds = model(images)

    print("Image shape:", images.shape)
    print("Fixation shape:", fixations.shape)
    print("Prediction shape:", preds.shape)
    print("Fixation min/max:", fixations.min().item(), fixations.max().item())
    print("Prediction logits min/max:", preds.min().item(), preds.max().item())

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        print(f"Train loss: {train_loss:.6f}")
        print(f"Val loss:   {val_loss:.6f}")

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        torch.save(checkpoint, save_dir / "last.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, save_dir / "best.pt")
            print("Saved new best model.")


if __name__ == "__main__":
    main()
