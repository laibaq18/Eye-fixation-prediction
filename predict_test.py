import argparse
from pathlib import Path

import imageio.v2 as imageio
import torch
from torch.utils.data import DataLoader

from dataset import EyeFixationDataset
from model import EyeFixationFCN


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="test_predictions")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    test_dataset = EyeFixationDataset(
        data_root=data_root,
        image_list_file="test_images.txt",
        fixation_list_file=None,
        image_size=(224, 224),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = EyeFixationFCN(
        center_bias_path=data_root / "center_bias_density.npy",
        freeze_backbone=True,
        image_size=(224, 224),
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    for batch in test_loader:
        images = batch["image"].to(device)
        names = batch["name"]

        logits = model(images)

        # Convert logits to probabilities in [0,1]
        probs = torch.sigmoid(logits)

        # Shape: [1, 1, H, W] -> [H, W]
        pred = probs.squeeze(0).squeeze(0)

        # Convert to uint8 grayscale image
        pred_uint8 = (pred.clamp(0, 1) * 255).to(torch.uint8)
        pred_np = pred_uint8.cpu().numpy()

        image_name = Path(names[0]).stem
        out_path = save_dir / f"{image_name}.png"

        imageio.imwrite(out_path, pred_np)

    print(f"Saved predictions to: {save_dir}")


if __name__ == "__main__":
    main()
