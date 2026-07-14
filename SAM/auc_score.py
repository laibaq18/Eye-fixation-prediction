import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import EyeFixationDataset

from FCN_with_Resnet50.model import EyeFixationFCN
from DeepGaze2.deepgaze_model import DeepGaze2
from SAM.model import EyeFixationSAMResNet

def auc_judd(saliency_map, fixation_map, jitter=True):
    """
    AUC-Judd implementation based on the reference provided
    https://github.com/MemoonaTahira/Visual-Saliency-Metrics-for-Evaluating-Deep-Learning-Model-performance/blob/main/_visual_attention_metrics.py

    Nonzero ground-truth pixels are treated as fixations.
    """

    saliency_map = np.asarray(
        saliency_map,
        dtype=np.float64,
    ).squeeze()

    fixation_map = np.asarray(
        fixation_map,
    ).squeeze()

    if saliency_map.shape != fixation_map.shape:
        raise ValueError(
            f"Shape mismatch: prediction={saliency_map.shape}, "
            f"fixation={fixation_map.shape}"
        )

    # No fixation pixels.
    if not fixation_map.any():
        return float("nan")

    # Add tiny random values to break ties.
    if jitter:
        saliency_map = (
            saliency_map
            + np.random.random(saliency_map.shape) / 10**7
        )

    # Normalize prediction to [0, 1].
    saliency_min = saliency_map.min()
    saliency_max = saliency_map.max()

    if saliency_max == saliency_min:
        return 0.5

    saliency_map = (
        saliency_map - saliency_min
    ) / (
        saliency_max - saliency_min
    )

    # Flatten prediction and fixation maps.
    S = saliency_map.flatten()
    F = fixation_map.flatten()

    # Prediction values at nonzero fixation locations.
    Sth = S[F > 0]

    number_of_fixations = len(Sth)
    number_of_pixels = len(S)

    if number_of_fixations == 0:
        return float("nan")

    if number_of_fixations == number_of_pixels:
        return float("nan")

    # Use saliency values at fixation locations as thresholds.
    thresholds = sorted(
        Sth,
        reverse=True,
    )

    true_positive_rates = np.zeros(
        number_of_fixations + 2
    )

    false_positive_rates = np.zeros(
        number_of_fixations + 2
    )

    true_positive_rates[0] = 0.0
    true_positive_rates[-1] = 1.0

    false_positive_rates[0] = 0.0
    false_positive_rates[-1] = 1.0

    for i, threshold in enumerate(thresholds):
        above_threshold = (
            S >= threshold
        ).sum()

        true_positive_rates[i + 1] = (
            float(i + 1)
            / number_of_fixations
        )

        # Same formula as the reference implementation.
        false_positive_rates[i + 1] = (
            float(above_threshold - i)
            / (
                number_of_pixels
                - number_of_fixations
            )
        )

    return float(
        np.trapz(
            true_positive_rates,
            x=false_positive_rates,
        )
    )


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
    parser.add_argument("--model_name", type=str, help="Model can be either FCN-resnet50 or deepgaze2 or SAM")


    args = parser.parse_args()

    data_root = Path(args.data_root)
    device = get_device()

    checkpoint_dir = (Path(f"checkpoints_{args.model_name}") / "best.pt")
    batch_size = 8
    model_name = args.model_name

    val_dataset = EyeFixationDataset(
        data_root=data_root,
        image_list_file="val_images.txt",
        fixation_list_file="val_fixations.txt",
        image_size=(224, 224),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    if model_name == "FCN-resnet50":
        model = EyeFixationFCN(
        center_bias_path=data_root / "center_bias_density.npy",
        freeze_backbone=True,
        image_size=(224, 224),
        ).to(device)

    elif model_name == "deepgaze2":
        model = DeepGaze2(
        center_bias_path=data_root / "center_bias_density.npy",
        freeze_backbone=True,
        feature_indices=(28, 29, 31, 32, 35),
        use_smoothing=True,
        ).to(device)

    elif model_name == "SAM":
        model = EyeFixationSAMResNet()

    checkpoint = torch.load(checkpoint_dir, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scores = []

    for batch_index, batch in enumerate(val_loader):
        images = batch["image"].to(device)
        fixations = batch["fixation"].to(device)

        logits = model(images)

        for image_index in range(logits.shape[0]):
            prediction_map = (
                logits[image_index, 0]
                .cpu()
                .numpy()
            )

            fixation_map = (
                fixations[image_index, 0]
                .cpu()
                .numpy()
            )

            score = auc_judd(prediction_map, fixation_map,jitter=True)

            if np.isfinite(score):
                scores.append(score)

    scores = np.asarray(scores)

    print(f"Number of valid images: {len(scores)}")
    print(f"Mean AUC-Judd: {scores.mean():.6f}")
    print(f"Median AUC-Judd: {np.median(scores):.6f}")
    print(f"Std AUC-Judd: {scores.std(ddof=1):.6f}")


if __name__ == "__main__":
    main()
