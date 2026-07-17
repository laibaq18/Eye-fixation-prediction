import os
import re
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================

LOG_DIR = "logs"
OUTPUT_DIR = "plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILES = {
    "SAM": "sam_training_log.txt",
    "DeepGaze II": "deep_gaze_training.txt"
}

# ============================================================
# Regex patterns
# ============================================================

epoch_pattern = re.compile(r"Epoch\s+(\d+)")
train_pattern = re.compile(r"Train loss:\s*([-+]?\d*\.?\d+)")
val_pattern = re.compile(r"Val loss:\s*([-+]?\d*\.?\d+)")

# ============================================================
# Parse Log
# ============================================================

def parse_log(filepath):

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    epochs = [int(x) for x in epoch_pattern.findall(text)]
    train_loss = [float(x) for x in train_pattern.findall(text)]
    val_loss = [float(x) for x in val_pattern.findall(text)]

    n = min(len(epochs), len(train_loss), len(val_loss))

    epochs = epochs[:n]
    train_loss = train_loss[:n]
    val_loss = val_loss[:n]

    return epochs, train_loss, val_loss

# ============================================================
# Plot Function
# ============================================================

def plot_losses(model_name, epochs, train_loss, val_loss):

    plt.figure(figsize=(8,5))

    plt.plot(
        epochs,
        train_loss,
        linewidth=2,
        label="Training Loss"
    )

    plt.plot(
        epochs,
        val_loss,
        linewidth=2,
        label="Validation Loss"
    )

    best_epoch = epochs[val_loss.index(min(val_loss))]
    best_loss = min(val_loss)

    plt.scatter(
        best_epoch,
        best_loss,
        s=80,
        marker='o',
        label=f"Best Validation ({best_epoch})"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Training")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()

    png_path = os.path.join(
        OUTPUT_DIR,
        model_name.replace(" ", "_").replace("/", "_").lower()
        + "_loss_curve.png"
    )

    plt.savefig(png_path, dpi=300)

    plt.close()

# ============================================================
# Main
# ============================================================

for model, logfile in LOG_FILES.items():

    path = os.path.join(LOG_DIR, logfile)

    if not os.path.exists(path):
        print(f"Missing: {path}")
        continue

    epochs, train_loss, val_loss = parse_log(path)

    print("="*50)
    print(model)
    print("Epochs:", len(epochs))
    print("Best Validation Loss:", min(val_loss))
    print("Best Epoch:", epochs[val_loss.index(min(val_loss))])

    plot_losses(model, epochs, train_loss, val_loss)

print("\nGraphs saved inside:", OUTPUT_DIR)