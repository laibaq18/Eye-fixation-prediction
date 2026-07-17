import os
import re
import matplotlib.pyplot as plt

# =====================================================
# Configuration
# =====================================================

LOG_DIR = "logs"
OUTPUT_DIR = "plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = {
    "SAM": "sam_AUC.txt",
    "DeepGaze II": "deep_gaze_AUC.txt"
}

# =====================================================
# Regex (supports many formats)
# =====================================================

patterns = [
    r"Mean\s*AUC(?:-Judd)?\s*[:=]\s*([0-9]*\.?[0-9]+)",
    r"Average\s*AUC(?:-Judd)?\s*[:=]\s*([0-9]*\.?[0-9]+)",
    r"Mean.*?AUC.*?([0-9]*\.?[0-9]+)"
]

scores = {}

# =====================================================
# Read Files
# =====================================================

for model, filename in FILES.items():

    filepath = os.path.join(LOG_DIR, filename)

    print(f"\nReading: {filepath}")

    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        continue

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    auc = None

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            auc = float(match.group(1))
            break

    if auc is None:
        print(f"❌ Could not find Mean AUC in {filename}")
        print("\nFirst 500 characters:\n")
        print(text[:500])
        continue

    scores[model] = auc
    print(f"✓ {model}: {auc:.4f}")

# =====================================================
# Plot
# =====================================================

if len(scores) == 0:
    print("\nNo AUC values were found.")
    exit()

plt.figure(figsize=(6,5))

bars = plt.bar(scores.keys(), scores.values())

for bar, value in zip(bars, scores.values()):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        value + 0.002,
        f"{value:.3f}",
        ha="center",
        fontsize=11
    )

plt.ylabel("Mean AUC-Judd")
plt.title("Model Performance Comparison")

ymin = max(0, min(scores.values()) - 0.05)
ymax = min(1.0, max(scores.values()) + 0.05)

plt.ylim(ymin, ymax)

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

png_path = os.path.join(OUTPUT_DIR, "auc_comparison.png")

plt.savefig(png_path, dpi=300)

plt.show()

print("\nSaved:")
print(" ", png_path)