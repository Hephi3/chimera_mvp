import matplotlib.pyplot as plt
import os
import random
from PIL import Image

# Folder containing images
image_folder = "/local/scratch/phempel/chimera/debugging_augmented4"

# Valid image extensions
VALID_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')

# Collect files
files = [f for f in os.listdir(image_folder) if f.lower().endswith(VALID_EXTS)]

# Build mapping from stem -> {'cor': filename, 'final': filename}
pairs_map = {}
for f in files:
    name, ext = os.path.splitext(f)
    if name.endswith('_cor'):
        stem = name[:-4]
        pairs_map.setdefault(stem, {})['cor'] = f
    elif name.endswith('_final'):
        stem = name[:-6]
        pairs_map.setdefault(stem, {})['final'] = f

# Keep only stems that have both cor and final
pairs = []
for stem, d in pairs_map.items():
    if 'cor' in d and 'final' in d:
        pairs.append((d['cor'], d['final'], stem))

if not pairs:
    print(f"No pairs found in {image_folder}. Looking for filenames that end with '_cor' and '_final' (any image extension).")
    raise SystemExit(1)

# Choose up to 8 random pairs (4x4 grid -> 16 images -> 8 pairs)
num_pairs = min(8, len(pairs))
selected_pairs = random.sample(pairs, num_pairs)

# Flatten selected pairs into image list (cor then final for each pair)
images_to_show = []  # list of (filename, title)
for cor, final, stem in selected_pairs:
    images_to_show.append((cor, f"{stem}_cor"))
    images_to_show.append((final, f"{stem}_final"))

# Create a 4x4 grid
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
axes = axes.flatten()

for idx, (fname, title) in enumerate(images_to_show):
    ax = axes[idx]
    img_path = os.path.join(image_folder, fname)
    try:
        img = Image.open(img_path).convert('RGB')
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(title, fontsize=8)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error\n{e}", ha='center', va='center')
        ax.axis('off')

# Hide any remaining empty subplots
for idx in range(len(images_to_show), 16):
    axes[idx].axis('off')

plt.tight_layout()
plt.show()
