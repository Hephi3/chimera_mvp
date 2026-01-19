import matplotlib.pyplot as plt
import os

path = "/local/scratch/phempel/chimera/debugging_augmented9"

images = [f.split("_final.png")[0] for f in os.listdir(path) if f.endswith("_final.png")]

# print first 4 images in both "orig" and "cor"
n = 4
fig, axes = plt.subplots(2, n, figsize=(2*n, 4))
for i in range(n):
    img_name = images[i+0]
    img_orig = plt.imread(os.path.join(path, f"{img_name}_orig.png"))
    # img_aug = plt.imread(os.path.join(path, f"{img_name}_aug.png"))
    img_cor = plt.imread(os.path.join(path, f"{img_name}_cor.png"))
    
    axes[0, i].imshow(img_orig)
    axes[0, i].set_title(f"Original")
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])
    # axes[0, i].axis('off')
    # axes[0, i].b
    
    # axes[1, i].imshow(img_aug)
    # axes[1, i].set_title(f"Augmented: {img_name}")
    # axes[1, i].axis('off')
    
    axes[1, i].imshow(img_cor)
    axes[1, i].set_title(f"Augmented")
    axes[1, i].set_xticks([])
    axes[1, i].set_yticks([])
    # axes[1, i].axis('off')
plt.tight_layout()
plt.savefig("brightness_augmentation_debug.png", dpi=600)