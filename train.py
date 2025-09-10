"""
train.py
--------
Main entry point for fine-tuning BLIP on OCT datasets.
"""

import os
from collections import Counter
from datasets import load_from_disk, load_dataset

from model import ImageCaptioningModel


def main():
    # Validation dataset
    val_dataset = load_dataset("QIAIUNCC/OCT-Text-Dataset", split="test")

    # Initialize model
    image_captioning_model = ImageCaptioningModel()

    # Path to subsets
    save_base_path = "/home/tania/BLIP/balanced_training_subsets"
    training_sizes = [100, 248, 500, 748, 1000, 2000, 5000,
                      10000, 15000, 20000, 30000, 39000]

    for size in training_sizes:
        subset_path = os.path.join(save_base_path, str(size))
        if os.path.exists(subset_path):
            train_subset = load_from_disk(subset_path)
            print(f"\nTraining with {len(train_subset)} samples")
            labels = [item["label"] for item in train_subset]
            print(f"Label distribution for {size} samples:", Counter(labels))

            save_dir = f"run2-base-encoders-trained-{size}"
            image_captioning_model.train(train_subset, val_dataset, save_dir)
        else:
            print(f"Subset for size {size} not found at {subset_path}")


if __name__ == "__main__":
    main()
