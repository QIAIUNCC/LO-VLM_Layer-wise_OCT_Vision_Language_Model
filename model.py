"""
model.py
--------
BLIP model wrapper for fine-tuning on OCT image captioning.
"""

import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, BlipForConditionalGeneration

from dataset import ImageCaptioningDataset
from utils import count_trainable_params


class ImageCaptioningModel:
    """
    Wrapper for BLIP training and evaluation.
    """

    def __init__(self, model_name="Salesforce/blip-image-captioning-base", device="cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)

        # Print number of trainable params
        print(f"Number of trainable parameters: {count_trainable_params(self.model)}")

    # --------------------------------------------------------
    def train(self, train_dataset, val_dataset, save_dir,
              num_epochs=50, batch_size=32, validation_steps=20):
        """
        Training loop for BLIP model.
        """
        # DataLoaders
        train_dataloader = DataLoader(
            ImageCaptioningDataset(train_dataset, self.processor),
            shuffle=True, batch_size=batch_size, pin_memory=True,
        )
        val_dataloader = DataLoader(
            ImageCaptioningDataset(val_dataset, self.processor),
            batch_size=4, pin_memory=True,
        )

        # Optimizer + Scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )

        # Logging + checkpoint dirs
        log_dir = os.path.join(save_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

        checkpoint_dir = os.path.join(save_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        step_counter = 0

        # -------------------------------
        # Epoch loop
        # -------------------------------
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc="Training")

            for _, batch in enumerate(progress_bar):
                input_ids = batch.pop("input_ids").to(self.device)
                pixel_values = batch.pop("pixel_values").to(self.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids,
                                     pixel_values=pixel_values,
                                     labels=input_ids)
                loss = outputs.loss
                total_loss += loss.item()

                # Backprop
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                step_counter += 1

                progress_bar.set_postfix({"Loss": loss.item()})

                # Logging
                avg_train_loss = total_loss / len(train_dataloader)
                writer.add_scalar("Loss/Train", avg_train_loss, step_counter)
                writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], step_counter)

                # Scheduler step
                scheduler.step()

            # Validation at epoch end
            self.evaluate(val_dataloader, step_counter, writer, checkpoint_dir)

        writer.close()

    # --------------------------------------------------------
    def evaluate(self, val_dataloader, step_counter, writer, checkpoint_dir):
        """
        Validation loop + checkpoint saving.
        """
        self.model.eval()
        val_loss = 0
        val_results = []

        total_val_samples = len(val_dataloader.dataset)

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                image_label = batch.pop("image_label")
                original_text = batch.pop("original_text")
                pixel_values = batch["pixel_values"].to(self.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids,
                                     pixel_values=pixel_values,
                                     labels=input_ids)
                val_loss += outputs.loss.item()

                # Generate captions
                generated_ids = self.model.generate(pixel_values=pixel_values, max_length=256)
                generated_text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

                for i in range(len(batch["input_ids"])):
                    val_results.append({
                        "image_label": image_label[i],
                        "generated_text": generated_text[i],
                        "Expected_Text": original_text[i],
                    })

        avg_val_loss = val_loss / total_val_samples
        writer.add_scalar("Loss/Val", avg_val_loss, step_counter)

        # Save results
        results_path = os.path.join(checkpoint_dir, f"val_results_step_{step_counter}.json")
        with open(results_path, "w") as f:
            json.dump(val_results, f, indent=4)

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"model_step_{step_counter}.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
