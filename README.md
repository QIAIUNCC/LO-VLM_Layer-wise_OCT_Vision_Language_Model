# LO-VLM

# OCT-BLIP Captioning

This repository contains code to fine-tune and evaluate the LO-VLM model for **retinal OCT (Optical Coherence Tomography) image captioning**.  

It accompanies the paper:  

> **Compact Vision–Language Models Enable Efficient and Interpretable Automated OCT Analysis Through Layer-Specific Multimodal Learning**  
> *bioRxiv preprint, August 2025*  
> [Read the paper](https://www.biorxiv.org/content/10.1101/2025.08.07.669187v1.full.pdf)

The goal is to translate the anatomical signatures of retinal disease from OCT B-scans into **structured, clinically meaningful text captions** (diagnosis + retinal layer description).


---

## 📂 Repository Structure

```
oct-blip-captioning/
│── train.py               # Entry point for training (CLI with hyperparams)
│── inference.py           # Script for running inference on test data
│── dataset.py             # Custom PyTorch dataset wrapper (for training)
│── model.py               # BLIP wrapper (training & validation loops)
│── utils.py               # Helper functions (e.g., count params)
│── requirements.txt       # Dependencies
│── README.md              # Documentation (this file)
│── checkpoints/           # (Generated) model checkpoints
│── logs/                  # (Generated) TensorBoard logs
```

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/oct-blip-captioning.git
cd oct-blip-captioning
pip install -r requirements.txt
```

Dependencies include:
- `torch`
- `transformers`
- `datasets`
- `tqdm`
- `tensorboard`
- `Pillow`

---

## 📊 Data Format

This code expects the dataset in **Hugging Face `datasets` format**, either loaded from the Hugging Face Hub or saved locally with `datasets.save_to_disk()`.

Each entry should contain:
- `img_path` → path to the OCT image (e.g., `"data/sample_001.png"`)
- `label` → diagnosis label (e.g., `"AMD"`, `"DME"`)
- `desc` → retinal layer description (e.g., `"Disruption of IS/OS junction with subretinal fluid"`)

**Example dataset entry:**
```json
{
  "img_path": "data/images/patient123_slice45.png",
  "label": "AMD",
  "desc": "Disruption in outer retinal layers with pigment epithelial detachment"
}
```

Validation data in the paper comes from:  
[`QIAIUNCC/OCT-Text-Dataset`](https://huggingface.co/datasets/QIAIUNCC/OCT-Text-Dataset)  

---

## 🚀 Training

Training fine-tunes BLIP on paired image–text data.  

### Example command:

```bash
python train.py   --train_base_path /path/to/balanced_training_subsets   --training_sizes 500 2000 5000   --epochs 30   --batch_size 16   --val_batch_size 4   --val_dataset QIAIUNCC/OCT-Text-Dataset   --val_split test   --device cuda
```

### Arguments

| Argument           | Description |
|--------------------|-------------|
| `--train_base_path` | Base directory containing pre-saved training subsets (`100/`, `500/`, etc.) |
| `--training_sizes` | Subset sizes to train on (e.g., `500 2000 5000`) |
| `--epochs`         | Number of training epochs |
| `--batch_size`     | Training batch size |
| `--val_batch_size` | Validation batch size |
| `--val_dataset`    | Validation dataset (HF format) |
| `--val_split`      | Split used for validation (default: `test`) |
| `--model_name`     | Hugging Face model name (default: BLIP base) |
| `--device`         | Compute device (`cuda` or `cpu`) |

### Outputs during training
- **Logs** → TensorBoard logs in `logs/`  
  ```bash
  tensorboard --logdir run-base-500/logs
  ```
- **Checkpoints** → Saved in `checkpoints/model_step_<N>.pth`
- **Validation results** → JSON file with generated vs. expected captions:
  ```json
  [
    {
      "image_label": "AMD",
      "generated_text": "OCT shows subretinal fluid with PED",
      "Expected_Text": "Disruption in outer retinal layers with pigment epithelial detachment"
    }
  ]
  ```

---

## 🔍 Inference

Once a model is trained, run **inference** to generate captions on test data.

### Example command:

```bash
python inference.py   --dataset_name QIAIUNCC/OCT-Text-Dataset   --dataset_split test   --model_name Salesforce/blip-image-captioning-base   --checkpoint_path checkpoints/model_step_5000.pth   --output_json results_test.json   --output_txt results_test.txt   --device cuda
```

### Outputs
1. **JSON file** (default: `inference_results.json`):
   ```json
   {
     "image_path": "data/images/patient123_slice45.png",
     "image_label": "AMD",
     "generated_text": "OCT shows retinal layer disruption and fluid pockets",
     "Expected_Text": "Disruption in outer retinal layers with pigment epithelial detachment"
   }
   ```
2. **Plain text file** (default: `inference_results.txt`) for quick viewing.

---

## 🧠 How It Works

1. **Dataset Wrapping**  
   `ImageCaptioningDataset` constructs paired image–text inputs, using prompts of the form:  
   ```
   Diagnosed disease: <label>,
   Layer Information: <desc>
   ```

2. **Model Fine-tuning**  
   - Loads **BLIP base** (`Salesforce/blip-image-captioning-base`)  
   - Optimizer: `AdamW`  
   - Scheduler: Cosine Annealing  

3. **Evaluation**  
   - Uses BLIP’s `.generate()` to produce captions.  
   - Saves expected vs. generated outputs for later analysis.  

4. **Scaling Experiments**  
   - Training is repeated across increasing subset sizes (`100` → `39000`) to study data scaling effects.  

---

## 📑 Paper Context

This repository reproduces experiments from our bioRxiv preprint:  
**Compact Vision–Language Models Enable Efficient and Interpretable Automated OCT Analysis Through Layer-Specific Multimodal Learning**  
([PDF link](https://www.biorxiv.org/content/10.1101/2025.08.07.669187v1.full.pdf))

Key findings:
- A compact BLIP model can generate **accurate clinical narratives** from OCT B-scans.  
- Scaling experiments show performance improvements with more data.  
- Captions explicitly describe **disease diagnosis** and **layer-specific structural features**, enabling **interpretability**.  

---

## 📝 Citation

If you use this code, please cite:

```
@article{gholami2025lovlm,
  title   = {Compact Vision--Language Models Enable Efficient and Interpretable Automated OCT Analysis Through Layer-Specific Multimodal Learning},
  author  = {Gholami, Sina and Alam, Minhaj and Lim, Jennifer and et al.},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.08.07.669187}
}
```

---

## 🙏 Acknowledgements

- BLIP: Junnan Li et al. (2022) — [Bootstrapping Language–Image Pretraining](https://arxiv.org/abs/2201.12086)  
- Hugging Face Datasets/Transformers libraries  
- UNC Charlotte QIAI Lab for dataset curation  
