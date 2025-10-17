## Paper Context

This repository reproduces experiments from  
[Paper link](https://www.biorxiv.org/content/10.1101/2025.08.07.669187v1.full.pdf)
 
# OCT-BLIP Captioning

This repository contains code to fine-tune and evaluate the LO-VLM model for **retinal OCT (Optical Coherence Tomography) image captioning**.  

It accompanies the paper:  

> **Compact Vision–Language Models Enable Efficient and Interpretable Automated OCT Analysis Through Layer-Specific Multimodal Learning**  
> *bioRxiv preprint, August 2025*  
> [Read the paper](https://www.biorxiv.org/content/10.1101/2025.08.07.669187v1.full.pdf)

The goal is to translate the anatomical signatures of retinal disease from OCT B-scans into **structured, clinically meaningful text captions** (diagnosis + retinal layer description).

---

## 📊 Input Format

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

---
# 🔗 Hugging Face Resources

All datasets, trained models, and results used in this project are hosted on [Hugging Face](https://huggingface.co/QIAIUNCC).

---

## 📊 Dataset

**[QIAIUNCC/OCT-summary-Dataset](https://huggingface.co/datasets/QIAIUNCC/OCT-summary-Dataset)**  
Contains OCT images paired with structured textual descriptions.

```python
from datasets import load_dataset
dataset = load_dataset("QIAIUNCC/OCT-summary-Dataset", split="train")
print(dataset[0])
```

---

## 🤖 Models

**[QIAIUNCC/LO-VLM](https://huggingface.co/QIAIUNCC/LO-VLM)**  
→ Full compact LO-VLM.

**[QIAIUNCC/LO-VLM-trained-encoders](https://huggingface.co/QIAIUNCC/LO-VLM-trained-encoders)**  
→ LO-VLM model with trained encoders.

```python
from transformers import AutoProcessor, BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("QIAIUNCC/LO-VLM")
model = BlipForConditionalGeneration.from_pretrained("QIAIUNCC/LO-VLM").to("cuda")
```

---

## 📑 Results

**[QIAIUNCC/LO-VLM-and-EYELlama-Results](https://huggingface.co/QIAIUNCC/LO-VLM-and-EYELlama-Results)**  
→ Contains LO-VLM generated captions and EYE-Llama's enhanced interpretations. 

---

## 🚀 Training

### Training command:

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
 **JSON file** (default: `inference_results.json`):
   ```json
   {
     "image_path": "data/images/patient123_slice45.png",
     "image_label": "AMD",
     "generated_text": "OCT shows retinal layer disruption and fluid pockets",
     "Expected_Text": "Disruption in outer retinal layers with pigment epithelial detachment"
   }
   ```
---

## Citation

If you use this code, please cite:

```
@article{gholami2025lovlm,
  title   = {Compact Vision--Language Models Enable Efficient and Interpretable Automated OCT Analysis Through Layer-Specific Multimodal Learning},
  author = {Tania Haghighi and Sina Gholami and Jared Todd Sokol and Aayush Biswas and Jennifer I. Lim and Theodore Leng and Atalie C. Thompson and Hamed Tabkhi and Minhaj Nur Alam},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.08.07.669187}
}
```

