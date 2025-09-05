
# 🎨 Conditional Image Generation using GaussianGAN (GauGAN)

## 📌 Project Overview
This project implements **Conditional Image Generation** using a **Gaussian-based Generative Adversarial Network (GAN)** inspired by NVIDIA’s **GauGAN**.  
The model learns to generate **photorealistic images from segmentation maps**, conditioned on semantic labels.  

## 📂 Repository Structure
   - Final_Gaugan_Training.ipynb for Main training & evaluation script
   - Datasets for Dataset info and download links
   - Output Vs Ground Truth for generated outputs, ground truth images and segmentation label maps
   - Project_Report
   - README.md for Project documentation


## 📂 Dataset
Two datasets were used for training and evaluation.  
Links are provided inside [`Dataset/README.md`](Dataset/README.md).

For PASCAL_VOC_2012 Dataset: 
- `images/` → real images  
- `segmentation_map/` → RGB visualization of segmentation  
- `segmentation_labels/` → single-channel labels (IDs ∈ `[0…NUM_CLASSES-1]`, `255` = void class)

For Buildings Facades Dataset:
- Facades dataset consists of 506 Building Facades & corresponding Segmentations with split into train and test subsets.

## ⚙️ Model Architecture
- **Encoder:** CNN encoder predicting μ, σ for Gaussian latent (KL regularization)  
- **Generator:** SPADE residual blocks conditioned on one-hot segmentation labels, latent z from a Gaussian encoder  
- **Discriminator:** Multi-scale PatchGAN (hinge loss)

**Losses used:**
- GAN Loss (hinge)  
- Feature Matching Loss  
- Perceptual Loss (VGG19)  
- KL Divergence Loss

## 🏋️ Training

### Data Pipeline
- Images, segmentation maps, and label maps are loaded from the dataset folder.
- Preprocessing:
  - Resize to **288×288**, random crop to **256×256**.
  - Normalize images to **[-1, 1]**.
  - Segmentation labels: `255` (void class) → `0` (background).
  - Labels converted to one-hot for **SPADE conditioning**.

### Model
- **Generator:** SPADE residual blocks + Gaussian encoder (μ, σ → latent z).
- **Discriminator:** Multi-scale PatchGAN with hinge loss.
- **Encoder:** CNN encoder for Gaussian latent (KL divergence regularization).

### Loss Functions
- **GAN Loss (hinge)**
- **Feature Matching Loss** (L1 across discriminator features)
- **Perceptual Loss (VGG19)**
- **KL Divergence Loss**

### Training Loop
1. Generator produces fake images from (segmentation labels + latent z).  
2. Discriminator receives real vs fake pairs.  
3. Compute generator + discriminator losses.  
4. Update parameters using Adam:  
   - Generator LR = `1e-4`  
   - Discriminator LR = `4e-4`  
5. Every few epochs:
   - Save **checkpoints** (`gaugan_epoch_xxx.weights.h5`)  
   - Display triplets → **(segmentation | real | generated)**  

---

## 📊 Evaluation

### 1. Image Export
After training:
- Load best checkpoint.
- Run inference on validation set.
- Save outputs.

### 2. Metrics

**FID (Fréchet Inception Distance)**
- Compares realism of generated vs real images i.e., Compares the difference between distribution of real and generated images.

**Semantic Consistency Metrics**
- A pretrained DeepLabV3-ResNet50 (torchvision) is used to predict masks on generated images.
- Metrics computed: Pixel Accuracy and Mean IoU (Intersection-over-Union)
- Compared against ground-truth label maps.

## 📑 Project Report
Detailed analysis in Project_Report.pdf

## 📌 Key Learnings
- How SPADE-based GANs (GauGAN) work
- Role of KL regularization in conditional GANs
- Stabilization from feature matching & perceptual loss
- Evaluation with FID + semantic metrics
 
