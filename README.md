# MoEDiff-SR: Mixture of Experts-Guided Diffusion Model for Region-Adaptive MRI Super-Resolution

This repository implements **MoEDiff-SR**, a Mixture of Experts (MoE)-Guided diffusion framework for **region-adaptive super-resolution (SR)** of brain MRI. The model leverages anatomical specialization across multiple expert networks and a Swin Transformer-based gating mechanism conditioned on multi-scale 3T MRI inputs.

---

## 🚀 Highlights

- **Multi-Scale Patch Encoding**: Extracts global and local features from 3T MRI slices.
- **Swin Transformer Gating**: Token-aware expert routing for adaptive denoising.
- **Three Anatomical Experts**:
  - Expert 1: Centrum semiovale (white matter)
  - Expert 2: Cortical surface (sulcal/gyral edges)
  - Expert 3: Grey–white matter junction
- **Denoising via Diffusion**: Conditional Latent Diffusion Models (CLDM) trained per expert.
- **Joint Optimization**: Combines reconstruction, perceptual, frequency, edge, and diversity-aware gating losses.

---

## 📁 File Structure

```bash
.
├── data_loader.py          # MRI dataset loader for paired 3T and 7T images
├── vqvae.py                # VQ-VAE (encoder/decoder, vector quantizer)
├── experts.py              # ExpertNetE1, E2, E3 with anatomical specializations
├── moe_model.py            # SwinTransformer-based patch encoder + token-aware gating
├── train_moediffsr.py      # Training pipeline
└── README.md
