# Llama Training - Hindi Language Model

Production-ready setup for training LLaMA models on Hindi text using Nanotron framework.

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/sairupesh-blubridge/llama-training.git
cd llama-training

# 2. Set HuggingFace token (for data download)
export HF_TOKEN=your_huggingface_token_here

# 3. Run setup (installs all dependencies)
bash setup.sh

# 4. Prepare training data
source nanotron_env/bin/activate
python prepare_data.py

# 5. Start training
bash train.sh
```

## 📋 Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: Will auto-install Python 3.11 if needed
- **GPUs**: NVIDIA GPUs with CUDA 12.1+
- **RAM**: 32GB+ recommended
- **Disk**: 50GB+ free space

## 🛠️ What's Included

- **`setup.sh`** - One-command setup script (handles everything)
- **`train.sh`** - Training launcher
- **`prepare_data.py`** - Data preparation script
- **`monitor.sh`** - Training monitor
- **`config_hindi_100m.yaml`** - 100M parameter model config
- **`run_train_hindi.py`** - Main training script

## 📦 Setup Details

The `setup.sh` script automatically:

1. ✅ Detects and installs Python 3.11 (if needed)
2. ✅ Creates virtual environment
3. ✅ Installs PyTorch with CUDA support
4. ✅ Installs nanotron framework
5. ✅ Installs flash-attention (optimized kernels)
6. ✅ Installs grouped_gemm (for MoE support)
7. ✅ Applies critical bug fix to nanotron
8. ✅ Sets up helper scripts

**Time**: ~12-15 minutes first run, <1 minute on re-runs (idempotent)

## 🎯 Training Models

### Quick Test (Minimal Model)
```bash
CONFIG_FILE=config_hindi_100m.yaml bash train.sh
```

### Full Training
1. Edit config file for your model size
2. Prepare more data shards if needed
3. Run training:
```bash
CONFIG_FILE=config_hindi_500m_indic.yaml bash train.sh
```

## 📊 Monitoring

```bash
# Watch training progress
bash monitor.sh

# Or attach to training session
tmux attach -t training
```

## 🐛 Known Issues & Solutions

### Issue: Out of Memory (OOM)
**Solution**: Reduce model size or batch size in config, or use larger GPUs (24GB+)

### Issue: Python 3.12 detected
**Solution**: Setup script auto-installs Python 3.11 - just re-run `setup.sh`

### Issue: Flash-attention compilation fails
**Solution**: Ensure CUDA toolkit is installed (`nvidia-smi` should work)

## 📁 Project Structure

```
llama-training/
├── setup.sh              # Main setup script
├── train.sh              # Training launcher
├── prepare_data.py       # Data preparation
├── run_train_hindi.py    # Training script
├── monitor.sh            # Monitor helper
├── config_*.yaml         # Model configurations
└── README.md             # This file
```

## 🔧Advanced Configuration

Edit the YAML config files to customize:
- Model size (layers, hidden dims)
- Batch size and sequence length
- Learning rate and schedule
- Parallelism settings (DP, TP, PP)
- Checkpoint intervals

## 💡 Tips

1. **First run**: Use small test config to validate setup
2. **Production**: Use 24GB+ GPUs for meaningful training
3. **Checkpoints**: Saved every N steps (configurable)
4. **Resume**: Set `resume_checkpoint_path` in config

## 🤝 Contributing

This repo contains fixes for nanotron compatibility issues. The main fix is in `setup.sh` which patches nanotron's `llama.py` file.

## 📝 License

MIT License - see original nanotron repository for framework license.

## ⚠️ Important Notes

- **Nanotron Bug Fix**: This setup includes a critical fix for nanotron's parametrizator initialization
- **GPU Memory**: Minimum 12GB per GPU, 24GB+ recommended for production models
- **Data**: Script uses FineWeb2 Hindi dataset by default

## 🆘 Support

If you encounter issues:
1. Check `setup.sh` output for errors
2. Verify GPU drivers: `nvidia-smi`
3. Check Python version: `python3.11 --version`
4. Review config file for typos

---

**Maintained by**: [@sairupesh-blubridge](https://github.com/sairupesh-blubridge)  
**Based on**: [Nanotron](https://github.com/huggingface/nanotron) by HuggingFace
