# Quick Start Guide

## New Instance Setup (One Command)

```bash
# Clone and setup everything
git clone https://github.com/sairupesh-blubridge/llama-training.git
cd llama-training
bash setup.sh
```

**That's it!** The setup script handles everything.

## What Gets Installed

- Python 3.11 (auto-installed if needed)
- PyTorch with CUDA support
- Nanotron framework (with bug fix)
- Flash-attention
- All dependencies

**Time:** ~12-15 minutes

## Running Training

```bash
# Activate environment
source nanotron_env/bin/activate

# Prepare data (first time only)
python prepare_data.py

# Start training
bash train.sh
```

## Configs Available

- **`config_hindi_100m.yaml`** - Small test model (100M params)
- **`config_hindi_500m_indic.yaml`** - Full model (500M params)
- **`config_hindi_500m_test.yaml`** - Test run (200 steps)

## Monitoring

```bash
bash monitor.sh
```

## Re-running on New Instance

Just run `setup.sh` again - it's idempotent and will skip already completed steps!

---

**See [README.md](README.md) for full documentation.**
