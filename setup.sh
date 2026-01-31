#!/bin/bash
# Nanotron Setup Script for Hindi Model Training
# Usage: bash nanotron_setup.sh
# This script is idempotent - safe to run multiple times

set -e

echo "==========================================="
echo "Nanotron Setup for Hindi Model Training"
echo "==========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "WARNING: Not running as root. Some commands may require sudo."
fi

# Display instance info
echo "Instance Information:"
echo "--------------------"
echo "Hostname: $(hostname)"
echo "GPUs detected:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Total GPUs: $GPU_COUNT"
echo ""

# Check disk space
DISK_SPACE=$(df -h / | awk 'NR==2 {print $4}')
echo "Available disk space: $DISK_SPACE"
echo ""

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "Installing tmux (for persistent sessions)..."
    apt-get update -qq && apt-get install -y tmux
fi

# Check Python version - nanotron requires Python 3.10 or 3.11 (NOT 3.12!)
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "Default Python version: $PYTHON_VERSION"

# Check if we need Python 3.11
PYTHON311_NEEDED=false
if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 12 ]; then
    echo "⚠ Python 3.12+ detected - nanotron requires Python 3.11 or earlier"
    PYTHON311_NEEDED=true
elif [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]; then
    echo "⚠ Python 3.9 or earlier detected - nanotron requires Python 3.10+"
    PYTHON311_NEEDED=true
fi

# Install Python 3.11 if needed
if [ "$PYTHON311_NEEDED" = true ]; then
    if ! command -v python3.11 &> /dev/null; then
        echo ""
        echo "Installing Python 3.11..."
        apt-get update -qq
        apt-get install -y software-properties-common
        add-apt-repository ppa:deadsnakes/ppa -y
        apt-get update -qq
        apt-get install -y python3.11 python3.11-venv python3.11-dev
        echo "✓ Python 3.11 installed"
    else
        echo "✓ Python 3.11 already installed"
    fi
    PYTHON_CMD="python3.11"
else
    PYTHON_CMD="python3"
fi

echo "Using Python command: $PYTHON_CMD"
$PYTHON_CMD --version
echo ""

# Set workspace to /workspace
WORKSPACE="/workspace"
echo "Using workspace at $WORKSPACE..."
cd $WORKSPACE

# Check if required files exist
echo "Checking required files..."
REQUIRED_FILES=(
    "$WORKSPACE/run_train_hindi.py"
    "$WORKSPACE/config_hindi_500m_indic.yaml"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ Found: $file"
    else
        echo "✗ Missing: $file"
        echo "ERROR: Required file not found!"
        exit 1
    fi
done
echo ""

# Check if virtual environment exists and is valid
VENV_VALID=false
if [ -d "nanotron_env" ]; then
    echo "Virtual environment exists, checking validity..."
    if [ -f "nanotron_env/bin/activate" ]; then
        # Check if it's using the right Python version
        VENV_PYTHON=$($WORKSPACE/nanotron_env/bin/python --version 2>&1 | awk '{print $2}')
        VENV_MINOR=$(echo $VENV_PYTHON | cut -d. -f2)
        if [ "$VENV_MINOR" -eq 11 ] || { [ "$VENV_MINOR" -eq 10 ] && [ "$PYTHON_MINOR" -eq 10 ]; }; then
            echo "✓ Valid virtual environment found (Python $VENV_PYTHON)"
            VENV_VALID=true
        else
            echo "⚠ Virtual environment uses incompatible Python version ($VENV_PYTHON)"
            echo "  Removing and recreating..."
            rm -rf nanotron_env
        fi
    fi
fi

if [ "$VENV_VALID" = false ]; then
    echo "Creating virtual environment with $PYTHON_CMD..."
    $PYTHON_CMD -m venv nanotron_env
    echo "✓ Virtual environment created"
else
    echo "Skipping virtual environment creation (already exists)"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source nanotron_env/bin/activate

# Check if packages are already installed
PACKAGES_INSTALLED=false
if python -c "import nanotron" 2>/dev/null; then
    echo "✓ Nanotron already installed, checking version..."
    PACKAGES_INSTALLED=true
else
    echo "Nanotron not found, will install packages..."
fi

if [ "$PACKAGES_INSTALLED" = false ]; then
    # Upgrade pip
    echo "Upgrading pip..."
    pip install --upgrade pip setuptools wheel -q

    # Detect CUDA version from nvidia-smi
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | grep -oP '\d+\.\d+' | head -n 1)
    echo ""
    echo "Detected CUDA Version: $CUDA_VERSION"

    # Map to PyTorch wheel
    if [[ "$CUDA_VERSION" == "12."* ]]; then
        CUDA_WHEEL="cu121"
    elif [[ "$CUDA_VERSION" == "11.8"* ]]; then
        CUDA_WHEEL="cu118"
    elif [[ "$CUDA_VERSION" == "11.7"* ]]; then
        CUDA_WHEEL="cu117"
    else
        CUDA_WHEEL="cu121"
        echo "WARNING: Unknown CUDA version, using cu121"
    fi

    # Install PyTorch with correct version
    echo ""
    echo "Installing PyTorch with CUDA support ($CUDA_WHEEL)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$CUDA_WHEEL

    # Install core dependencies
    echo ""
    echo "Installing core dependencies..."
    pip install \
        transformers \
        datasets \
        tokenizers \
        sentencepiece \
        accelerate \
        wandb \
        boto3 \
        safetensors \
        huggingface-hub \
        -q
    
    # Install flash-attention (required by nanotron, takes time to compile)
    echo ""
    echo "Installing flash-attention (this may take 5-10 minutes)..."
    pip install flash-attn==2.7.0.post2 --no-build-isolation
    
    # Install grouped_gemm (required for MoE models)
    echo ""
    echo "Installing grouped_gemm (for MoE support)..."
    pip install --no-build-isolation git+https://github.com/fanshiqing/grouped_gemm@main

    # Clone or update nanotron
    echo ""
    echo "Setting up nanotron repository..."
    if [ ! -d "nanotron" ]; then
        echo "Cloning nanotron repository..."
        git clone https://github.com/huggingface/nanotron.git
    else
        echo "Nanotron repository already exists"
    fi

    # Install nanotron in editable mode
    echo "Installing nanotron..."
    cd nanotron
    pip install -e . -q
    cd ..
    
    # Apply critical bug fix to nanotron
    echo ""
    echo "🔧 Applying critical bug fix to nanotron..."
    LLAMA_FILE="$WORKSPACE/nanotron/src/nanotron/models/llama.py"
    
    if [ -f "$LLAMA_FILE" ]; then
        # Check if patch already applied
        if grep -q "parametrizator = parametrizator_cls(config=config)" "$LLAMA_FILE" && ! grep -q "parametrizator = parametrizator_cls(config=config.model)" "$LLAMA_FILE"; then
            echo "✅ Bug fix already applied"
        else
            echo "📝 Patching llama.py (fixing parametrizator initialization)..."
            # Create backup
            cp "$LLAMA_FILE" "${LLAMA_FILE}.backup"
            
            # Apply fix: change config=config.model to config=config
            sed -i 's/parametrizator = parametrizator_cls(config=config\.model)/parametrizator = parametrizator_cls(config=config)/' "$LLAMA_FILE"
            
            # Verify the fix
            if grep -q "parametrizator = parametrizator_cls(config=config)" "$LLAMA_FILE"; then
                echo "✅ Bug fix applied successfully!"
                echo "   Fixed: llama.py line ~1095"
                echo "   Changed: config=config.model → config=config"
            else
                echo "⚠️  Warning: Bug fix may not have been applied correctly"
                echo "   Please check $LLAMA_FILE manually"
            fi
        fi
    else
        echo "⚠️  Warning: llama.py not found at expected location"
    fi
else
    echo "Skipping package installation (already installed)"
    
    # Still check if bug fix needs to be applied
    LLAMA_FILE="$WORKSPACE/nanotron/src/nanotron/models/llama.py"
    if [ -f "$LLAMA_FILE" ]; then
        if grep -q "parametrizator = parametrizator_cls(config=config.model)" "$LLAMA_FILE"; then
            echo ""
            echo "🔧 Nanotron bug fix not applied yet, applying now..."
            sed -i 's/parametrizator = parametrizator_cls(config=config\.model)/parametrizator = parametrizator_cls(config=config)/' "$LLAMA_FILE"
            echo "✅ Bug fix applied!"
        fi
    fi
fi

# Verify installation
echo ""
echo "==========================================="
echo "Verifying Installation"
echo "==========================================="

python3 << 'EOF'
import torch
import transformers
import nanotron

print(f"✓ PyTorch:        {torch.__version__}")
print(f"✓ Python:         {torch.sys.version.split()[0]}")
print(f"✓ CUDA Available: {torch.cuda.is_available()}")
print(f"✓ GPU Count:      {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}:         {torch.cuda.get_device_name(i)}")
print(f"✓ Transformers:   {transformers.__version__}")
print(f"✓ Nanotron:       Ready")
EOF

# Create directories for checkpoints and logs
echo ""
echo "Creating checkpoint and log directories..."
mkdir -p checkpoints_hindi_500m_indic
mkdir -p checkpoints_hindi_500m_test
mkdir -p logs

# Create training launcher script
echo ""
echo "Creating training launcher script..."

cat > $WORKSPACE/start_training.sh << 'LAUNCHER_EOF'
#!/bin/bash
# Training launcher for Hindi model

# Activate environment
source /workspace/nanotron_env/bin/activate

# Set environment variables
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Get GPU count
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Config file path (can be overridden with environment variable)
CONFIG_FILE="${CONFIG_FILE:-/workspace/config_hindi_500m_indic.yaml}"
TRAIN_SCRIPT="/workspace/run_train_hindi.py"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: Training script not found: $TRAIN_SCRIPT"
    exit 1
fi

echo "==========================================="
echo "Starting Hindi Model Training"
echo "==========================================="
echo "GPUs:          $GPU_COUNT"
echo "Config:        $CONFIG_FILE"
echo "Train Script:  $TRAIN_SCRIPT"
echo "Time:          $(date)"
echo "==========================================="

# Create log file with timestamp
LOG_FILE="/workspace/logs/training_$(date +%Y%m%d_%H%M%S).log"

# Run training
cd /workspace
torchrun --nproc_per_node=$GPU_COUNT \
    $TRAIN_SCRIPT \
    --config-file "$CONFIG_FILE" \
    2>&1 | tee "$LOG_FILE"
LAUNCHER_EOF

chmod +x $WORKSPACE/start_training.sh

# Create monitoring script
cat > $WORKSPACE/monitor.sh << 'MONITOR_EOF'
#!/bin/bash
# Monitor training progress

echo "Training Monitoring"
echo "==================="
echo ""

# GPU Usage
echo "GPU Usage:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv

echo ""
echo "Recent Training Logs:"
tail -n 30 /workspace/logs/training_*.log 2>/dev/null || echo "No training logs found yet"

echo ""
echo "Disk Usage:"
df -h /

echo ""
echo "Checkpoint Storage:"
du -sh /workspace/checkpoints_*/* 2>/dev/null || echo "No checkpoints yet"
MONITOR_EOF

chmod +x $WORKSPACE/monitor.sh

# Check data directory
echo ""
echo "==========================================="
echo "Data Directory Check"
echo "==========================================="
echo ""

DATA_DIR="/workspace/2b_hindi_indicbart_shards"
if [ -d "$DATA_DIR" ]; then
    SHARD_COUNT=$(ls -1 $DATA_DIR/hindi_shard_*.npy 2>/dev/null | wc -l)
    if [ $SHARD_COUNT -gt 0 ]; then
        echo "✓ Data directory exists: $DATA_DIR"
        echo "✓ Found $SHARD_COUNT shard files"
        du -sh $DATA_DIR
    else
        echo "⚠ Data directory exists but no shards found"
        echo "  Run fast_shard.py to create data shards"
    fi
else
    echo "⚠ Data directory not found: $DATA_DIR"
    echo "  Run fast_shard.py to create data shards"
fi

# Print final instructions
echo ""
echo "==========================================="
echo "Setup Complete!"
echo "==========================================="
echo ""
echo "Workspace: $WORKSPACE"
echo ""
echo "Files Verified:"
echo "  ✓ run_train_hindi.py"
echo "  ✓ config_hindi_500m_indic.yaml"
echo ""
echo "Next Steps:"
echo "----------"
echo ""
echo "1. Prepare data shards (if not done):"
echo "   source nanotron_env/bin/activate"
echo "   python fast_shard.py"
echo ""
echo "2. For TEST training (200 steps with current data):"
echo "   CONFIG_FILE=/workspace/config_hindi_500m_test.yaml ./start_training.sh"
echo ""
echo "3. For FULL training (19073 steps - requires more data):"
echo "   ./start_training.sh"
echo ""
echo "4. Monitor training:"
echo "   ./monitor.sh"
echo ""
echo "5. Use tmux for long training sessions:"
echo "   tmux new -s training"
echo "   ./start_training.sh"
echo "   # Detach: Ctrl+B, then D"
echo "   # Reattach: tmux attach -t training"
echo ""
echo "Helper Scripts Created:"
echo "  - start_training.sh : Launch training"
echo "  - monitor.sh        : Check status"
echo ""
echo "GPU Configuration:"
echo "  Available GPUs: $GPU_COUNT"
echo "  Config expects: Check parallelism.dp in your YAML config"
echo ""
echo "==========================================="
echo ""
