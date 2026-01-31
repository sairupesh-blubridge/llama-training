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
du -sh /workspace/checkpoints_hindi_500m_indic/* 2>/dev/null || echo "No checkpoints yet"
