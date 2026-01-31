# Nanotron Bug Fix

## Issue
Nanotron has a bug in `src/nanotron/models/llama.py` line 1095 where it passes `config.model` (ModelArgs) to `StandardParametrizator` which expects the full `Config` object.

## Error Without Fix
```
AttributeError: 'ModelArgs' object has no attribute 'model'
```

## The Fix
**File**: `nanotron/src/nanotron/models/llama.py`  
**Line**: 1095

**Before**:
```python
parametrizator = parametrizator_cls(config=config.model)
```

**After**:
```python
parametrizator = parametrizator_cls(config=config)
```

## Automatic Application
The `setup.sh` script automatically applies this fix after cloning nanotron. No manual intervention needed.

## Manual Fix (if needed)
```bash
cd nanotron/src/nanotron/models
# Edit llama.py line 1095
# Change: config=config.model → config=config
```

---
This fix is critical for training to work. Without it, all training runs will fail during model initialization.
