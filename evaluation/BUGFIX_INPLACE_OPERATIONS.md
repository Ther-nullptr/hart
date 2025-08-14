# Bug Fix: InferenceMode Inplace Operation Error

## Problem Description

When running HART evaluation scripts, you may encounter the following error:

```
Error generating image for prompt 'a painting by Caravaggio with a scene of a young e...': 
Inplace update to inference tensor outside InferenceMode is not allowed.
You can make a clone to get a normal tensor before doing inplace update.
See https://github.com/pytorch/rfcs/pull/17 for more details.
```

## Root Cause

The error occurs because the code was using inplace operations (`mul_()`) on tensors while inside PyTorch's `torch.inference_mode()` context. Starting from PyTorch 1.9+, inplace operations on tensors are not allowed in inference mode to prevent accidental modifications to tensors that might be used elsewhere.

## Fix Applied

The problematic code was in the `_save_image` and `_save_images_batch` methods:

### Before (Problematic)
```python
def _save_image(self, img_tensor: torch.Tensor, filepath: str):
    """Save image tensor to file."""
    img_np = img_tensor.mul_(255).cpu().numpy()  # ❌ Inplace operation
    # ...
```

### After (Fixed)
```python
def _save_image(self, img_tensor: torch.Tensor, filepath: str):
    """Save image tensor to file."""
    # Clone tensor to avoid inplace operation in inference mode
    img_np = img_tensor.clone().mul(255).cpu().numpy()  # ✅ Non-inplace operation
    # ...
```

## Files Modified

1. **`compute_fid_improved.py`** - Line 213
2. **`evaluate_geneval_improved.py`** - Line 395
3. **`evaluate_hart.py`** - Line 100

## Technical Details

### What Changed
- `tensor.mul_(255)` → `tensor.clone().mul(255)`
- Added `.clone()` before the multiplication to create a new tensor
- This avoids modifying the original tensor in-place

### Why This Works
- `.clone()` creates a new tensor that is not tracked by the inference mode
- `.mul(255)` (non-inplace) creates a new tensor with scaled values
- The original tensor remains unchanged, satisfying inference mode requirements

### Performance Impact
- Minimal: The clone operation is only on individual image tensors
- Memory overhead is temporary and gets cleaned up immediately after conversion
- No impact on generation speed or quality

## Verification

After applying this fix, the evaluation scripts should run without the inplace operation error. You can verify by running:

```bash
# Test FID evaluation
python compute_fid_improved.py --compute_fid_only /path/to/real /path/to/generated

# Test GenEval evaluation
python evaluate_geneval_improved.py --generate_prompts --model_path /path/to/model

# Test comprehensive evaluation
./run_hart_evaluation.sh
```

## Related Information

- **PyTorch RFC**: https://github.com/pytorch/rfcs/pull/17
- **Inference Mode Documentation**: https://pytorch.org/docs/stable/generated/torch.inference_mode.html
- **Related PyTorch Issue**: This is a common issue when upgrading to newer PyTorch versions

## Prevention

To prevent similar issues in future code:

1. **Use non-inplace operations** in inference contexts:
   - ✅ `tensor.mul(scalar)` instead of `tensor.mul_(scalar)`
   - ✅ `tensor + value` instead of `tensor.add_(value)`
   - ✅ `tensor.clamp(0, 1)` instead of `tensor.clamp_(0, 1)`

2. **Clone tensors** when you need to modify them:
   ```python
   # If you must modify
   modified_tensor = original_tensor.clone().mul_(255)
   ```

3. **Use `.detach()`** when moving tensors out of computational graph:
   ```python
   # For tensors that don't need gradients
   output_tensor = model_output.detach().mul(255)
   ```

## Compatibility

This fix is compatible with:
- ✅ PyTorch 1.9+
- ✅ PyTorch 2.0+
- ✅ All CUDA/CPU configurations
- ✅ Both training and inference modes