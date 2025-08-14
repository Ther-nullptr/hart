# GenEval Format Corrections

## Issue Identified

The original GenEval implementation in `evaluate_geneval_improved.py` was based on assumptions about the GenEval format rather than the actual dataset structure. After examining the official GenEval repository at https://github.com/djghosh13/geneval, several critical issues were found.

## Actual GenEval Format

The real GenEval dataset uses a JSONL format with the following structure:

### File Format
- **File**: `evaluation_metadata.jsonl` (single JSONL file)
- **Structure**: One JSON object per line, not a directory structure

### Prompt Structure Examples

#### 1. Single Object
```json
{"tag": "single_object", "include": [{"class": "bench", "count": 1}], "prompt": "a photo of a bench"}
```

#### 2. Counting
```json
{"tag": "counting", "include": [{"class": "clock", "count": 2}], "exclude": [{"class": "clock", "count": 3}], "prompt": "a photo of two clocks"}
```

#### 3. Colors
```json
{"tag": "colors", "include": [{"class": "car", "count": 1, "color": "pink"}], "prompt": "a photo of a pink car"}
```

#### 4. Position (Key Format Difference)
```json
{"tag": "position", "include": [{"class": "teddy bear", "count": 1}, {"class": "dog", "count": 1, "position": ["right of", 0]}], "prompt": "a photo of a dog right of a teddy bear"}
```

#### 5. Color Attributes (Multi-object with colors)
```json
{"tag": "color_attr", "include": [{"class": "wine glass", "count": 1, "color": "purple"}, {"class": "apple", "count": 1, "color": "black"}], "prompt": "a photo of a purple wine glass and a black apple"}
```

## Key Corrections Made

### 1. **Data Loading**
**Before**: Custom prompt generation with assumed format
```python
# Wrong: Generated synthetic prompts
prompts = GenEvalDataGenerator.generate_all_prompts()
```

**After**: Load actual GenEval JSONL file
```python
# Correct: Load official GenEval metadata
prompts = GenEvalDataManager.load_geneval_prompts("evaluation_metadata.jsonl")
```

### 2. **Position Format**
**Before**: Assumed simple string format
```python
# Wrong assumption
"position": "left of"
```

**After**: Array with relation and target index
```python
# Correct format: ["relation", target_group_index]
"position": ["right of", 0]
```

### 3. **Task Tags**
**Before**: Used assumed tag names
```python
# Wrong tags
"counting", "color", "position", "cooccurrence"
```

**After**: Actual GenEval tags
```python
# Correct tags from official dataset
"single_object", "two_object", "counting", "colors", "position", "color_attr"
```

### 4. **Exclude Logic**
**Before**: Not properly implemented
```python
# Incomplete exclude handling
for req in metadata.get('exclude', []):
    # Basic implementation
```

**After**: Proper exclude constraint evaluation
```python
# Correct: exclude means count should be < specified value
max_allowed = req['count'] - 1
if found_count >= req['count']:
    correct = False
```

### 5. **Directory Structure**
**Before**: Assumed nested directory structure
```
prompts/
├── 0/
│   ├── metadata.jsonl
│   └── samples/
├── 1/
│   └── ...
```

**After**: Creates structure from JSONL file
```python
# Convert JSONL to evaluation structure
eval_dir = GenEvalDataManager.create_evaluation_structure(prompts, output_dir)
```

## New Corrected Implementation

### Files Created
1. **`evaluate_geneval_corrected.py`** - Uses actual GenEval format
2. **`run_hart_evaluation_corrected.sh`** - Updated evaluation script

### Key Features of Corrected Version

#### 1. **Automatic Data Download**
```python
GenEvalDataManager.download_geneval_metadata(metadata_path)
```

#### 2. **Proper Position Evaluation**
```python
# Handle actual GenEval position format: ["relation", target_index]
expected_rel, target_group_idx = req['position']
```

#### 3. **Correct Task Categories**
- `single_object`: Single object detection
- `two_object`: Two object co-occurrence  
- `counting`: Object counting with exclude constraints
- `colors`: Single object with color
- `position`: Spatial relationships with proper indexing
- `color_attr`: Multi-object with individual colors

#### 4. **Official Dataset Integration**
```bash
# Automatically downloads official GenEval metadata
--download_geneval
```

## Usage of Corrected Version

### Quick Start
```bash
# Use corrected evaluation script
./run_hart_evaluation_corrected.sh
```

### Manual Usage
```bash
# Generate images and evaluate with official GenEval dataset
python evaluate_geneval_corrected.py \
    --model_path /path/to/hart/model \
    --detector_checkpoint /path/to/checkpoint.pth \
    --download_geneval \
    --generate_images
```

## Performance Comparison

### Original (Incorrect) Implementation
- Used synthetic prompts that didn't match GenEval format
- Position relationships were incorrectly interpreted
- Task categories didn't align with official evaluation
- Results were not comparable to published GenEval benchmarks

### Corrected Implementation
- Uses official GenEval evaluation metadata
- Proper position relationship evaluation with target indexing
- Correct task categorization matching the paper
- Results directly comparable to GenEval benchmarks in literature

## Verification

The corrected implementation can be verified by:

1. **Data Format**: Compare generated structure with official GenEval
2. **Task Distribution**: Check that task counts match the official dataset
3. **Evaluation Logic**: Verify position relationships use proper indexing
4. **Results Comparability**: Results should align with GenEval baselines

## Migration Guide

### From Original to Corrected Version

1. **Replace evaluation script**:
   ```bash
   # Old
   python evaluate_geneval_improved.py
   
   # New  
   python evaluate_geneval_corrected.py --download_geneval
   ```

2. **Update configuration**:
   ```bash
   # Old
   ./run_hart_evaluation.sh
   
   # New
   ./run_hart_evaluation_corrected.sh
   ```

3. **Results interpretation**:
   - Results from corrected version are comparable to published GenEval scores
   - Task-specific accuracies now match official GenEval categories
   - Position evaluation is more accurate with proper spatial indexing

This correction ensures that HART evaluation results are directly comparable to other models evaluated on the GenEval benchmark.