# HART Cache Mapping Update

## 更新内容

已成功更新HART缓存系统的stage映射，使其适应更大的分辨率范围和更细粒度的缓存控制。

## 修改的映射变量

### length2iteration 映射
```python
# 旧版本 (VAR风格，10个stage)
length2iteration = {
    1: 0, 4: 1, 9: 2, 16: 3, 25: 4, 36: 5, 64: 6, 100: 7, 169: 8, 256: 9,
}

# 新版本 (HART扩展，15个stage)
length2iteration = {
    1: 0, 4: 1, 9: 2, 16: 3, 25: 4, 49: 6, 81: 7, 144: 8, 
    256: 9, 441: 10, 729: 11, 1296: 12, 2304: 13, 4096: 14
}
```

### next_patch_size 映射
```python
# 旧版本
next_patch_size = {
    1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 8, 8: 10, 10: 13, 13: 16
}

# 新版本
next_patch_size = {
    1: 2, 2: 3, 3: 4, 4: 5, 5: 7, 7: 9, 9: 12, 12: 16, 
    16: 21, 21: 27, 27: 36, 36: 48, 48: 64
}
```

## 分辨率对应关系

| Stage | Resolution | 用途 |
|-------|------------|------|
| 1 | 1×1 | 初始阶段 |
| 4 | 2×2 | 早期阶段 |
| 9 | 3×3 | 早期阶段 |
| 16 | 4×4 | 早中期阶段 |
| 25 | 5×5 | 中期阶段 |
| 49 | 7×7 | 中期阶段，开始适合缓存 |
| 81 | 9×9 | 中后期阶段，缓存效果好 |
| 144 | 12×12 | 后期阶段，主要缓存目标 |
| 256 | 16×16 | 后期阶段，常用缓存 |
| 441 | 21×21 | 高分辨率阶段 |
| 729 | 27×27 | 很高分辨率阶段 |
| 1296 | 36×36 | 超高分辨率阶段 |
| 2304 | 48×48 | 极高分辨率阶段 |
| 4096 | 64×64 | 最大分辨率阶段 |

## 更新的文件

### 核心实现文件
1. **`hart/modules/networks/basic_hart.py`** - 原始basic_hart的映射更新
2. **`hart/modules/networks/basic_hart_enhanced.py`** - 增强版本的映射更新
3. **`hart/modules/models/transformer/hart_transformer_t2i.py`** - 原始transformer的缓存数组大小更新
4. **`hart/modules/models/transformer/hart_transformer_t2i_enhanced.py`** - 增强transformer的缓存数组和预设更新

### 配置文件
5. **`hart_cache_presets.json`** - 所有预设配置的stage值更新
6. **`README_VAR_STYLE_CACHING.md`** - 文档中的stage信息更新

## 预设配置更新

### 保守型缓存 (Conservative)
- Skip: [256] → [256] (保持)
- Cache: [169] → [144]

### 原始缓存 (Original)
- Skip: [169, 256] → [144, 256]
- Cache: [100, 169] → [81, 144]

### 激进缓存 (Aggressive)
- Skip: [64, 100, 169, 256] → [49, 81, 144, 256]
- Cache: [36, 64, 100, 169] → [25, 49, 81, 144]

### 超快速缓存 (Ultra-fast)
- Skip: [36, 64, 100, 169, 256] → [25, 49, 81, 144, 256, 441]
- Cache: [25, 36, 64, 100, 169] → [16, 25, 49, 81, 144, 256]

### 注意力专用 (Attention-only)
- Skip: [169, 256] → [144, 256]
- Cache: [100, 169] → [81, 144]

### 内存高效 (Memory-efficient)
- Skip: [169, 256] → [144, 256]
- Cache: [169] → [144]

## 缓存数组大小更新

相似度跟踪数组从10个元素扩展到15个元素：
```python
# 旧版本
self.cache_similarity_mlp = [[0.0] * 10 for _ in range(depth)]
self.cache_similarity_attn = [[0.0] * 10 for _ in range(depth)]

# 新版本
self.cache_similarity_mlp = [[0.0] * 15 for _ in range(depth)]
self.cache_similarity_attn = [[0.0] * 15 for _ in range(depth)]
```

## 兼容性说明

### 向后兼容
- 原有的低分辨率stage (1, 4, 9, 16, 25, 256) 保持相同的iteration索引
- 现有的缓存数据结构仍然有效
- 旧的配置文件会自动适配到最接近的新stage

### 新增功能
- 支持更高分辨率的生成和缓存
- 更细粒度的缓存控制
- 扩展的预设配置选项

## 使用示例

### 使用新的高分辨率缓存
```bash
# 高分辨率激进缓存
python enhanced_inference_cli.py \
    --cache-skip-stages 441 729 1296 2304 4096 \
    --cache-cache-stages 144 256 441 729 \
    --cache-threshold 0.6

# 超高分辨率保守缓存
python enhanced_inference_cli.py \
    --cache-skip-stages 2304 4096 \
    --cache-cache-stages 1296 2304 \
    --cache-threshold 0.8
```

### 使用更新的预设
```bash
# 使用更新的激进预设
python enhanced_inference_cli.py --cache-preset aggressive

# 使用更新的超快速预设
python enhanced_inference_cli.py --cache-preset ultra-fast
```

## 测试建议

1. **性能测试**: 对比新旧映射的性能差异
2. **质量评估**: 验证高分辨率缓存对图像质量的影响
3. **内存使用**: 监控扩展缓存数组的内存占用
4. **兼容性测试**: 确保旧配置仍能正常工作

这次更新显著扩展了HART缓存系统的能力，支持更高分辨率的生成和更精细的缓存控制，同时保持了与现有代码的兼容性。