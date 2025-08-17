"""
Enhanced HART model with improved caching mechanisms inspired by VAR's implementation.
Provides flexible multi-stage caching with separate control over attention and MLP layers.
"""

import math
from typing import Optional, Tuple, Union, List, Dict, Any
import torch
import torch.nn as nn
import numpy as np

from .cache_config import HARTCacheConfig, CacheStatistics
from .models.transformer.hart_transformer_t2i import HARTForT2I
from .models.transformer.configuration import HARTForT2IConfig


class EnhancedHARTForT2I(HARTForT2I):
    """
    Enhanced HART model with improved caching capabilities
    """
    
    def __init__(self, config: HARTForT2IConfig, cache_config: Optional[HARTCacheConfig] = None):
        super().__init__(config)
        
        # Enhanced cache configuration
        self.cache_config = cache_config or HARTCacheConfig()
        self.cache_config.validate()
        
        # Cache state management
        self.cache_statistics = CacheStatistics()
        self.adaptive_thresholds = {}
        self.similarity_history = {'attn': [], 'mlp': []}
        
        # Override original cache settings
        self.use_cache = self.cache_config.is_cache_enabled()
        
        # Initialize enhanced cache structures
        self._init_enhanced_cache()
    
    def _init_enhanced_cache(self):
        """Initialize enhanced cache structures"""
        depth = self.depth
        
        # Enhanced cache storage
        self.enhanced_cache_mlp = [None] * depth
        self.enhanced_cache_attn = [None] * depth
        
        # Similarity tracking with history
        self.enhanced_similarity_mlp = [[0.0] * 10 for _ in range(depth)]
        self.enhanced_similarity_attn = [[0.0] * 10 for _ in range(depth)]
        
        # Adaptive threshold tracking
        if self.cache_config.adaptive_threshold:
            for layer_idx in range(depth):
                self.adaptive_thresholds[layer_idx] = {
                    'attn': self.cache_config.threshold,
                    'mlp': self.cache_config.threshold
                }
    
    def set_cache_config(self, cache_config: HARTCacheConfig):
        """Update cache configuration"""
        cache_config.validate()
        self.cache_config = cache_config
        self.use_cache = cache_config.is_cache_enabled()
        self._init_enhanced_cache()
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        base_stats = {
            'cache_hit_rate': self.cache_statistics.get_cache_hit_rate(),
            'similarity_stats': self.cache_statistics.get_similarity_stats(),
            'config': {
                'skip_stages': self.cache_config.skip_stages,
                'cache_stages': self.cache_config.cache_stages,
                'enable_attn_cache': self.cache_config.enable_attn_cache,
                'enable_mlp_cache': self.cache_config.enable_mlp_cache,
                'threshold': self.cache_config.threshold,
                'interpolation_mode': self.cache_config.interpolation_mode
            }
        }
        
        if self.cache_config.adaptive_threshold:
            base_stats['adaptive_thresholds'] = self.adaptive_thresholds
        
        return base_stats
    
    def reset_cache_statistics(self):
        """Reset cache statistics"""
        self.cache_statistics.reset()
        self.similarity_history = {'attn': [], 'mlp': []}
    
    def update_adaptive_threshold(self, layer_idx: int, layer_type: str, similarity: float):
        """Update adaptive threshold based on similarity history"""
        if not self.cache_config.adaptive_threshold:
            return
        
        # Add to history
        self.similarity_history[layer_type].append(similarity)
        
        # Keep only recent history
        if len(self.similarity_history[layer_type]) > self.cache_config.similarity_window:
            self.similarity_history[layer_type] = self.similarity_history[layer_type][-self.cache_config.similarity_window:]
        
        # Update threshold to 80th percentile of recent similarities
        if len(self.similarity_history[layer_type]) >= 10:
            new_threshold = np.percentile(self.similarity_history[layer_type], 80)
            self.adaptive_thresholds[layer_idx][layer_type] = new_threshold
    
    def get_effective_threshold(self, layer_idx: int, layer_type: str) -> float:
        """Get effective threshold (adaptive or fixed)"""
        if self.cache_config.adaptive_threshold and layer_idx in self.adaptive_thresholds:
            return self.adaptive_thresholds[layer_idx][layer_type]
        return self.cache_config.threshold
    
    def enhanced_feature_interpolate(self, x: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
        """
        Enhanced feature interpolation with configurable modes
        
        Args:
            x: Input tensor [B, L, C]
            mode: Interpolation mode (overrides config if provided)
            
        Returns:
            Interpolated tensor [B, L_new, C]
        """
        if mode is None:
            mode = self.cache_config.interpolation_mode
        
        # Use the existing feature_interpolate but with configurable mode
        from .networks.basic_hart import next_patch_size
        
        hw = int(math.sqrt(x.shape[1]))
        if hw not in next_patch_size:
            return x
            
        result = x.view(x.shape[0], hw, hw, -1)
        
        # Enhanced interpolation with align_corners handling
        align_corners = mode != 'nearest'
        
        result = torch.nn.functional.interpolate(
            result.permute(0, 3, 1, 2), 
            size=(next_patch_size[hw], next_patch_size[hw]), 
            mode=mode, 
            align_corners=align_corners if mode in ['bilinear', 'bicubic'] else None
        ).permute(0, 2, 3, 1)
        
        result = result.view(result.shape[0], -1, result.shape[-1])
        return result
    
    def should_use_cache(self, stage: int, layer_idx: int, layer_type: str, similarity: float) -> bool:
        """
        Enhanced cache usage decision with stage-aware logic
        
        Args:
            stage: Current stage number
            layer_idx: Layer index
            layer_type: 'attn' or 'mlp'
            similarity: Computed similarity
            
        Returns:
            Whether to use cached result
        """
        # Check if layer type caching is enabled
        if layer_type == 'attn' and not self.cache_config.enable_attn_cache:
            return False
        if layer_type == 'mlp' and not self.cache_config.enable_mlp_cache:
            return False
        
        # Check if stage should be skipped
        if stage in self.cache_config.skip_stages:
            return True
        
        # Check if stage is in cache stages and similarity threshold is met
        if stage in self.cache_config.cache_stages:
            threshold = self.get_effective_threshold(layer_idx, layer_type)
            return similarity > threshold
        
        return False
    
    def calibrate_thresholds(self, num_samples: Optional[int] = None) -> Dict[str, float]:
        """
        Calibrate optimal thresholds based on similarity distribution
        
        Args:
            num_samples: Number of samples to use for calibration
            
        Returns:
            Dictionary with recommended thresholds
        """
        if num_samples is None:
            num_samples = self.cache_config.calibration_samples
        
        print(f"Starting threshold calibration with {num_samples} samples...")
        
        # Temporarily enable adaptive thresholding for calibration
        original_adaptive = self.cache_config.adaptive_threshold
        self.cache_config.adaptive_threshold = True
        
        # Reset statistics
        self.reset_cache_statistics()
        
        # Run inference samples to collect statistics
        with torch.no_grad():
            for i in range(num_samples):
                try:
                    # Generate a sample (you may need to adjust this based on your setup)
                    _ = self.autoregressive_infer_cfg(
                        B=1,
                        label_B=None,
                        g_seed=i,
                        cfg=1.5
                    )
                    
                    if (i + 1) % 10 == 0:
                        print(f"Calibration progress: {i+1}/{num_samples}")
                        
                except Exception as e:
                    print(f"Error during calibration sample {i}: {e}")
                    continue
        
        # Restore original adaptive setting
        self.cache_config.adaptive_threshold = original_adaptive
        
        # Analyze results and recommend thresholds
        similarity_stats = self.cache_statistics.get_similarity_stats()
        
        recommendations = {}
        for layer_type in ['attn', 'mlp']:
            if layer_type in similarity_stats and similarity_stats[layer_type]:
                stats = similarity_stats[layer_type]
                # Recommend threshold at 70th percentile for good speed/quality balance
                recommendations[f'{layer_type}_threshold'] = stats.get('p80', self.cache_config.threshold)
            else:
                recommendations[f'{layer_type}_threshold'] = self.cache_config.threshold
        
        print("Calibration complete. Recommended thresholds:")
        for key, value in recommendations.items():
            print(f"  {key}: {value:.3f}")
        
        return recommendations
    
    def save_cache_config(self, path: str):
        """Save current cache configuration to file"""
        self.cache_config.to_json(path)
        print(f"Cache configuration saved to {path}")
    
    def load_cache_config(self, path: str):
        """Load cache configuration from file"""
        self.cache_config = HARTCacheConfig.from_json(path)
        self.use_cache = self.cache_config.is_cache_enabled()
        self._init_enhanced_cache()
        print(f"Cache configuration loaded from {path}")
    
    @torch.no_grad()
    def benchmark_performance(self, num_samples: int = 10, batch_size: int = 1) -> Dict[str, Any]:
        """
        Benchmark model performance with current cache configuration
        
        Args:
            num_samples: Number of samples to generate for benchmarking
            batch_size: Batch size for generation
            
        Returns:
            Performance metrics dictionary
        """
        import time
        
        # Warm up
        _ = self.autoregressive_infer_cfg(B=1, label_B=None, g_seed=42)
        
        # Reset statistics
        self.reset_cache_statistics()
        
        times = []
        memory_usage = []
        
        for i in range(num_samples):
            # Monitor memory before
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_before = torch.cuda.memory_allocated()
            
            start_time = time.time()
            
            # Generate sample
            _ = self.autoregressive_infer_cfg(
                B=batch_size,
                label_B=None,
                g_seed=i,
                cfg=1.5
            )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
            
            # Monitor memory after
            if torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated()
                memory_usage.append((mem_after - mem_before) / 1024 / 1024)  # MB
        
        # Calculate statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        mean_memory = np.mean(memory_usage) if memory_usage else 0
        throughput = batch_size / mean_time
        
        results = {
            'benchmark': {
                'mean_time': mean_time,
                'std_time': std_time,
                'mean_memory_mb': mean_memory,
                'throughput_img_per_sec': throughput,
                'cache_hit_rate': self.cache_statistics.get_cache_hit_rate()
            },
            'cache_config': {
                'skip_stages': self.cache_config.skip_stages,
                'cache_stages': self.cache_config.cache_stages,
                'threshold': self.cache_config.threshold,
                'interpolation_mode': self.cache_config.interpolation_mode
            },
            'similarity_stats': self.cache_statistics.get_similarity_stats()
        }
        
        return results


def create_enhanced_hart_model(
    config: HARTForT2IConfig,
    cache_preset: str = "original-cache",
    custom_cache_config: Optional[HARTCacheConfig] = None
) -> EnhancedHARTForT2I:
    """
    Factory function to create enhanced HART model with caching
    
    Args:
        config: HART model configuration
        cache_preset: Name of cache preset to use
        custom_cache_config: Custom cache configuration (overrides preset)
        
    Returns:
        Enhanced HART model with caching
    """
    if custom_cache_config is not None:
        cache_config = custom_cache_config
    else:
        cache_config = HARTCacheConfig.from_preset(cache_preset)
    
    return EnhancedHARTForT2I(config, cache_config)