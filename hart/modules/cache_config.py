"""
Enhanced cache configuration for HART, inspired by VAR's caching system.
Provides flexible multi-stage caching with separate control over attention and MLP layers.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import json
import os


@dataclass
class HARTCacheConfig:
    """
    Enhanced cache configuration for HART transformer
    
    Args:
        skip_stages: List of stages to skip computation (e.g., [169, 256])
        cache_stages: List of stages to cache results (e.g., [100, 169])
        enable_attn_cache: Whether to enable attention layer caching
        enable_mlp_cache: Whether to enable MLP layer caching
        threshold: Similarity threshold for cache usage
        max_skip_stages: Maximum number of stages to skip
        adaptive_threshold: Whether to use adaptive thresholding
        interpolation_mode: Interpolation mode ('bilinear', 'nearest', 'bicubic')
        calibration_samples: Number of samples for threshold calibration
        similarity_window: Window size for adaptive threshold calculation
    """
    skip_stages: List[int] = None
    cache_stages: List[int] = None
    enable_attn_cache: bool = True
    enable_mlp_cache: bool = True
    threshold: float = 0.7
    max_skip_stages: int = 9
    adaptive_threshold: bool = False
    interpolation_mode: str = 'bilinear'
    calibration_samples: int = 100
    similarity_window: int = 20
    
    def __post_init__(self):
        if self.skip_stages is None:
            self.skip_stages = []
        if self.cache_stages is None:
            self.cache_stages = []
            
        # Validate interpolation mode
        valid_modes = ['bilinear', 'nearest', 'bicubic']
        if self.interpolation_mode not in valid_modes:
            raise ValueError(f"interpolation_mode must be one of {valid_modes}")
    
    @classmethod
    def from_preset(cls, preset_name: str) -> 'HARTCacheConfig':
        """Load configuration from preset name"""
        presets = get_cache_presets()
        if preset_name not in presets:
            available = list(presets.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
        return cls(**presets[preset_name])
    
    @classmethod
    def from_json(cls, json_path: str) -> 'HARTCacheConfig':
        """Load configuration from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, json_path: str):
        """Save configuration to JSON file"""
        config_dict = {
            'skip_stages': self.skip_stages,
            'cache_stages': self.cache_stages,
            'enable_attn_cache': self.enable_attn_cache,
            'enable_mlp_cache': self.enable_mlp_cache,
            'threshold': self.threshold,
            'max_skip_stages': self.max_skip_stages,
            'adaptive_threshold': self.adaptive_threshold,
            'interpolation_mode': self.interpolation_mode,
            'calibration_samples': self.calibration_samples,
            'similarity_window': self.similarity_window
        }
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def is_cache_enabled(self) -> bool:
        """Check if any caching is enabled"""
        return bool(self.skip_stages or self.cache_stages)
    
    def validate(self):
        """Validate configuration parameters"""
        # Valid stage numbers for HART (matching VAR stages)
        valid_stages = {1, 4, 9, 16, 25, 36, 64, 100, 169, 256}
        
        for stage in self.skip_stages:
            if stage not in valid_stages:
                raise ValueError(f"Invalid skip stage {stage}. Valid stages: {valid_stages}")
        
        for stage in self.cache_stages:
            if stage not in valid_stages:
                raise ValueError(f"Invalid cache stage {stage}. Valid stages: {valid_stages}")
        
        if self.threshold < 0 or self.threshold > 1:
            raise ValueError("threshold must be between 0 and 1")
        
        if self.max_skip_stages < 0 or self.max_skip_stages > 9:
            raise ValueError("max_skip_stages must be between 0 and 9")


def get_cache_presets() -> Dict[str, Dict[str, Any]]:
    """Get predefined cache configuration presets"""
    return {
        "no-cache": {
            "skip_stages": [],
            "cache_stages": [],
            "enable_attn_cache": False,
            "enable_mlp_cache": False,
            "threshold": 0.7
        },
        
        "conservative-cache": {
            "skip_stages": [256],
            "cache_stages": [169],
            "enable_attn_cache": True,
            "enable_mlp_cache": True,
            "threshold": 0.8
        },
        
        "original-cache": {
            "skip_stages": [169, 256],
            "cache_stages": [100, 169],
            "enable_attn_cache": True,
            "enable_mlp_cache": True,
            "threshold": 0.7
        },
        
        "aggressive-cache": {
            "skip_stages": [64, 100, 169, 256],
            "cache_stages": [36, 64, 100, 169],
            "enable_attn_cache": True,
            "enable_mlp_cache": True,
            "threshold": 0.6
        },
        
        "ultra-fast": {
            "skip_stages": [36, 64, 100, 169, 256],
            "cache_stages": [25, 36, 64, 100, 169],
            "enable_attn_cache": True,
            "enable_mlp_cache": True,
            "threshold": 0.5,
            "adaptive_threshold": True
        },
        
        "attn-only-cache": {
            "skip_stages": [169, 256],
            "cache_stages": [100, 169],
            "enable_attn_cache": True,
            "enable_mlp_cache": False,
            "threshold": 0.7
        },
        
        "mlp-only-cache": {
            "skip_stages": [169, 256],
            "cache_stages": [100, 169],
            "enable_attn_cache": False,
            "enable_mlp_cache": True,
            "threshold": 0.7
        },
        
        "high-quality": {
            "skip_stages": [256],
            "cache_stages": [169],
            "enable_attn_cache": True,
            "enable_mlp_cache": True,
            "threshold": 0.9,
            "interpolation_mode": "bicubic"
        },
        
        "memory-efficient": {
            "skip_stages": [169, 256],
            "cache_stages": [169],
            "enable_attn_cache": True,
            "enable_mlp_cache": False,
            "threshold": 0.8
        }
    }


class CacheStatistics:
    """Track and analyze cache usage statistics"""
    
    def __init__(self):
        self.attn_similarities = []
        self.mlp_similarities = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.effective_thresholds = []
    
    def reset(self):
        """Reset all statistics"""
        self.attn_similarities.clear()
        self.mlp_similarities.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.effective_thresholds.clear()
    
    def record_similarity(self, similarity: float, layer_type: str):
        """Record similarity value for a layer"""
        if layer_type == 'attn':
            self.attn_similarities.append(similarity)
        elif layer_type == 'mlp':
            self.mlp_similarities.append(similarity)
    
    def record_cache_usage(self, hit: bool):
        """Record cache hit or miss"""
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    def get_similarity_stats(self) -> Dict[str, Dict[str, float]]:
        """Get similarity statistics"""
        import numpy as np
        
        stats = {}
        for layer_type, similarities in [('attn', self.attn_similarities), 
                                       ('mlp', self.mlp_similarities)]:
            if similarities:
                stats[layer_type] = {
                    'mean': np.mean(similarities),
                    'std': np.std(similarities),
                    'min': np.min(similarities),
                    'max': np.max(similarities),
                    'median': np.median(similarities),
                    'p80': np.percentile(similarities, 80),
                    'p90': np.percentile(similarities, 90)
                }
            else:
                stats[layer_type] = {}
        
        return stats


def create_enhanced_hart_cache_config(
    performance_level: str = "balanced",
    memory_constrained: bool = False,
    quality_priority: bool = False
) -> HARTCacheConfig:
    """
    Create cache configuration based on requirements
    
    Args:
        performance_level: "conservative", "balanced", "aggressive", "ultra"
        memory_constrained: If True, reduce memory usage
        quality_priority: If True, prioritize quality over speed
    """
    
    if quality_priority:
        base_preset = "high-quality"
    elif memory_constrained:
        base_preset = "memory-efficient"
    elif performance_level == "conservative":
        base_preset = "conservative-cache"
    elif performance_level == "balanced":
        base_preset = "original-cache"
    elif performance_level == "aggressive":
        base_preset = "aggressive-cache"
    elif performance_level == "ultra":
        base_preset = "ultra-fast"
    else:
        raise ValueError(f"Unknown performance_level: {performance_level}")
    
    return HARTCacheConfig.from_preset(base_preset)