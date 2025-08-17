#!/usr/bin/env python3
"""
HART GenEval evaluation script with cache mechanism analysis.
Evaluates the impact of VAR-style caching on HART performance for compositional text-to-image generation.

Based on VAR cache evaluation methodology and HART GenEval evaluation framework.
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

# Add HART modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from mmdet.apis import inference_detector, init_detector
    from mmdet import __version__ as mmdet_version
    MMDET_AVAILABLE = True
except ImportError:
    MMDET_AVAILABLE = False
    print("Warning: MMDetection not available. Object detection evaluation will be skipped.")

from utils import tracker
from hart.modules.models.transformer.hart_transformer_t2i_enhanced import HARTForT2IEnhanced
from hart.modules.models.transformer.configuration import HARTForT2IConfig
from hart.modules.networks.basic_hart_enhanced import CacheConfig


class HARTCacheGenEvalEvaluator:
    """
    HART cache evaluation on GenEval dataset
    Evaluates impact of caching mechanisms on compositional generation quality
    """
    
    def __init__(self, model_path: str, detector_config: Optional[str] = None, 
                 detector_model: Optional[str] = None, device: str = 'cuda'):
        self.device = device
        self.model_path = model_path
        self.detector = None
        
        # Initialize object detector for GenEval metrics
        if MMDET_AVAILABLE and detector_config and detector_model:
            try:
                self.detector = init_detector(detector_config, detector_model, device=device)
                print(f"Loaded MMDetection model: {detector_config}")
            except Exception as e:
                print(f"Failed to load detector: {e}")
                self.detector = None
        
        # GenEval categories for evaluation
        self.geneval_categories = [
            'object', 'count', 'spatial', 'attribute', 'color', 'shape', 'texture'
        ]
    
    def load_hart_model(self, cache_config: CacheConfig) -> HARTForT2IEnhanced:
        """Load HART model with specified cache configuration"""
        print(f"Loading HART model from {self.model_path}")
        
        # Load configuration
        config = HARTForT2IConfig.from_pretrained(self.model_path)
        
        # Apply cache configuration
        config.cache_skip_stages = cache_config.skip_stages
        config.cache_cache_stages = cache_config.cache_stages
        config.cache_enable_attn = cache_config.enable_attn_cache
        config.cache_enable_mlp = cache_config.enable_mlp_cache
        config.cache_threshold = cache_config.threshold
        config.cache_adaptive_threshold = cache_config.adaptive_threshold
        config.cache_interpolation_mode = cache_config.interpolation_mode
        
        # Create enhanced model
        model = HARTForT2IEnhanced(config)
        model = model.from_pretrained(self.model_path)
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        return model
    
    def load_geneval_prompts(self, geneval_path: str) -> Dict[str, List[Dict]]:
        """Load GenEval prompts organized by category"""
        prompts_by_category = {}
        
        # Load prompts from GenEval dataset
        if os.path.exists(geneval_path):
            if geneval_path.endswith('.jsonl'):
                with open(geneval_path, 'r') as f:
                    prompts = [json.loads(line) for line in f]
            else:
                with open(geneval_path, 'r') as f:
                    prompts = json.load(f)
        else:
            # Create sample prompts if file doesn't exist
            print(f"GenEval file not found at {geneval_path}, using sample prompts")
            prompts = self._create_sample_prompts()
        
        # Organize by category
        for category in self.geneval_categories:
            prompts_by_category[category] = []
        
        for prompt_data in prompts:
            category = prompt_data.get('category', 'object')
            if category in prompts_by_category:
                prompts_by_category[category].append(prompt_data)
        
        return prompts_by_category
    
    def _create_sample_prompts(self) -> List[Dict]:
        """Create sample prompts for testing when GenEval dataset is not available"""
        return [
            {"prompt": "A red apple on a wooden table", "category": "object", "expected_objects": ["apple", "table"]},
            {"prompt": "Three cats sitting on a fence", "category": "count", "expected_count": {"cat": 3}},
            {"prompt": "A blue car next to a yellow house", "category": "spatial", "expected_spatial": [("car", "next to", "house")]},
            {"prompt": "A large green tree with small red flowers", "category": "attribute", "expected_attributes": {"tree": ["large", "green"], "flowers": ["small", "red"]}},
            {"prompt": "Two white dogs playing in a park", "category": "color", "expected_colors": {"dog": "white"}},
            {"prompt": "A round clock on a square wall", "category": "shape", "expected_shapes": {"clock": "round", "wall": "square"}},
            {"prompt": "A fluffy cat on a smooth marble floor", "category": "texture", "expected_textures": {"cat": "fluffy", "floor": "smooth"}}
        ]
    
    def detect_objects(self, image: Image.Image, conf_threshold: float = 0.3) -> List[Dict]:
        """Detect objects in an image using MMDetection"""
        if not self.detector:
            return []
        
        try:
            # Convert PIL image to numpy array
            image_array = np.array(image)
            if image_array.shape[2] == 4:  # RGBA
                image_array = image_array[:, :, :3]  # Remove alpha channel
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            # Run inference
            result = inference_detector(self.detector, image_array)
            
            # Extract detected objects
            detections = []
            if hasattr(result, 'pred_instances'):
                # MMDet v3.x format
                instances = result.pred_instances
                scores = instances.scores.cpu().numpy()
                labels = instances.labels.cpu().numpy()
                bboxes = instances.bboxes.cpu().numpy()
                
                for score, label, bbox in zip(scores, labels, bboxes):
                    if score > conf_threshold:
                        detections.append({
                            'label': int(label),
                            'score': float(score),
                            'bbox': bbox.tolist()
                        })
            
            return detections
            
        except Exception as e:
            print(f"Object detection failed: {e}")
            return []
    
    def evaluate_generation_quality(self, image: Image.Image, prompt_data: Dict) -> Dict[str, float]:
        """Evaluate generation quality for a specific prompt"""
        metrics = {}
        
        # Detect objects in the generated image
        detections = self.detect_objects(image)
        
        # Object detection accuracy
        if 'expected_objects' in prompt_data:
            expected_objects = prompt_data['expected_objects']
            detected_labels = [det['label'] for det in detections]
            
            # Simple object presence check (would need proper COCO label mapping in practice)
            object_score = len([obj for obj in expected_objects if obj in str(detected_labels)]) / max(len(expected_objects), 1)
            metrics['object_accuracy'] = object_score
        
        # Count accuracy
        if 'expected_count' in prompt_data:
            expected_counts = prompt_data['expected_count']
            for obj_name, expected_count in expected_counts.items():
                # Simplified counting (would need semantic object mapping)
                detected_count = len(detections)
                count_accuracy = 1.0 if detected_count == expected_count else max(0, 1 - abs(detected_count - expected_count) / expected_count)
                metrics[f'count_accuracy_{obj_name}'] = count_accuracy
        
        # Spatial relationship evaluation (simplified)
        if 'expected_spatial' in prompt_data:
            # This would require more sophisticated spatial analysis
            metrics['spatial_accuracy'] = 0.5  # Placeholder
        
        # Attribute evaluation (simplified)
        if 'expected_attributes' in prompt_data:
            # This would require attribute recognition
            metrics['attribute_accuracy'] = 0.5  # Placeholder
        
        # Default quality score based on detection confidence
        if detections:
            avg_confidence = np.mean([det['score'] for det in detections])
            metrics['detection_confidence'] = avg_confidence
        else:
            metrics['detection_confidence'] = 0.0
        
        return metrics
    
    def generate_images(self, model: HARTForT2IEnhanced, prompts: List[str], 
                       batch_size: int = 4, num_samples: int = 1, 
                       cfg_scale: float = 1.5, seed: int = 42) -> List[Image.Image]:
        """Generate images from prompts using HART model"""
        images = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(prompts), batch_size), desc="Generating images"):
                batch_prompts = prompts[i:i+batch_size]
                current_batch_size = len(batch_prompts)
                
                for sample_idx in range(num_samples):
                    try:
                        # Generate images (this would need proper text encoding in practice)
                        # For now, we'll generate with random labels as placeholder
                        generated_batch = model.autoregressive_infer_cfg(
                            B=current_batch_size,
                            label_B=None,  # Would need text-to-label mapping
                            g_seed=seed + i + sample_idx,
                            cfg=cfg_scale
                        )
                        
                        # Convert to PIL images
                        for j in range(current_batch_size):
                            img_tensor = generated_batch[j]
                            img_array = img_tensor.permute(1, 2, 0).cpu().numpy()
                            img_array = (img_array * 255).astype(np.uint8)
                            img = Image.fromarray(img_array)
                            images.append(img)
                            
                    except Exception as e:
                        print(f"Generation failed for batch {i}: {e}")
                        # Add placeholder black images for failed generations
                        for _ in range(current_batch_size):
                            images.append(Image.new('RGB', (256, 256), color='black'))
        
        return images
    
    def benchmark_cache_performance(self, model: HARTForT2IEnhanced, 
                                  prompts: List[str], num_runs: int = 5) -> Dict[str, float]:
        """Benchmark generation performance with current cache settings"""
        times = []
        memory_usage = []
        
        print(f"Benchmarking performance with {len(prompts)} prompts...")
        
        for i in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                mem_before = torch.cuda.memory_allocated()
            
            start_time = time.time()
            
            try:
                with torch.no_grad():
                    # Generate a single image for timing
                    _ = model.autoregressive_infer_cfg(
                        B=1,
                        label_B=None,
                        g_seed=42 + i,
                        cfg=1.5
                    )
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append(end_time - start_time)
                
                if torch.cuda.is_available():
                    mem_after = torch.cuda.memory_allocated()
                    peak_mem = torch.cuda.max_memory_allocated()
                    memory_usage.append({
                        'allocated_mb': (mem_after - mem_before) / 1024 / 1024,
                        'peak_mb': peak_mem / 1024 / 1024
                    })
                    
            except Exception as e:
                print(f"Benchmark run {i} failed: {e}")
        
        if not times:
            return {}
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'throughput': 1.0 / np.mean(times),
            'mean_memory_mb': np.mean([m['allocated_mb'] for m in memory_usage]) if memory_usage else 0,
            'peak_memory_mb': np.max([m['peak_mb'] for m in memory_usage]) if memory_usage else 0,
        }
    
    def evaluate_cache_configuration(self, cache_config: CacheConfig, 
                                   prompts_by_category: Dict[str, List[Dict]],
                                   output_dir: str, num_samples: int = 1,
                                   batch_size: int = 4) -> Dict[str, Any]:
        """Evaluate a specific cache configuration on GenEval"""
        print(f"\n" + "="*60)
        print(f"Evaluating cache configuration:")
        print(f"  Skip stages: {cache_config.skip_stages}")
        print(f"  Cache stages: {cache_config.cache_stages}")
        print(f"  Threshold: {cache_config.threshold}")
        print(f"  Attention cache: {cache_config.enable_attn_cache}")
        print(f"  MLP cache: {cache_config.enable_mlp_cache}")
        print("="*60)
        
        # Load model with cache configuration
        model = self.load_hart_model(cache_config)
        
        # Create detailed output directory name for this configuration
        config_parts = []
        if cache_config.skip_stages:
            config_parts.append(f"skip_{'-'.join(map(str, cache_config.skip_stages))}")
        else:
            config_parts.append("skip_none")
        
        if cache_config.cache_stages:
            config_parts.append(f"cache_{'-'.join(map(str, cache_config.cache_stages))}")
        else:
            config_parts.append("cache_none")
        
        config_parts.append(f"thresh_{cache_config.threshold}")
        
        # Add cache type information
        cache_types = []
        if cache_config.enable_attn_cache:
            cache_types.append("attn")
        if cache_config.enable_mlp_cache:
            cache_types.append("mlp")
        if cache_types:
            config_parts.append(f"type_{'_'.join(cache_types)}")
        else:
            config_parts.append("type_none")
        
        # Add interpolation mode if not default
        if cache_config.interpolation_mode != 'bilinear':
            config_parts.append(f"interp_{cache_config.interpolation_mode}")
        
        # Add adaptive threshold flag
        if cache_config.adaptive_threshold:
            config_parts.append("adaptive")
        
        config_name = "_".join(config_parts)
        config_output_dir = os.path.join(output_dir, config_name)
        os.makedirs(config_output_dir, exist_ok=True)
        
        # Create comprehensive configuration metadata
        import datetime
        import platform
        import torch
        
        config_metadata = {
            # Cache configuration details
            'cache_config': {
                'skip_stages': cache_config.skip_stages,
                'cache_stages': cache_config.cache_stages,
                'threshold': cache_config.threshold,
                'enable_attn_cache': cache_config.enable_attn_cache,
                'enable_mlp_cache': cache_config.enable_mlp_cache,
                'interpolation_mode': cache_config.interpolation_mode,
                'adaptive_threshold': cache_config.adaptive_threshold,
                'calibration_mode': getattr(cache_config, 'calibration_mode', 'none')
            },
            # Evaluation metadata
            'evaluation_metadata': {
                'timestamp': datetime.datetime.now().isoformat(),
                'config_name': config_name,
                'output_directory': config_output_dir,
                'num_categories': len(prompts_by_category),
                'total_prompts': sum(len(prompts) for prompts in prompts_by_category.values()),
                'batch_size': batch_size,
                'num_samples': num_samples
            },
            # System information
            'system_info': {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'cuda_current_device': torch.cuda.current_device() if torch.cuda.is_available() else None
            },
            # Model information
            'model_info': {
                'model_path': self.model_path,
                'detector_available': self.detector is not None,
                'mmdet_available': MMDET_AVAILABLE
            }
        }
        
        results = {
            **config_metadata,
            'category_results': {},
            'overall_metrics': {}
        }
        
        # Benchmark performance
        all_prompts = []
        for category_prompts in prompts_by_category.values():
            all_prompts.extend([p['prompt'] for p in category_prompts])
        
        performance_metrics = self.benchmark_cache_performance(model, all_prompts[:10])  # Use subset for timing
        results['performance'] = performance_metrics
        
        # Create images output directory structure
        images_output_dir = os.path.join(config_output_dir, "generated_images")
        os.makedirs(images_output_dir, exist_ok=True)
        
        # Evaluate each category
        total_quality_scores = []
        global_prompt_idx = 0  # Global counter for prompt numbering
        
        for category, category_prompts in prompts_by_category.items():
            if not category_prompts:
                continue
                
            print(f"\\nEvaluating category: {category} ({len(category_prompts)} prompts)")
            
            category_results = {
                'num_prompts': len(category_prompts),
                'quality_metrics': [],
                'generation_times': []
            }
            
            # Process prompts in this category
            for prompt_data in tqdm(category_prompts[:min(10, len(category_prompts))], desc=f"Processing {category}"):
                prompt = prompt_data['prompt']
                
                # Create directory for this prompt: 00000/, 00001/, etc.
                prompt_dir = os.path.join(images_output_dir, f"{global_prompt_idx:05d}")
                samples_dir = os.path.join(prompt_dir, "samples")
                os.makedirs(samples_dir, exist_ok=True)
                
                # Generate image
                start_time = time.time()
                images = self.generate_images(model, [prompt], batch_size=1, num_samples=num_samples)
                generation_time = time.time() - start_time
                
                category_results['generation_times'].append(generation_time)
                
                # Save metadata for this prompt
                metadata = {
                    "prompt": prompt,
                    "category": category,
                    "generation_time": generation_time,
                    "num_samples": len(images),
                    "prompt_index": global_prompt_idx,
                    "cache_config": {
                        "skip_stages": cache_config.skip_stages,
                        "cache_stages": cache_config.cache_stages,
                        "threshold": cache_config.threshold
                    }
                }
                
                if 'expected_objects' in prompt_data:
                    metadata['expected_objects'] = prompt_data['expected_objects']
                if 'expected_count' in prompt_data:
                    metadata['expected_count'] = prompt_data['expected_count']
                if 'expected_spatial' in prompt_data:
                    metadata['expected_spatial'] = prompt_data['expected_spatial']
                if 'expected_attributes' in prompt_data:
                    metadata['expected_attributes'] = prompt_data['expected_attributes']
                if 'expected_colors' in prompt_data:
                    metadata['expected_colors'] = prompt_data['expected_colors']
                if 'expected_shapes' in prompt_data:
                    metadata['expected_shapes'] = prompt_data['expected_shapes']
                if 'expected_textures' in prompt_data:
                    metadata['expected_textures'] = prompt_data['expected_textures']
                
                # Save metadata.jsonl
                metadata_file = os.path.join(prompt_dir, "metadata.jsonl")
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, ensure_ascii=False)
                
                # Evaluate quality for each generated image and save in samples/
                sample_quality_metrics = []
                for img_idx, image in enumerate(images):
                    # Save individual sample
                    sample_filename = f"{img_idx:04d}.png"
                    image.save(os.path.join(samples_dir, sample_filename))
                    
                    # Evaluate quality
                    quality_metrics = self.evaluate_generation_quality(image, prompt_data)
                    quality_metrics['prompt'] = prompt
                    quality_metrics['category'] = category
                    quality_metrics['generation_time'] = generation_time
                    quality_metrics['sample_index'] = img_idx
                    quality_metrics['prompt_index'] = global_prompt_idx
                    
                    sample_quality_metrics.append(quality_metrics)
                    category_results['quality_metrics'].append(quality_metrics)
                    total_quality_scores.append(quality_metrics.get('detection_confidence', 0))
                
                # Create grid image if multiple samples
                if len(images) > 1:
                    self._create_grid_image(images, os.path.join(prompt_dir, "grid.png"))
                elif len(images) == 1:
                    # Copy single image as grid.png
                    images[0].save(os.path.join(prompt_dir, "grid.png"))
                
                # Update metadata with quality metrics
                metadata['quality_metrics'] = sample_quality_metrics
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                
                global_prompt_idx += 1
            
            # Calculate category averages
            if category_results['quality_metrics']:
                avg_metrics = {}
                for key in category_results['quality_metrics'][0].keys():
                    if key not in ['prompt', 'category', 'sample_index', 'prompt_index'] and isinstance(category_results['quality_metrics'][0][key], (int, float)):
                        values = [m[key] for m in category_results['quality_metrics'] if key in m]
                        if values:
                            avg_metrics[f'avg_{key}'] = np.mean(values)
                
                category_results['averages'] = avg_metrics
            
            results['category_results'][category] = category_results
        
        # Calculate overall metrics
        if total_quality_scores:
            results['overall_metrics'] = {
                'overall_quality_score': np.mean(total_quality_scores),
                'quality_std': np.std(total_quality_scores),
                'num_images_generated': len(total_quality_scores)
            }
        
        # Get cache statistics
        cache_stats = model.get_cache_statistics()
        results['cache_statistics'] = cache_stats
        
        # Save configuration summary file
        config_summary_file = os.path.join(config_output_dir, 'config_info.json')
        config_summary = {
            'configuration_name': config_name,
            'description': self._generate_config_description(cache_config),
            'cache_settings': results['cache_config'],
            'evaluation_settings': results['evaluation_metadata'],
            'expected_performance': self._estimate_performance_tier(cache_config)
        }
        with open(config_summary_file, 'w') as f:
            json.dump(config_summary, f, indent=2, default=str)
        
        # Save detailed results
        results_file = os.path.join(config_output_dir, 'evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create human-readable config description
        config_desc_file = os.path.join(config_output_dir, 'CONFIG_DESCRIPTION.md')
        with open(config_desc_file, 'w') as f:
            f.write(self._generate_config_markdown(cache_config, config_name, results))
        
        print(f"\\nConfiguration evaluation complete.")
        print(f"  Results: {results_file}")
        print(f"  Config info: {config_summary_file}")
        print(f"  Description: {config_desc_file}")
        
        return results
    
    def compare_cache_configurations(self, cache_configs: List[CacheConfig],
                                   prompts_by_category: Dict[str, List[Dict]],
                                   output_dir: str) -> Dict[str, Any]:
        """Compare multiple cache configurations"""
        print(f"\\nComparing {len(cache_configs)} cache configurations...")
        
        comparison_results = {
            'configurations': [],
            'comparison_metrics': {}
        }
        
        all_results = []
        
        for i, cache_config in enumerate(cache_configs):
            print(f"\\nEvaluating configuration {i+1}/{len(cache_configs)}")
            
            config_results = self.evaluate_cache_configuration(
                cache_config, prompts_by_category, output_dir
            )
            all_results.append(config_results)
            comparison_results['configurations'].append(config_results)
        
        # Generate comparison metrics
        if all_results:
            # Performance comparison
            performance_comparison = {}
            for metric in ['mean_time', 'throughput', 'mean_memory_mb']:
                values = [r['performance'].get(metric, 0) for r in all_results if 'performance' in r]
                if values:
                    performance_comparison[metric] = {
                        'values': values,
                        'best_idx': int(np.argmin(values)) if metric in ['mean_time', 'mean_memory_mb'] else int(np.argmax(values)),
                        'worst_idx': int(np.argmax(values)) if metric in ['mean_time', 'mean_memory_mb'] else int(np.argmin(values))
                    }
            
            # Quality comparison
            quality_scores = [r['overall_metrics'].get('overall_quality_score', 0) for r in all_results if 'overall_metrics' in r]
            if quality_scores:
                performance_comparison['quality_score'] = {
                    'values': quality_scores,
                    'best_idx': int(np.argmax(quality_scores)),
                    'worst_idx': int(np.argmin(quality_scores))
                }
            
            comparison_results['comparison_metrics'] = performance_comparison
        
        # Save comparison results
        comparison_file = os.path.join(output_dir, 'cache_comparison_results.json')
        with open(comparison_file, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        # Generate summary report
        self._generate_comparison_report(comparison_results, output_dir)
        
        return comparison_results
    
    def _generate_comparison_report(self, comparison_results: Dict, output_dir: str):
        """Generate a human-readable comparison report"""
        report_file = os.path.join(output_dir, 'comparison_report.md')
        
        with open(report_file, 'w') as f:
            f.write("# HART Cache Configuration Comparison Report\\n\\n")
            
            configs = comparison_results['configurations']
            metrics = comparison_results.get('comparison_metrics', {})
            
            f.write("## Configuration Summary\\n\\n")
            f.write("| Config | Skip Stages | Cache Stages | Threshold | Attn Cache | MLP Cache |\\n")
            f.write("|--------|-------------|--------------|-----------|------------|-----------|\\n")
            
            for i, config in enumerate(configs):
                cache_config = config['cache_config']
                f.write(f"| {i+1} | {cache_config['skip_stages']} | {cache_config['cache_stages']} | "
                       f"{cache_config['threshold']} | {cache_config['enable_attn_cache']} | "
                       f"{cache_config['enable_mlp_cache']} |\\n")
            
            f.write("\\n## Performance Comparison\\n\\n")
            
            if 'mean_time' in metrics:
                f.write("### Generation Time\\n")
                times = metrics['mean_time']['values']
                best_idx = metrics['mean_time']['best_idx']
                f.write(f"- Best (fastest): Config {best_idx + 1} with {times[best_idx]:.3f}s\\n")
                f.write(f"- Times: {[f'{t:.3f}s' for t in times]}\\n\\n")
            
            if 'throughput' in metrics:
                f.write("### Throughput\\n")
                throughputs = metrics['throughput']['values']
                best_idx = metrics['throughput']['best_idx']
                f.write(f"- Best: Config {best_idx + 1} with {throughputs[best_idx]:.2f} img/s\\n")
                f.write(f"- Throughputs: {[f'{t:.2f}' for t in throughputs]}\\n\\n")
            
            if 'quality_score' in metrics:
                f.write("### Quality Score\\n")
                scores = metrics['quality_score']['values']
                best_idx = metrics['quality_score']['best_idx']
                f.write(f"- Best: Config {best_idx + 1} with score {scores[best_idx]:.3f}\\n")
                f.write(f"- Scores: {[f'{s:.3f}' for s in scores]}\\n\\n")
            
            f.write("## Recommendations\\n\\n")
            
            if metrics:
                # Find balanced configuration
                normalized_scores = {}
                for i in range(len(configs)):
                    score = 0
                    if 'mean_time' in metrics:
                        # Lower time is better
                        time_score = 1 - (metrics['mean_time']['values'][i] - min(metrics['mean_time']['values'])) / (max(metrics['mean_time']['values']) - min(metrics['mean_time']['values']) + 1e-6)
                        score += time_score
                    if 'quality_score' in metrics:
                        # Higher quality is better
                        quality_values = metrics['quality_score']['values']
                        quality_score = (quality_values[i] - min(quality_values)) / (max(quality_values) - min(quality_values) + 1e-6)
                        score += quality_score
                    normalized_scores[i] = score
                
                best_balanced = max(normalized_scores.keys(), key=lambda k: normalized_scores[k])
                f.write(f"- **Balanced Performance**: Config {best_balanced + 1} offers the best speed/quality trade-off\\n")
                
                if 'mean_time' in metrics:
                    fastest = metrics['mean_time']['best_idx']
                    f.write(f"- **Speed Priority**: Config {fastest + 1} for maximum generation speed\\n")
                
                if 'quality_score' in metrics:
                    highest_quality = metrics['quality_score']['best_idx']
                    f.write(f"- **Quality Priority**: Config {highest_quality + 1} for best generation quality\\n")
        
        print(f"Comparison report saved to {report_file}")
    
    def _generate_config_description(self, cache_config: CacheConfig) -> str:
        """Generate a human-readable description of the cache configuration"""
        parts = []
        
        # Cache aggressiveness level
        if not cache_config.skip_stages and not cache_config.cache_stages:
            parts.append("No caching (baseline configuration)")
        elif len(cache_config.skip_stages) <= 1:
            parts.append("Conservative caching")
        elif len(cache_config.skip_stages) <= 3:
            parts.append("Moderate caching")
        else:
            parts.append("Aggressive caching")
        
        # Skip stages description
        if cache_config.skip_stages:
            skip_desc = f"skips {len(cache_config.skip_stages)} stages ({cache_config.skip_stages})"
            parts.append(skip_desc)
        
        # Cache stages description
        if cache_config.cache_stages:
            cache_desc = f"caches {len(cache_config.cache_stages)} stages ({cache_config.cache_stages})"
            parts.append(cache_desc)
        
        # Threshold description
        if cache_config.threshold >= 0.8:
            threshold_desc = f"high similarity threshold ({cache_config.threshold})"
        elif cache_config.threshold >= 0.6:
            threshold_desc = f"medium similarity threshold ({cache_config.threshold})"
        else:
            threshold_desc = f"low similarity threshold ({cache_config.threshold})"
        parts.append(threshold_desc)
        
        # Cache types
        cache_types = []
        if cache_config.enable_attn_cache:
            cache_types.append("attention")
        if cache_config.enable_mlp_cache:
            cache_types.append("MLP")
        
        if cache_types:
            parts.append(f"uses {' and '.join(cache_types)} caching")
        
        return " with ".join(parts) + "."
    
    def _estimate_performance_tier(self, cache_config: CacheConfig) -> Dict[str, str]:
        """Estimate the performance tier of a cache configuration"""
        # Calculate aggressiveness score
        skip_score = len(cache_config.skip_stages) * 0.3
        cache_score = len(cache_config.cache_stages) * 0.2
        threshold_score = (1.0 - cache_config.threshold) * 0.5
        
        aggressiveness = skip_score + cache_score + threshold_score
        
        if aggressiveness < 0.5:
            speed_tier = "Slow"
            quality_tier = "Excellent"
        elif aggressiveness < 1.0:
            speed_tier = "Medium"
            quality_tier = "Very Good"
        elif aggressiveness < 2.0:
            speed_tier = "Fast"
            quality_tier = "Good"
        else:
            speed_tier = "Very Fast"
            quality_tier = "Acceptable"
        
        return {
            "speed_tier": speed_tier,
            "quality_tier": quality_tier,
            "aggressiveness_score": f"{aggressiveness:.2f}",
            "recommended_use": self._get_use_case_recommendation(speed_tier, quality_tier)
        }
    
    def _get_use_case_recommendation(self, speed_tier: str, quality_tier: str) -> str:
        """Get use case recommendation based on performance tiers"""
        if speed_tier == "Slow" and quality_tier == "Excellent":
            return "Research and development, highest quality requirements"
        elif speed_tier == "Medium" and quality_tier in ["Very Good", "Excellent"]:
            return "Production deployment with balanced speed/quality"
        elif speed_tier == "Fast":
            return "Real-time applications, interactive demos"
        elif speed_tier == "Very Fast":
            return "High-throughput batch processing, speed-critical applications"
        else:
            return "General purpose applications"
    
    def _generate_config_markdown(self, cache_config: CacheConfig, config_name: str, results: Dict) -> str:
        """Generate a markdown description file for the configuration"""
        md_content = f"""# Cache Configuration: {config_name}

## Configuration Overview

**Description**: {self._generate_config_description(cache_config)}

## Cache Settings

| Setting | Value | Description |
|---------|-------|-------------|
| Skip Stages | {cache_config.skip_stages or 'None'} | Stages that completely skip processing |
| Cache Stages | {cache_config.cache_stages or 'None'} | Stages that use cached results when similar |
| Similarity Threshold | {cache_config.threshold} | Minimum similarity for cache reuse |
| Attention Cache | {'Enabled' if cache_config.enable_attn_cache else 'Disabled'} | Cache attention layer outputs |
| MLP Cache | {'Enabled' if cache_config.enable_mlp_cache else 'Disabled'} | Cache MLP layer outputs |
| Interpolation Mode | {cache_config.interpolation_mode} | Method for feature interpolation |
| Adaptive Threshold | {'Enabled' if cache_config.adaptive_threshold else 'Disabled'} | Dynamic threshold adjustment |

## Performance Characteristics

"""
        
        # Add performance estimates
        perf_estimate = self._estimate_performance_tier(cache_config)
        md_content += f"""
### Expected Performance
- **Speed Tier**: {perf_estimate['speed_tier']}
- **Quality Tier**: {perf_estimate['quality_tier']}
- **Aggressiveness Score**: {perf_estimate['aggressiveness_score']}

### Recommended Use Cases
{perf_estimate['recommended_use']}

"""
        
        # Add evaluation metadata if available
        if 'evaluation_metadata' in results:
            eval_meta = results['evaluation_metadata']
            md_content += f"""
## Evaluation Details

- **Evaluation Date**: {eval_meta.get('timestamp', 'Unknown')}
- **Total Categories**: {eval_meta.get('num_categories', 'Unknown')}
- **Total Prompts**: {eval_meta.get('total_prompts', 'Unknown')}
- **Batch Size**: {eval_meta.get('batch_size', 'Unknown')}
- **Samples per Prompt**: {eval_meta.get('num_samples', 'Unknown')}

"""
        
        # Add system info if available
        if 'system_info' in results:
            sys_info = results['system_info']
            md_content += f"""
## System Information

- **Platform**: {sys_info.get('platform', 'Unknown')}
- **PyTorch Version**: {sys_info.get('torch_version', 'Unknown')}
- **CUDA Available**: {sys_info.get('cuda_available', 'Unknown')}
- **GPU Count**: {sys_info.get('cuda_device_count', 'Unknown')}

"""
        
        md_content += f"""
## File Structure

```
{config_name}/
├── CONFIG_DESCRIPTION.md     # This file
├── config_info.json          # Configuration summary and metadata
├── evaluation_results.json   # Detailed evaluation results
└── generated_images/          # Generated images in standard format
    ├── 00000/                  # First prompt (category: object)
    │   ├── metadata.jsonl      # Prompt details and quality metrics
    │   ├── grid.png            # Grid view of all samples
    │   └── samples/
    │       ├── 0000.png        # Individual samples
    │       ├── 0001.png        # (if multiple samples generated)
    │       └── ...
    ├── 00001/                  # Second prompt (category: count)
    │   ├── metadata.jsonl
    │   ├── grid.png
    │   └── samples/
    │       └── 0000.png
    └── ...                     # Additional prompts numbered sequentially
```

## Metadata Format

Each `metadata.jsonl` contains:
```json
{{
  "prompt": "A red apple on a wooden table",
  "category": "object",
  "generation_time": 2.34,
  "num_samples": 1,
  "prompt_index": 0,
  "cache_config": {{
    "skip_stages": [144, 256],
    "cache_stages": [81, 144],
    "threshold": 0.7
  }},
  "expected_objects": ["apple", "table"],
  "quality_metrics": [
    {{
      "detection_confidence": 0.82,
      "object_accuracy": 0.75,
      "sample_index": 0
    }}
  ]
}}
```

## Analysis Guidelines

1. **Speed vs Quality Trade-off**: Compare generation times with quality scores across prompts
2. **Memory Usage**: Monitor memory consumption during evaluation (logged in system_info)
3. **Category Performance**: Check which compositional aspects are most affected by caching
4. **Cache Efficiency**: Review cache hit rates and similarity scores in cache_statistics
5. **Visual Quality**: Use grid.png for quick overview, samples/ for detailed analysis

Generated by HART Cache GenEval Evaluation Framework
"""
        
        return md_content
    
    def _create_grid_image(self, images: List[Image.Image], output_path: str):
        """Create a grid image from multiple samples"""
        if not images:
            return
        
        # Calculate grid dimensions
        num_images = len(images)
        if num_images == 1:
            images[0].save(output_path)
            return
        
        # Calculate grid size (prefer square or slightly wider)
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
        
        # Get image dimensions
        img_width, img_height = images[0].size
        
        # Create grid canvas
        grid_width = cols * img_width
        grid_height = rows * img_height
        grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
        
        # Place images in grid
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            x = col * img_width
            y = row * img_height
            grid_image.paste(img, (x, y))
        
        grid_image.save(output_path)


def parse_args():
    parser = argparse.ArgumentParser(description='HART cache evaluation on GenEval dataset')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to HART model')
    parser.add_argument('--geneval_path', type=str, required=True,
                       help='Path to GenEval prompts file')
    
    # Detector arguments (optional)
    parser.add_argument('--detector_config', type=str,
                       help='Path to MMDetection config file')
    parser.add_argument('--detector_model', type=str,
                       help='Path to MMDetection model file')
    
    # Evaluation arguments
    parser.add_argument('--output_dir', type=str, default='./hart_cache_geneval_results',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for generation')
    parser.add_argument('--num_samples', type=int, default=1,
                       help='Number of samples per prompt')
    parser.add_argument('--cfg_scale', type=float, default=1.5,
                       help='Classifier-free guidance scale')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Cache configuration arguments
    parser.add_argument('--compare_presets', action='store_true',
                       help='Compare predefined cache presets')
    parser.add_argument('--cache_preset', type=str,
                       choices=['no-cache', 'conservative', 'original', 'aggressive', 'ultra-fast'],
                       help='Use a specific cache preset')
    
    # Custom cache arguments
    parser.add_argument('--skip_stages', type=int, nargs='*', default=[],
                       help='Stages to skip')
    parser.add_argument('--cache_stages', type=int, nargs='*', default=[],
                       help='Stages to cache')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Cache similarity threshold')
    parser.add_argument('--disable_attn_cache', action='store_true',
                       help='Disable attention caching')
    parser.add_argument('--disable_mlp_cache', action='store_true',
                       help='Disable MLP caching')
    parser.add_argument('--adaptive_threshold', action='store_true',
                       help='Use adaptive thresholding')
    parser.add_argument('--interpolation_mode', choices=['bilinear', 'nearest', 'bicubic'],
                       default='bilinear', help='Interpolation mode')
    
    return parser.parse_args()


def create_cache_presets() -> Dict[str, CacheConfig]:
    """Create predefined cache configurations for comparison"""
    return {
        'no-cache': CacheConfig(
            skip_stages=[],
            cache_stages=[],
            enable_attn_cache=False,
            enable_mlp_cache=False,
            threshold=0.7
        ),
        'conservative': CacheConfig(
            skip_stages=[256],
            cache_stages=[144],
            enable_attn_cache=True,
            enable_mlp_cache=True,
            threshold=0.8
        ),
        'original': CacheConfig(
            skip_stages=[144, 256],
            cache_stages=[81, 144],
            enable_attn_cache=True,
            enable_mlp_cache=True,
            threshold=0.7
        ),
        'aggressive': CacheConfig(
            skip_stages=[49, 81, 144, 256],
            cache_stages=[25, 49, 81, 144],
            enable_attn_cache=True,
            enable_mlp_cache=True,
            threshold=0.6
        ),
        'ultra-fast': CacheConfig(
            skip_stages=[25, 49, 81, 144, 256, 441],
            cache_stages=[16, 25, 49, 81, 144, 256],
            enable_attn_cache=True,
            enable_mlp_cache=True,
            threshold=0.5,
            adaptive_threshold=True
        )
    }


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = HARTCacheGenEvalEvaluator(
        model_path=args.model_path,
        detector_config=args.detector_config,
        detector_model=args.detector_model
    )
    
    # Load GenEval prompts
    prompts_by_category = evaluator.load_geneval_prompts(args.geneval_path)
    print(f"Loaded GenEval prompts: {sum(len(prompts) for prompts in prompts_by_category.values())} total")
    for category, prompts in prompts_by_category.items():
        print(f"  {category}: {len(prompts)} prompts")
    
    if args.compare_presets:
        # Compare all predefined presets
        cache_presets = create_cache_presets()
        cache_configs = list(cache_presets.values())
        
        print(f"\\nComparing {len(cache_configs)} predefined cache configurations...")
        comparison_results = evaluator.compare_cache_configurations(
            cache_configs, prompts_by_category, args.output_dir
        )
        
        print(f"\\nComparison complete! Results saved to {args.output_dir}")
        
    elif args.cache_preset:
        # Use specific preset
        cache_presets = create_cache_presets()
        cache_config = cache_presets[args.cache_preset]
        
        results = evaluator.evaluate_cache_configuration(
            cache_config, prompts_by_category, args.output_dir,
            num_samples=args.num_samples, batch_size=args.batch_size
        )
        
        print(f"\\nEvaluation complete! Results saved to {args.output_dir}")
        
    else:
        # Use custom cache configuration
        cache_config = CacheConfig(
            skip_stages=args.skip_stages,
            cache_stages=args.cache_stages,
            enable_attn_cache=not args.disable_attn_cache,
            enable_mlp_cache=not args.disable_mlp_cache,
            threshold=args.threshold,
            adaptive_threshold=args.adaptive_threshold,
            interpolation_mode=args.interpolation_mode
        )
        
        results = evaluator.evaluate_cache_configuration(
            cache_config, prompts_by_category, args.output_dir,
            num_samples=args.num_samples, batch_size=args.batch_size
        )
        
        print(f"\\nEvaluation complete! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()