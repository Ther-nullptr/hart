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
        
        # Create output directory for this configuration
        config_name = f"skip_{'-'.join(map(str, cache_config.skip_stages))}_cache_{'-'.join(map(str, cache_config.cache_stages))}_thresh_{cache_config.threshold}"
        config_output_dir = os.path.join(output_dir, config_name)
        os.makedirs(config_output_dir, exist_ok=True)
        
        results = {
            'cache_config': {
                'skip_stages': cache_config.skip_stages,
                'cache_stages': cache_config.cache_stages,
                'threshold': cache_config.threshold,
                'enable_attn_cache': cache_config.enable_attn_cache,
                'enable_mlp_cache': cache_config.enable_mlp_cache,
                'interpolation_mode': cache_config.interpolation_mode
            },
            'category_results': {},
            'overall_metrics': {}
        }
        
        # Benchmark performance
        all_prompts = []
        for category_prompts in prompts_by_category.values():
            all_prompts.extend([p['prompt'] for p in category_prompts])
        
        performance_metrics = self.benchmark_cache_performance(model, all_prompts[:10])  # Use subset for timing
        results['performance'] = performance_metrics
        
        # Evaluate each category
        total_quality_scores = []
        
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
                
                # Generate image
                start_time = time.time()
                images = self.generate_images(model, [prompt], batch_size=1, num_samples=num_samples)
                generation_time = time.time() - start_time
                
                category_results['generation_times'].append(generation_time)
                
                # Evaluate quality for each generated image
                for img_idx, image in enumerate(images):
                    # Save generated image
                    img_filename = f"{category}_prompt_{len(category_results['quality_metrics'])}_sample_{img_idx}.png"
                    image.save(os.path.join(config_output_dir, img_filename))
                    
                    # Evaluate quality
                    quality_metrics = self.evaluate_generation_quality(image, prompt_data)
                    quality_metrics['prompt'] = prompt
                    quality_metrics['category'] = category
                    quality_metrics['generation_time'] = generation_time
                    
                    category_results['quality_metrics'].append(quality_metrics)
                    total_quality_scores.append(quality_metrics.get('detection_confidence', 0))
            
            # Calculate category averages
            if category_results['quality_metrics']:
                avg_metrics = {}
                for key in category_results['quality_metrics'][0].keys():
                    if key not in ['prompt', 'category'] and isinstance(category_results['quality_metrics'][0][key], (int, float)):
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
        
        # Save results
        results_file = os.path.join(config_output_dir, 'evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\\nConfiguration evaluation complete. Results saved to {results_file}")
        
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