"""
Comprehensive GenEval evaluation for HART model
Implements GenEval's object-focused framework for evaluating text-to-image alignment
with automated prompt generation and complete evaluation pipeline
"""

import argparse
import copy
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, set_seed

# Suppress warnings
warnings.filterwarnings("ignore")

# Add HART modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from hart.modules.models.transformer import HARTForT2I
    from hart.utils import default_prompts, encode_prompts, llm_system_prompt, safety_check
    HART_AVAILABLE = True
except ImportError:
    print("Warning: HART modules not found. Generation features disabled.")
    HART_AVAILABLE = False

# Object detection imports
try:
    import mmdet
    from mmdet.apis import inference_detector, init_detector
    MMDET_AVAILABLE = True
except ImportError:
    MMDET_AVAILABLE = False
    print("Warning: MMDetection not available. Object detection evaluation will be disabled.")

# CLIP imports
try:
    import open_clip
    from clip_benchmark.metrics import zeroshot_classification as zsc
    # Disable tqdm in clip_benchmark to avoid nested progress bars
    zsc.tqdm = lambda it, *args, **kwargs: it
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available. Color classification will be disabled.")

from utils import tracker


class GenEvalDataGenerator:
    """Generates evaluation data following GenEval prompt structure."""
    
    COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
    OBJECTS = [
        "apple", "banana", "cat", "dog", "car", "bicycle", "chair", "table", 
        "book", "bottle", "cup", "phone", "laptop", "bag", "shoes", "hat",
        "ball", "toy", "flower", "tree", "house", "building", "person", "woman",
        "man", "child", "bird", "fish", "boat", "plane"
    ]
    POSITIONS = ["left of", "right of", "above", "below", "in front of", "behind"]
    
    @classmethod
    def create_counting_prompts(cls, max_count: int = 5) -> List[Dict[str, Any]]:
        """Create counting evaluation prompts."""
        prompts = []
        for obj in cls.OBJECTS[:10]:  # Limit for manageable evaluation
            for count in range(1, max_count + 1):
                if count == 1:
                    prompt = f"A photo of a {obj}."
                else:
                    prompt = f"A photo of {count} {obj}s."
                
                metadata = {
                    "tag": "counting",
                    "prompt": prompt,
                    "include": [{"class": obj, "count": count}],
                    "exclude": []
                }
                prompts.append(metadata)
        return prompts
    
    @classmethod
    def create_color_prompts(cls) -> List[Dict[str, Any]]:
        """Create color evaluation prompts."""
        prompts = []
        for obj in cls.OBJECTS[:8]:
            for color in cls.COLORS[:6]:  # Limit colors for manageable evaluation
                prompt = f"A photo of a {color} {obj}."
                metadata = {
                    "tag": "color",
                    "prompt": prompt,
                    "include": [{"class": obj, "count": 1, "color": color}],
                    "exclude": []
                }
                prompts.append(metadata)
        return prompts
    
    @classmethod
    def create_position_prompts(cls) -> List[Dict[str, Any]]:
        """Create spatial position evaluation prompts."""
        prompts = []
        obj_pairs = [
            ("cat", "dog"), ("car", "bicycle"), ("apple", "banana"),
            ("book", "cup"), ("chair", "table")
        ]
        
        for obj1, obj2 in obj_pairs:
            for position in cls.POSITIONS:
                prompt = f"A photo of a {obj1} {position} a {obj2}."
                metadata = {
                    "tag": "position",
                    "prompt": prompt,
                    "include": [
                        {"class": obj1, "count": 1, "position": [position, 1]},
                        {"class": obj2, "count": 1}
                    ],
                    "exclude": []
                }
                prompts.append(metadata)
        return prompts
    
    @classmethod
    def create_cooccurrence_prompts(cls) -> List[Dict[str, Any]]:
        """Create object co-occurrence evaluation prompts."""
        prompts = []
        obj_pairs = [
            ("cat", "dog"), ("apple", "banana"), ("car", "bicycle"),
            ("book", "pen"), ("chair", "table"), ("cup", "bottle")
        ]
        
        for obj1, obj2 in obj_pairs:
            # Both objects present
            prompt = f"A photo of a {obj1} and a {obj2}."
            metadata = {
                "tag": "cooccurrence",
                "prompt": prompt,
                "include": [
                    {"class": obj1, "count": 1},
                    {"class": obj2, "count": 1}
                ],
                "exclude": []
            }
            prompts.append(metadata)
        return prompts
    
    @classmethod
    def generate_all_prompts(cls, output_dir: str) -> str:
        """Generate all evaluation prompts and save to directory structure."""
        
        all_prompts = []
        all_prompts.extend(cls.create_counting_prompts())
        all_prompts.extend(cls.create_color_prompts())
        all_prompts.extend(cls.create_position_prompts())
        all_prompts.extend(cls.create_cooccurrence_prompts())
        
        # Create directory structure
        base_dir = Path(output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Save prompts in GenEval format
        for i, prompt_data in enumerate(all_prompts):
            prompt_dir = base_dir / str(i)
            prompt_dir.mkdir(exist_ok=True)
            
            # Create samples directory
            samples_dir = prompt_dir / "samples"
            samples_dir.mkdir(exist_ok=True)
            
            # Save metadata
            with open(prompt_dir / "metadata.jsonl", 'w') as f:
                json.dump(prompt_data, f)
        
        print(f"Generated {len(all_prompts)} evaluation prompts in {output_dir}")
        return str(base_dir)


class HARTGenEvalEvaluator:
    """Enhanced GenEval evaluator for HART model with complete pipeline."""
    
    def __init__(
        self,
        model_path: str = None,
        text_model_path: str = None,
        device: str = "cuda",
        use_ema: bool = True,
        img_size: int = 1024,
        max_token_length: int = 300
    ):
        self.device = torch.device(device)
        self.img_size = img_size
        self.max_token_length = max_token_length
        
        # Initialize models if paths provided and HART is available
        self.model = None
        self.ema_model = None
        self.text_tokenizer = None
        self.text_model = None
        self.use_ema = use_ema
        
        if HART_AVAILABLE and model_path and text_model_path:
            self._load_hart_models(model_path, text_model_path)
        
        # Detection and classification models (loaded on demand)
        self.object_detector = None
        self.clip_model = None
        self.clip_transform = None
        self.clip_tokenizer = None
        self.classnames = None
        self.color_classifiers = {}
        
    def _load_hart_models(self, model_path: str, text_model_path: str):
        """Load HART and text models."""
        try:
            # Load HART model
            self.model = AutoModel.from_pretrained(model_path).to(self.device)
            self.model.eval()
            
            # Load EMA model if specified
            if self.use_ema:
                self.ema_model = copy.deepcopy(self.model)
                ema_path = os.path.join(model_path, "ema_model.bin")
                if os.path.exists(ema_path):
                    self.ema_model.load_state_dict(torch.load(ema_path))
                else:
                    print(f"Warning: EMA model not found at {ema_path}, using main model")
                    self.use_ema = False
            
            # Load text model
            self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)
            self.text_model = AutoModel.from_pretrained(text_model_path).to(self.device)
            self.text_model.eval()
            
            print("HART models loaded successfully")
            
        except Exception as e:
            print(f"Error loading HART models: {e}")
            self.model = None
    
    def load_detection_models(
        self,
        detector_config: str = None,
        detector_checkpoint: str = None,
        clip_model: str = "ViT-L-14"
    ):
        """Load object detection and CLIP models for evaluation."""
        
        if not MMDET_AVAILABLE:
            print("MMDetection not available, skipping object detection setup")
            return
            
        # Load object detector
        try:
            if detector_config is None:
                detector_config = os.path.join(
                    os.path.dirname(mmdet.__file__),
                    "../configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
                )
            
            if detector_checkpoint is None:
                raise ValueError("detector_checkpoint must be provided")
                
            self.object_detector = init_detector(detector_config, detector_checkpoint, device=self.device)
            print("Object detector loaded successfully")
            
        except Exception as e:
            print(f"Error loading object detector: {e}")
            self.object_detector = None
        
        # Load CLIP model
        if CLIP_AVAILABLE:
            try:
                self.clip_model, _, self.clip_transform = open_clip.create_model_and_transforms(
                    clip_model, pretrained="openai", device=self.device
                )
                self.clip_tokenizer = open_clip.get_tokenizer(clip_model)
                print("CLIP model loaded successfully")
                
            except Exception as e:
                print(f"Error loading CLIP model: {e}")
                self.clip_model = None
        
        # Load COCO class names (standard 80 classes)
        self.classnames = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
            "toothbrush"
        ]
    
    def generate_images_for_geneval(
        self,
        prompts_dir: str,
        cfg: float = 4.5,
        seed: Optional[int] = None,
        more_smooth: bool = True,
        images_per_prompt: int = 4
    ):
        """Generate images for all GenEval prompts."""
        
        if not self.model:
            raise ValueError("HART model not loaded. Cannot generate images.")
        
        if seed is not None:
            set_seed(seed)
        
        prompts_path = Path(prompts_dir)
        if not prompts_path.exists():
            raise ValueError(f"Prompts directory does not exist: {prompts_dir}")
        
        infer_func = (
            self.ema_model.autoregressive_infer_cfg
            if self.use_ema and self.ema_model
            else self.model.autoregressive_infer_cfg
        )
        
        # Process each prompt directory
        for prompt_folder in sorted(prompts_path.iterdir()):
            if not prompt_folder.is_dir() or not prompt_folder.name.isdigit():
                continue
                
            metadata_file = prompt_folder / "metadata.jsonl"
            if not metadata_file.exists():
                continue
                
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            prompt = metadata['prompt']
            samples_dir = prompt_folder / "samples"
            samples_dir.mkdir(exist_ok=True)
            
            try:
                with torch.inference_mode():
                    with torch.autocast("cuda", enabled=True, dtype=torch.float16, cache_enabled=True):
                        # Encode prompt
                        (
                            context_tokens,
                            context_mask,
                            context_position_ids,
                            context_tensor,
                        ) = encode_prompts(
                            [prompt],
                            self.text_model,
                            self.text_tokenizer,
                            self.max_token_length,
                            llm_system_prompt,
                            True,
                        )
                        
                        # Generate multiple images per prompt
                        for img_idx in range(images_per_prompt):
                            output_imgs = infer_func(
                                B=context_tensor.size(0),
                                label_B=context_tensor,
                                cfg=cfg,
                                g_seed=seed + img_idx if seed is not None else None,
                                more_smooth=more_smooth,
                                context_position_ids=context_position_ids,
                                context_mask=context_mask,
                            )
                            
                            # Save image
                            img_path = samples_dir / f"{img_idx}.png"
                            self._save_image(output_imgs[0], str(img_path))
                
                print(f"Generated images for prompt: {prompt[:50]}...")
                
            except Exception as e:
                print(f"Error generating images for prompt '{prompt}': {e}")
                continue
    
    def _save_image(self, img_tensor: torch.Tensor, filepath: str):
        """Save image tensor to file."""
        img_np = img_tensor.mul_(255).cpu().numpy()
        img_np = img_np.transpose(1, 2, 0).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        img_pil.save(filepath)
    
    def evaluate_generated_images(
        self,
        prompts_dir: str,
        output_file: str = "geneval_results.jsonl",
        threshold: float = 0.3,
        counting_threshold: float = 0.9,
        max_objects: int = 16,
        nms_threshold: float = 1.0,
        position_threshold: float = 0.1
    ) -> Dict[str, float]:
        """Evaluate generated images using GenEval methodology."""
        
        if self.object_detector is None:
            raise ValueError("Object detector not loaded. Call load_detection_models() first.")
        
        prompts_path = Path(prompts_dir)
        full_results = []
        
        # Process each prompt directory
        for prompt_folder in tqdm(sorted(prompts_path.iterdir()), desc="Evaluating images"):
            if not prompt_folder.is_dir() or not prompt_folder.name.isdigit():
                continue
                
            metadata_file = prompt_folder / "metadata.jsonl"
            samples_dir = prompt_folder / "samples"
            
            if not metadata_file.exists() or not samples_dir.exists():
                continue
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Evaluate each image
            for img_file in samples_dir.iterdir():
                if not img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    continue
                
                try:
                    result = self._evaluate_single_image(
                        str(img_file), metadata, threshold, counting_threshold,
                        max_objects, nms_threshold, position_threshold
                    )
                    full_results.append(result)
                except Exception as e:
                    print(f"Error evaluating {img_file}: {e}")
                    continue
        
        # Save detailed results
        output_path = Path(prompts_dir).parent / output_file
        with open(output_path, 'w') as f:
            for result in full_results:
                f.write(json.dumps(result) + '\n')
        
        # Compute summary statistics
        summary = self._compute_summary_scores(full_results)
        
        # Save summary
        summary_file = Path(prompts_dir).parent / "geneval_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"GenEval evaluation completed!")
        print(f"Overall accuracy: {summary['overall_accuracy']:.3f}")
        print(f"Results saved to {output_path}")
        print(f"Summary saved to {summary_file}")
        
        return summary
    
    def _evaluate_single_image(
        self,
        image_path: str,
        metadata: Dict[str, Any],
        threshold: float,
        counting_threshold: float,
        max_objects: int,
        nms_threshold: float,
        position_threshold: float
    ) -> Dict[str, Any]:
        """Evaluate a single image using GenEval criteria."""
        
        # Run object detection
        result = inference_detector(self.object_detector, image_path)
        bbox = result[0] if isinstance(result, tuple) else result
        segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        
        image = ImageOps.exif_transpose(Image.open(image_path))
        detected = {}
        
        # Process detections
        conf_threshold = counting_threshold if metadata['tag'] == "counting" else threshold
        
        for index, classname in enumerate(self.classnames):
            # Sort by confidence
            ordering = np.argsort(bbox[index][:, 4])[::-1]
            ordering = ordering[bbox[index][ordering, 4] > conf_threshold]
            ordering = ordering[:max_objects].tolist()
            
            detected[classname] = []
            
            # Apply NMS
            while ordering:
                max_obj = ordering.pop(0)
                detected[classname].append((
                    bbox[index][max_obj],
                    None if segm is None else segm[index][max_obj]
                ))
                
                if nms_threshold < 1:
                    ordering = [
                        obj for obj in ordering
                        if self._compute_iou(bbox[index][max_obj], bbox[index][obj]) < nms_threshold
                    ]
            
            if not detected[classname]:
                del detected[classname]
        
        # Evaluate according to GenEval criteria
        is_correct, reason = self._evaluate_geneval_criteria(
            image, detected, metadata, position_threshold
        )
        
        return {
            'filename': image_path,
            'tag': metadata['tag'],
            'prompt': metadata['prompt'],
            'correct': is_correct,
            'reason': reason,
            'metadata': json.dumps(metadata),
            'details': json.dumps({
                key: [box.tolist() for box, _ in value]
                for key, value in detected.items()
            })
        }
    
    def _evaluate_geneval_criteria(
        self,
        image: Image.Image,
        detected: Dict[str, List],
        metadata: Dict[str, Any],
        position_threshold: float
    ) -> Tuple[bool, str]:
        """Evaluate using GenEval criteria."""
        
        correct = True
        reason = []
        matched_groups = []
        
        # Check for required objects
        for req in metadata.get('include', []):
            classname = req['class']
            matched = True
            found_objects = detected.get(classname, [])[:req['count']]
            
            if len(found_objects) < req['count']:
                correct = matched = False
                reason.append(f"expected {classname}>={req['count']}, found {len(found_objects)}")
            else:
                # Color check
                if 'color' in req and self.clip_model:
                    colors = self._classify_colors(image, found_objects, classname)
                    if colors.count(req['color']) < req['count']:
                        correct = matched = False
                        reason.append(
                            f"expected {req['color']} {classname}>={req['count']}, "
                            f"found {colors.count(req['color'])} {req['color']}"
                        )
                
                # Position check
                if 'position' in req and matched:
                    expected_rel, target_group = req['position']
                    if target_group >= len(matched_groups) or matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(f"no target for {classname} to be {expected_rel}")
                    else:
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                true_rels = self._compute_relative_position(
                                    obj, target_obj, position_threshold
                                )
                                if expected_rel not in true_rels:
                                    correct = matched = False
                                    reason.append(
                                        f"expected {classname} {expected_rel} target, "
                                        f"found {' and '.join(true_rels)} target"
                                    )
                                    break
                            if not matched:
                                break
            
            if matched:
                matched_groups.append(found_objects)
            else:
                matched_groups.append(None)
        
        # Check for excluded objects
        for req in metadata.get('exclude', []):
            classname = req['class']
            if len(detected.get(classname, [])) >= req['count']:
                correct = False
                reason.append(f"expected {classname}<{req['count']}, found {len(detected[classname])}")
        
        return correct, "; ".join(reason)
    
    def _classify_colors(
        self,
        image: Image.Image,
        objects: List[Tuple],
        classname: str
    ) -> List[str]:
        """Classify colors of detected objects using CLIP."""
        
        if not self.clip_model:
            return ["unknown"] * len(objects)
        
        colors = GenEvalDataGenerator.COLORS
        
        if classname not in self.color_classifiers:
            self.color_classifiers[classname] = zsc.zero_shot_classifier(
                self.clip_model,
                self.clip_tokenizer,
                colors,
                [
                    f"a photo of a {{c}} {classname}",
                    f"a photo of a {{c}}-colored {classname}",
                    f"a photo of a {{c}} object"
                ],
                self.device
            )
        
        clf = self.color_classifiers[classname]
        
        # Create crops dataset
        crops_dataset = ImageCropsDataset(image, objects, self.clip_transform)
        dataloader = torch.utils.data.DataLoader(
            crops_dataset, batch_size=16, num_workers=4
        )
        
        with torch.no_grad():
            pred, _ = zsc.run_classification(self.clip_model, clf, dataloader, self.device)
            return [colors[idx.item()] for idx in pred.argmax(1)]
    
    def _compute_iou(self, box_a: np.ndarray, box_b: np.ndarray) -> float:
        """Compute IoU between two bounding boxes."""
        area_fn = lambda box: max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
        
        i_area = area_fn([
            max(box_a[0], box_b[0]), max(box_a[1], box_b[1]),
            min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
        ])
        u_area = area_fn(box_a) + area_fn(box_b) - i_area
        return i_area / u_area if u_area else 0
    
    def _compute_relative_position(
        self,
        obj_a: Tuple,
        obj_b: Tuple,
        position_threshold: float
    ) -> set:
        """Compute relative position between two objects."""
        
        boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
        center_a, center_b = boxes.mean(axis=-2)
        dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
        offset = center_a - center_b
        
        # Apply threshold based on object dimensions
        revised_offset = np.maximum(
            np.abs(offset) - position_threshold * (dim_a + dim_b), 0
        ) * np.sign(offset)
        
        if np.all(np.abs(revised_offset) < 1e-3):
            return set()
        
        dx, dy = revised_offset / np.linalg.norm(offset)
        relations = set()
        
        if dx < -0.5:
            relations.add("left of")
        if dx > 0.5:
            relations.add("right of")
        if dy < -0.5:
            relations.add("above")
        if dy > 0.5:
            relations.add("below")
            
        return relations
    
    def _compute_summary_scores(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute summary scores from evaluation results."""
        
        if not results:
            return {"overall_accuracy": 0.0}
        
        # Overall accuracy
        total_correct = sum(1 for r in results if r['correct'])
        overall_accuracy = total_correct / len(results)
        
        # Per-tag accuracy
        tag_stats = {}
        for result in results:
            tag = result['tag']
            if tag not in tag_stats:
                tag_stats[tag] = {'correct': 0, 'total': 0}
            tag_stats[tag]['total'] += 1
            if result['correct']:
                tag_stats[tag]['correct'] += 1
        
        tag_accuracies = {
            f"{tag}_accuracy": stats['correct'] / stats['total']
            for tag, stats in tag_stats.items()
        }
        
        summary = {
            'overall_accuracy': overall_accuracy,
            'total_samples': len(results),
            'total_correct': total_correct,
            **tag_accuracies,
            'tag_statistics': tag_stats
        }
        
        return summary


class ImageCropsDataset(torch.utils.data.Dataset):
    """Dataset for cropped object regions."""
    
    def __init__(self, image: Image.Image, objects: List[Tuple], transform=None, bgcolor="#999"):
        self.image = image.convert("RGB")
        self.transform = transform
        
        if bgcolor == "original":
            self.blank = self.image.copy()
        else:
            self.blank = Image.new("RGB", image.size, color=bgcolor)
        
        self.objects = objects
    
    def __len__(self):
        return len(self.objects)
    
    def __getitem__(self, index):
        box, mask = self.objects[index]
        
        if mask is not None:
            # Use mask to composite object
            image = Image.composite(
                self.image, self.blank, Image.fromarray(mask)
            )
        else:
            image = self.image
        
        # Crop to bounding box
        image = image.crop(box[:4])
        
        if self.transform is not None:
            image = self.transform(image)
        
        return (image, 0)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive GenEval evaluation for HART model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model paths
    parser.add_argument(
        "--model_path", type=str,
        help="Path to HART model"
    )
    parser.add_argument(
        "--text_model_path", type=str,
        default="Qwen/Qwen2-VL-1.5B-Instruct",
        help="Path to text encoder model"
    )
    
    # Detection model paths
    parser.add_argument(
        "--detector_config", type=str,
        help="Path to object detector config"
    )
    parser.add_argument(
        "--detector_checkpoint", type=str,
        help="Path to object detector checkpoint"
    )
    parser.add_argument(
        "--clip_model", type=str, default="ViT-L-14",
        help="CLIP model architecture"
    )
    
    # Evaluation settings
    parser.add_argument(
        "--output_dir", type=str, default="./geneval_evaluation",
        help="Output directory for evaluation"
    )
    parser.add_argument(
        "--generate_prompts", action="store_true",
        help="Generate GenEval prompts"
    )
    parser.add_argument(
        "--generate_images", action="store_true",
        help="Generate images for evaluation"
    )
    parser.add_argument(
        "--evaluate_only", action="store_true",
        help="Only evaluate existing images"
    )
    parser.add_argument(
        "--prompts_dir", type=str,
        help="Directory containing GenEval prompts (if not generating)"
    )
    
    # Generation settings
    parser.add_argument("--cfg", type=float, default=4.5, help="CFG scale")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA model")
    parser.add_argument("--more_smooth", action="store_true", help="More smooth generation")
    parser.add_argument("--images_per_prompt", type=int, default=4, help="Images per prompt")
    
    # Evaluation thresholds
    parser.add_argument("--threshold", type=float, default=0.3, help="Detection threshold")
    parser.add_argument("--counting_threshold", type=float, default=0.9, help="Counting threshold")
    parser.add_argument("--max_objects", type=int, default=16, help="Max objects per class")
    parser.add_argument("--nms_threshold", type=float, default=1.0, help="NMS threshold")
    parser.add_argument("--position_threshold", type=float, default=0.1, help="Position threshold")
    
    # Technical settings
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--img_size", type=int, default=1024, help="Image resolution")
    parser.add_argument("--max_token_length", type=int, default=300, help="Max token length")
    
    # Experiment tracking
    parser.add_argument("--exp_name", type=str, default="geneval_experiment", help="Experiment name")
    parser.add_argument("--report_to", type=str, default="none", help="Where to report metrics")
    parser.add_argument("--tracker_project_name", type=str, default="hart-evaluation", help="Project name for tracking")
    parser.add_argument("--log_geneval", action="store_true", help="Log GenEval results to tracker")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = HARTGenEvalEvaluator(
        model_path=args.model_path,
        text_model_path=args.text_model_path,
        device=args.device,
        use_ema=args.use_ema,
        img_size=args.img_size,
        max_token_length=args.max_token_length
    )
    
    # Generate prompts if requested
    if args.generate_prompts:
        prompts_dir = GenEvalDataGenerator.generate_all_prompts(
            os.path.join(args.output_dir, "prompts")
        )
    else:
        prompts_dir = args.prompts_dir
        
    if prompts_dir is None:
        raise ValueError("Either --generate_prompts or --prompts_dir must be specified")
    
    # Generate images if requested
    if args.generate_images:
        if not evaluator.model:
            raise ValueError("HART model not loaded. Cannot generate images.")
        
        print("Generating images...")
        evaluator.generate_images_for_geneval(
            prompts_dir=prompts_dir,
            cfg=args.cfg,
            seed=args.seed,
            more_smooth=args.more_smooth,
            images_per_prompt=args.images_per_prompt
        )
    
    # Evaluate images
    if not args.generate_prompts or not args.generate_images or args.evaluate_only:
        if args.detector_checkpoint is None:
            print("Warning: --detector_checkpoint not provided.")
            print("Please download Mask2Former checkpoint for evaluation:")
            print("mkdir -p checkpoints && cd checkpoints")
            print("wget https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220326_224521-11a44721.pth")
            if not args.generate_prompts and not args.generate_images:
                return
            
        if args.detector_checkpoint:
            print("Loading detection models...")
            evaluator.load_detection_models(
                detector_config=args.detector_config,
                detector_checkpoint=args.detector_checkpoint,
                clip_model=args.clip_model
            )
            
            print("Evaluating images...")
            summary = evaluator.evaluate_generated_images(
                prompts_dir=prompts_dir,
                threshold=args.threshold,
                counting_threshold=args.counting_threshold,
                max_objects=args.max_objects,
                nms_threshold=args.nms_threshold,
                position_threshold=args.position_threshold
            )
            
            # Log to tracker if specified
            if args.log_geneval and args.report_to == "wandb":
                result_dict = {args.exp_name: summary['overall_accuracy']}
                tracker(args, result_dict, label="", pattern="epoch_step", metric="GenEval")
            
            print("\nGenEval Results:")
            print(f"Overall Accuracy: {summary['overall_accuracy']:.3f}")
            for key, value in summary.items():
                if key.endswith('_accuracy') and key != 'overall_accuracy':
                    print(f"{key.replace('_accuracy', '').title()} Accuracy: {value:.3f}")


if __name__ == "__main__":
    main()