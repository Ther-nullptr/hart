"""
GenEval benchmark evaluation for HART model
Adapted from SANA implementation and VAR evaluation methodology
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")

try:
    from mmdet.apis import inference_detector, init_detector
    from mmdet import __version__ as mmdet_version
    MMDET_AVAILABLE = True
except ImportError:
    MMDET_AVAILABLE = False
    print("Warning: MMDetection not available. Object detection evaluation will be skipped.")

from utils import tracker


class GenEvalEvaluator:
    """
    GenEval evaluator for compositional text-to-image generation
    Evaluates object detection, counting, spatial relationships, and attributes
    """
    
    def __init__(self, model_path=None, config_path=None, device='cuda'):
        self.device = device
        self.detector = None
        
        if MMDET_AVAILABLE and model_path and config_path:
            try:
                self.detector = init_detector(config_path, model_path, device=device)
                print(f"Loaded MMDetection model: {config_path}")
            except Exception as e:
                print(f"Failed to load detector: {e}")
                self.detector = None
    
    def detect_objects(self, image_path, conf_threshold=0.3):
        """
        Detect objects in an image using MMDetection
        """
        if not self.detector:
            return []
        
        try:
            # Load image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
            else:
                image = np.array(image_path)
            
            # Run inference
            result = inference_detector(self.detector, image)
            
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
            else:
                # MMDet v2.x format
                if isinstance(result, tuple):
                    bbox_result, segm_result = result
                else:
                    bbox_result = result
                
                for class_id, class_detections in enumerate(bbox_result):
                    for detection in class_detections:
                        if len(detection) >= 5 and detection[4] > conf_threshold:
                            detections.append({
                                'label': class_id,
                                'score': float(detection[4]),
                                'bbox': detection[:4].tolist()
                            })
            
            return detections
        
        except Exception as e:
            print(f"Error detecting objects in {image_path}: {e}")
            return []
    
    def evaluate_object_presence(self, image_path, required_objects, class_names):
        """
        Evaluate if required objects are present in the generated image
        """
        detections = self.detect_objects(image_path)
        
        detected_classes = set()
        for detection in detections:
            class_name = class_names[detection['label']] if detection['label'] < len(class_names) else f"class_{detection['label']}"
            detected_classes.add(class_name.lower())
        
        # Check if required objects are present
        present_objects = []
        missing_objects = []
        
        for obj in required_objects:
            obj_lower = obj.lower()
            if obj_lower in detected_classes:
                present_objects.append(obj)
            else:
                missing_objects.append(obj)
        
        accuracy = len(present_objects) / len(required_objects) if required_objects else 0.0
        
        return {
            'accuracy': accuracy,
            'present_objects': present_objects,
            'missing_objects': missing_objects,
            'total_detected': len(detected_classes),
            'detections': detections
        }
    
    def evaluate_object_counting(self, image_path, object_counts, class_names):
        """
        Evaluate object counting accuracy
        """
        detections = self.detect_objects(image_path)
        
        # Count detected objects by class
        detected_counts = {}
        for detection in detections:
            class_name = class_names[detection['label']] if detection['label'] < len(class_names) else f"class_{detection['label']}"
            class_name = class_name.lower()
            detected_counts[class_name] = detected_counts.get(class_name, 0) + 1
        
        # Evaluate counting accuracy
        count_accuracy = []
        for obj, expected_count in object_counts.items():
            detected_count = detected_counts.get(obj.lower(), 0)
            is_correct = detected_count == expected_count
            count_accuracy.append(is_correct)
        
        overall_accuracy = sum(count_accuracy) / len(count_accuracy) if count_accuracy else 0.0
        
        return {
            'count_accuracy': overall_accuracy,
            'detected_counts': detected_counts,
            'expected_counts': object_counts,
            'correct_counts': sum(count_accuracy)
        }
    
    def evaluate_spatial_relationships(self, image_path, spatial_relations, class_names):
        """
        Evaluate spatial relationships between objects
        """
        detections = self.detect_objects(image_path)
        
        # Group detections by class
        objects_by_class = {}
        for detection in detections:
            class_name = class_names[detection['label']] if detection['label'] < len(class_names) else f"class_{detection['label']}"
            class_name = class_name.lower()
            if class_name not in objects_by_class:
                objects_by_class[class_name] = []
            objects_by_class[class_name].append(detection['bbox'])
        
        # Evaluate spatial relationships
        correct_relations = 0
        total_relations = len(spatial_relations)
        
        for relation in spatial_relations:
            obj1, relationship, obj2 = relation.get('obj1', ''), relation.get('relation', ''), relation.get('obj2', '')
            
            obj1_boxes = objects_by_class.get(obj1.lower(), [])
            obj2_boxes = objects_by_class.get(obj2.lower(), [])
            
            if obj1_boxes and obj2_boxes:
                # Check spatial relationship
                relation_satisfied = self._check_spatial_relationship(
                    obj1_boxes[0], obj2_boxes[0], relationship
                )
                if relation_satisfied:
                    correct_relations += 1
        
        spatial_accuracy = correct_relations / total_relations if total_relations > 0 else 0.0
        
        return {
            'spatial_accuracy': spatial_accuracy,
            'correct_relations': correct_relations,
            'total_relations': total_relations
        }
    
    def _check_spatial_relationship(self, bbox1, bbox2, relationship):
        """
        Check if two bounding boxes satisfy a spatial relationship
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate centers
        center1_x, center1_y = (x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2
        center2_x, center2_y = (x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2
        
        relationship = relationship.lower()
        
        if relationship in ['left of', 'to the left of']:
            return center1_x < center2_x
        elif relationship in ['right of', 'to the right of']:
            return center1_x > center2_x
        elif relationship in ['above', 'on top of']:
            return center1_y < center2_y
        elif relationship in ['below', 'under', 'underneath']:
            return center1_y > center2_y
        elif relationship in ['next to', 'beside']:
            horizontal_distance = abs(center1_x - center2_x)
            vertical_distance = abs(center1_y - center2_y)
            return horizontal_distance < vertical_distance * 2  # Heuristic
        else:
            return False  # Unknown relationship


def load_geneval_prompts(prompt_file):
    """
    Load GenEval prompts with object and spatial information
    """
    with open(prompt_file, 'r') as f:
        data = json.load(f)
    
    prompts = []
    for item in data:
        prompt_info = {
            'id': item.get('id', ''),
            'prompt': item.get('prompt', ''),
            'objects': item.get('objects', []),
            'object_counts': item.get('object_counts', {}),
            'spatial_relations': item.get('spatial_relations', []),
            'attributes': item.get('attributes', {})
        }
        prompts.append(prompt_info)
    
    return prompts


def main():
    parser = argparse.ArgumentParser(description='Evaluate HART model using GenEval benchmark')
    parser.add_argument('--img_path', type=str, required=True,
                        help='Path to directory containing generated images')
    parser.add_argument('--prompt_file', type=str, required=True,
                        help='Path to GenEval prompt JSON file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to MMDetection model weights')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to MMDetection config file')
    parser.add_argument('--class_names_file', type=str, 
                        help='Path to file containing class names')
    parser.add_argument('--conf_threshold', type=float, default=0.3,
                        help='Confidence threshold for object detection')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for computation')
    parser.add_argument('--exp_name', type=str, default='geneval_experiment',
                        help='Experiment name')
    parser.add_argument('--report_to', type=str, default='wandb',
                        help='Where to report results')
    parser.add_argument('--name', type=str, default='hart_geneval',
                        help='Run name for tracking')
    parser.add_argument('--tracker_project_name', type=str, default='hart-evaluation',
                        help='Project name for tracking')
    parser.add_argument('--log_geneval', action='store_true',
                        help='Log GenEval results to tracker')

    args = parser.parse_args()

    if not MMDET_AVAILABLE:
        print("MMDetection is not available. Please install it for object detection evaluation.")
        return

    # Load class names
    if args.class_names_file and os.path.exists(args.class_names_file):
        with open(args.class_names_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        # Default COCO class names
        class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light']
    
    # Initialize evaluator
    evaluator = GenEvalEvaluator(
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device
    )
    
    # Load prompts
    print(f"Loading GenEval prompts from {args.prompt_file}")
    prompts = load_geneval_prompts(args.prompt_file)
    print(f"Loaded {len(prompts)} prompts")
    
    # Evaluate each prompt
    results = {
        'object_presence': [],
        'object_counting': [],
        'spatial_relations': [],
        'overall_accuracy': 0.0
    }
    
    for prompt_info in tqdm(prompts, desc="Evaluating GenEval prompts"):
        prompt_id = prompt_info['id']
        image_path = os.path.join(args.img_path, f"{prompt_id}.png")
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found for prompt {prompt_id}: {image_path}")
            continue
        
        # Evaluate object presence
        if prompt_info['objects']:
            presence_result = evaluator.evaluate_object_presence(
                image_path, prompt_info['objects'], class_names
            )
            results['object_presence'].append(presence_result['accuracy'])
        
        # Evaluate object counting
        if prompt_info['object_counts']:
            counting_result = evaluator.evaluate_object_counting(
                image_path, prompt_info['object_counts'], class_names
            )
            results['object_counting'].append(counting_result['count_accuracy'])
        
        # Evaluate spatial relationships
        if prompt_info['spatial_relations']:
            spatial_result = evaluator.evaluate_spatial_relationships(
                image_path, prompt_info['spatial_relations'], class_names
            )
            results['spatial_relations'].append(spatial_result['spatial_accuracy'])
    
    # Calculate overall metrics
    object_presence_acc = np.mean(results['object_presence']) if results['object_presence'] else 0.0
    object_counting_acc = np.mean(results['object_counting']) if results['object_counting'] else 0.0
    spatial_relations_acc = np.mean(results['spatial_relations']) if results['spatial_relations'] else 0.0
    
    overall_accuracy = np.mean([object_presence_acc, object_counting_acc, spatial_relations_acc])
    results['overall_accuracy'] = overall_accuracy
    
    # Print results
    print("\nGenEval Results:")
    print(f"Object Presence Accuracy: {object_presence_acc:.4f}")
    print(f"Object Counting Accuracy: {object_counting_acc:.4f}")
    print(f"Spatial Relations Accuracy: {spatial_relations_acc:.4f}")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    
    # Save results
    output_file = f"{args.exp_name}_geneval_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Log to tracker
    if args.log_geneval and args.report_to == 'wandb':
        result_dict = {args.exp_name: overall_accuracy}
        tracker(args, result_dict, label="", pattern="epoch_step", metric="GenEval")


if __name__ == '__main__':
    main()