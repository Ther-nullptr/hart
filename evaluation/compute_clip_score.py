"""
CLIP Score computation for HART model evaluation
Adapted from SANA and VAR evaluation implementations
"""
import argparse
import json
import os
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

from utils import tracker


class CLIPEvaluator:
    def __init__(self, device='cuda', model_name='openai/clip-vit-large-patch14'):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
    def compute_clip_score(self, images, texts, batch_size=32):
        """
        Compute CLIP score between images and texts
        Args:
            images: List of PIL Images or image paths
            texts: List of text prompts
            batch_size: Batch size for processing
        Returns:
            CLIP scores for each image-text pair
        """
        assert len(images) == len(texts), "Number of images and texts must match"
        
        scores = []
        for i in tqdm(range(0, len(images), batch_size), desc="Computing CLIP scores"):
            batch_images = images[i:i+batch_size]
            batch_texts = texts[i:i+batch_size]
            
            # Load images if they are paths
            if isinstance(batch_images[0], (str, Path)):
                batch_images = [Image.open(img).convert('RGB') for img in batch_images]
            
            # Process inputs
            inputs = self.processor(
                text=batch_texts, 
                images=batch_images, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                # CLIP score is the cosine similarity between image and text embeddings
                batch_scores = logits_per_image.diag().cpu().numpy()
                scores.extend(batch_scores)
        
        return np.array(scores)

    def compute_directional_similarity(self, src_images, dst_images, src_texts, dst_texts):
        """
        Compute directional CLIP similarity for image editing evaluation
        """
        # Encode images and texts
        with torch.no_grad():
            src_img_inputs = self.processor(images=src_images, return_tensors="pt").to(self.device)
            dst_img_inputs = self.processor(images=dst_images, return_tensors="pt").to(self.device)
            src_txt_inputs = self.processor(text=src_texts, return_tensors="pt").to(self.device)
            dst_txt_inputs = self.processor(text=dst_texts, return_tensors="pt").to(self.device)
            
            src_img_embeds = self.model.get_image_features(**src_img_inputs)
            dst_img_embeds = self.model.get_image_features(**dst_img_inputs)
            src_txt_embeds = self.model.get_text_features(**src_txt_inputs)
            dst_txt_embeds = self.model.get_text_features(**dst_txt_inputs)
            
            # Normalize embeddings
            src_img_embeds = src_img_embeds / src_img_embeds.norm(dim=-1, keepdim=True)
            dst_img_embeds = dst_img_embeds / dst_img_embeds.norm(dim=-1, keepdim=True)
            src_txt_embeds = src_txt_embeds / src_txt_embeds.norm(dim=-1, keepdim=True)
            dst_txt_embeds = dst_txt_embeds / dst_txt_embeds.norm(dim=-1, keepdim=True)
            
            # Compute directional similarities
            img_direction = dst_img_embeds - src_img_embeds
            txt_direction = dst_txt_embeds - src_txt_embeds
            
            directional_similarity = torch.nn.functional.cosine_similarity(
                img_direction, txt_direction, dim=-1
            )
        
        return directional_similarity.cpu().numpy()


def load_json_data(json_path):
    """Load image paths and corresponding text prompts from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    images = []
    texts = []
    
    if isinstance(data, list):
        for item in data:
            images.append(item.get('image', item.get('img_path', '')))
            texts.append(item.get('text', item.get('prompt', item.get('caption', ''))))
    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                images.append(value.get('image', value.get('img_path', '')))
                texts.append(value.get('text', value.get('prompt', value.get('caption', ''))))
    
    return images, texts


def main():
    parser = argparse.ArgumentParser(description='Compute CLIP Score for generated images')
    parser.add_argument('--real_path', type=str, required=True,
                        help='Path to JSON file containing real images and texts')
    parser.add_argument('--fake_path', type=str, required=True, 
                        help='Path to JSON file containing generated images and texts')
    parser.add_argument('--img_path', type=str, required=True,
                        help='Base path to image directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for computation')
    parser.add_argument('--model_name', type=str, default='openai/clip-vit-large-patch14',
                        help='CLIP model to use')
    parser.add_argument('--sample_nums', type=int, default=30000,
                        help='Number of samples to evaluate')
    parser.add_argument('--exp_name', type=str, default='experiment',
                        help='Experiment name')
    parser.add_argument('--report_to', type=str, default='wandb',
                        help='Where to report results')
    parser.add_argument('--name', type=str, default='hart_evaluation',
                        help='Run name for tracking')
    parser.add_argument('--tracker_project_name', type=str, default='hart-evaluation',
                        help='Project name for tracking')
    parser.add_argument('--log_clip_score', action='store_true',
                        help='Log CLIP score to tracker')

    args = parser.parse_args()

    # Initialize CLIP evaluator
    evaluator = CLIPEvaluator(device=args.device, model_name=args.model_name)
    
    # Load data
    print(f"Loading data from {args.fake_path}")
    images, texts = load_json_data(args.fake_path)
    
    # Convert relative paths to absolute paths
    if args.img_path:
        images = [os.path.join(args.img_path, img) if not os.path.isabs(img) else img 
                 for img in images]
    
    # Limit samples if specified
    if args.sample_nums and args.sample_nums < len(images):
        images = images[:args.sample_nums]
        texts = texts[:args.sample_nums]
    
    print(f"Evaluating {len(images)} image-text pairs")
    
    # Compute CLIP scores
    scores = evaluator.compute_clip_score(images, texts, batch_size=args.batch_size)
    
    # Calculate statistics
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    print(f"CLIP Score Results:")
    print(f"Mean CLIP Score: {mean_score:.4f}")
    print(f"Std CLIP Score: {std_score:.4f}")
    print(f"Min CLIP Score: {np.min(scores):.4f}")
    print(f"Max CLIP Score: {np.max(scores):.4f}")
    
    # Save detailed results
    results = {
        'mean_clip_score': float(mean_score),
        'std_clip_score': float(std_score),
        'min_clip_score': float(np.min(scores)),
        'max_clip_score': float(np.max(scores)),
        'individual_scores': scores.tolist()
    }
    
    output_file = f"{args.exp_name}_clip_scores.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Detailed results saved to {output_file}")
    
    # Log to tracker if specified
    if args.log_clip_score and args.report_to == 'wandb':
        result_dict = {args.exp_name: mean_score}
        tracker(args, result_dict, label="", pattern="epoch_step", metric="CLIP_Score")


if __name__ == '__main__':
    main()