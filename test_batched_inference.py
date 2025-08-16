#!/usr/bin/env python3
"""
Test script for batched inference functionality
"""

import argparse
import os
import tempfile
import json

def create_test_prompts():
    """Create test prompts file"""
    test_prompts = [
        "A beautiful sunset over the ocean",
        "A cat sleeping on a comfortable chair",
        "A futuristic city with flying cars",
        "A peaceful garden with colorful flowers"
    ]
    
    fd, temp_path = tempfile.mkstemp(suffix='.txt')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write('\n'.join(test_prompts))
        return temp_path
    except:
        os.close(fd)
        os.unlink(temp_path)
        raise

def test_sample_script(args):
    """Test the updated sample.py script with batching"""
    print("Testing sample.py with batched inference...")
    
    # Create test prompts
    prompts_file = create_test_prompts()
    output_dir = os.path.join(args.output_dir, "sample_test")
    
    try:
        # Test with different batch sizes
        batch_sizes = [1, 2, 4] if args.comprehensive else [2]
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            batch_output_dir = os.path.join(output_dir, f"batch_{batch_size}")
            
            from sample import main as sample_main
            
            # Create args for sample script
            class SampleArgs:
                def __init__(self):
                    self.model_path = args.model_path
                    self.text_model_path = args.text_model_path
                    self.shield_model_path = args.shield_model_path
                    self.prompt = None
                    self.prompt_list = None
                    self.seed = 1
                    self.use_ema = True
                    self.max_token_length = 300
                    self.use_llm_system_prompt = True
                    self.cfg = 4.5
                    self.more_smooth = True
                    self.sample_folder_dir = batch_output_dir
                    self.store_seperately = True
                    self.batch_size = batch_size
                    self.use_cache = False
                    self.calibration = False
                    self.threshold = 0.7
                    self.sim_path = None
                    
                    # Read prompts from file
                    with open(prompts_file, 'r') as f:
                        self.prompt_list = [line.strip() for line in f.readlines()]
            
            sample_args = SampleArgs()
            
            try:
                sample_main(sample_args)
                print(f"✓ Batch size {batch_size} test passed")
            except Exception as e:
                print(f"✗ Batch size {batch_size} test failed: {e}")
    
    finally:
        os.unlink(prompts_file)

def test_evaluation_script(args):
    """Test the updated evaluation script with integrated generation"""
    print("Testing evaluate_hart.py with integrated batched generation...")
    
    # Create test prompts
    prompts_file = create_test_prompts()
    output_dir = os.path.join(args.output_dir, "eval_test")
    
    try:
        from evaluation.evaluate_hart import HARTEvaluator
        
        # Create args for evaluation script
        class EvalArgs:
            def __init__(self):
                self.model_path = args.model_path
                self.text_model_path = args.text_model_path
                self.shield_model_path = args.shield_model_path
                self.output_dir = output_dir
                self.prompts_file = prompts_file
                self.device = 'cuda'
                self.generate_samples = True
                self.generation_batch_size = 2
                self.overwrite = True
                self.use_ema = True
                self.max_token_length = 300
                self.use_llm_system_prompt = True
                self.cfg = 4.5
                self.more_smooth = True
                self.seed = 1
                self.max_samples = 4
                
                # Disable other evaluations for this test
                self.eval_fid = False
                self.eval_clip_score = False
                self.eval_geneval = False
                self.eval_quality_metrics = False
                self.eval_speed = False
                self.exp_name = 'test_batch_generation'
        
        eval_args = EvalArgs()
        evaluator = HARTEvaluator(eval_args)
        
        try:
            evaluator.generate_samples()
            print("✓ Evaluation script batched generation test passed")
            
            # Check if images were generated
            if os.path.exists(output_dir):
                generated_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
                print(f"✓ Generated {len(generated_files)} images")
            else:
                print("✗ No output directory created")
                
        except Exception as e:
            print(f"✗ Evaluation script test failed: {e}")
    
    finally:
        os.unlink(prompts_file)

def main():
    parser = argparse.ArgumentParser(description='Test batched inference functionality')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to HART model')
    parser.add_argument('--text_model_path', type=str, required=True,
                        help='Path to text model')
    parser.add_argument('--shield_model_path', type=str,
                        help='Path to shield model')
    parser.add_argument('--output_dir', type=str, default='./test_output',
                        help='Directory for test outputs')
    parser.add_argument('--test_sample', action='store_true',
                        help='Test sample.py script')
    parser.add_argument('--test_eval', action='store_true',
                        help='Test evaluation script')
    parser.add_argument('--comprehensive', action='store_true',
                        help='Run comprehensive tests with multiple batch sizes')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Starting batched inference tests...")
    print(f"Model path: {args.model_path}")
    print(f"Text model path: {args.text_model_path}")
    print(f"Output directory: {args.output_dir}")
    
    if args.test_sample or (not args.test_sample and not args.test_eval):
        test_sample_script(args)
    
    if args.test_eval or (not args.test_sample and not args.test_eval):
        test_evaluation_script(args)
    
    print("\nBatched inference tests completed!")

if __name__ == '__main__':
    main()