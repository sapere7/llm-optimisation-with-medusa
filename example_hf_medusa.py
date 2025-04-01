#!/usr/bin/env python
"""
Example script showing how to use pre-trained Medusa heads from Hugging Face.
"""
import torch
import time
import argparse
from medusa_model import MedusaModel
from load_medusa_from_hf import get_compatible_medusa_repos

def parse_args():
    parser = argparse.ArgumentParser(description="Test pre-trained Medusa heads from Hugging Face")
    parser.add_argument("--model", type=str, default="lmsys/vicuna-7b-v1.5",
                        help="Base model name or path")
    parser.add_argument("--medusa-repo", type=str, default="auto",
                        help="Hugging Face repo with Medusa heads ('auto' for automatic selection)")
    parser.add_argument("--prompt", type=str, default="Explain the concept of speculative decoding in simple terms.",
                        help="Test prompt")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--with-medusa", action="store_true", default=True,
                        help="Use Medusa for generation")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark comparing with and without Medusa")
    return parser.parse_args()

def format_time(seconds):
    """Format time in seconds to human-readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

def main():
    args = parse_args()
    
    # List compatible Medusa repositories if in auto mode
    if args.medusa_repo == "auto":
        compatible_repos = get_compatible_medusa_repos(args.model)
        print(f"Found compatible Medusa repositories for {args.model}:")
        for repo in compatible_repos:
            print(f"  - {repo}")
        print(f"Automatically selecting: {compatible_repos[0]}")
        medusa_repo = compatible_repos[0]
    else:
        medusa_repo = args.medusa_repo
    
    # Initialize model with Medusa from HF
    print(f"Initializing model {args.model} with Medusa heads from {medusa_repo}")
    model = MedusaModel(
        model_name_or_path=args.model,
        precision="fp16",
        medusa_hf_repo=medusa_repo if args.with_medusa else None,
        temperature=args.temperature
    )
    
    if args.benchmark:
        # Run benchmark comparing with and without Medusa
        print("\n=== Running Benchmark ===")
        print(f"Prompt: {args.prompt}")
        print(f"Max tokens: {args.max_tokens}")
        
        # Run without Medusa first
        print("\nGenerating WITHOUT Medusa:")
        # Temporarily disable Medusa
        medusa_heads_backup = model.medusa_heads
        model.medusa_heads = None
        
        start_time = time.time()
        result_no_medusa = model.generate(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        time_no_medusa = time.time() - start_time
        
        tokens_no_medusa = result_no_medusa["usage"]["completion_tokens"]
        tps_no_medusa = tokens_no_medusa / time_no_medusa
        
        print(f"Generated {tokens_no_medusa} tokens in {time_no_medusa:.2f}s")
        print(f"Speed: {tps_no_medusa:.2f} tokens/s")
        
        # Restore Medusa heads
        model.medusa_heads = medusa_heads_backup
        
        # Now run with Medusa
        print("\nGenerating WITH Medusa:")
        start_time = time.time()
        result_with_medusa = model.generate(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        time_with_medusa = time.time() - start_time
        
        tokens_with_medusa = result_with_medusa["usage"]["completion_tokens"]
        tps_with_medusa = tokens_with_medusa / time_with_medusa
        efficiency = result_with_medusa["medusa_stats"]["medusa_efficiency"]
        
        print(f"Generated {tokens_with_medusa} tokens in {time_with_medusa:.2f}s")
        print(f"Speed: {tps_with_medusa:.2f} tokens/s")
        print(f"Medusa efficiency: {efficiency*100:.1f}%")
        
        # Print comparison
        speedup = tps_with_medusa / tps_no_medusa
        print("\n=== Comparison ===")
        print(f"Speedup with Medusa: {speedup:.2f}x")
        print(f"Without Medusa: {tps_no_medusa:.2f} tokens/s")
        print(f"With Medusa: {tps_with_medusa:.2f} tokens/s")
        
        # Calculate time savings for different generation lengths
        print("\nEstimated time for generating different lengths:")
        for length in [100, 500, 1000, 2000]:
            time_normal = length / tps_no_medusa
            time_medusa = length / tps_with_medusa
            time_saved = time_normal - time_medusa
            print(f"  {length} tokens: {format_time(time_normal)} → {format_time(time_medusa)} (save {format_time(time_saved)})")
    
    else:
        # Just run a single generation
        print(f"\nGenerating with prompt: {args.prompt}")
        start_time = time.time()
        result = model.generate(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        elapsed = time.time() - start_time
        
        # Print results
        print("\n=== Generated Text ===")
        print(result["choices"][0]["text"])
        
        print("\n=== Stats ===")
        print(f"Generated {result['usage']['completion_tokens']} tokens in {elapsed:.2f}s")
        print(f"Speed: {result['medusa_stats']['tokens_per_second']:.2f} tokens/s")
        if args.with_medusa:
            print(f"Medusa efficiency: {result['medusa_stats']['medusa_efficiency']*100:.1f}%")
            print(f"Medusa accepted tokens: {result['medusa_stats']['accepted_tokens']}")
            print(f"Medusa rejected tokens: {result['medusa_stats']['rejected_tokens']}")

if __name__ == "__main__":
    main()
