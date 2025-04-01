#!/usr/bin/env python
"""
Client for testing the Medusa LLM API with parallel requests and benchmarking.
"""
import argparse
import asyncio
import time
import json
import httpx
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import csv
import os
import random

@dataclass
class RequestResult:
    """Results from a single request."""
    request_id: str
    prompt: str
    response: str
    tokens_generated: int
    prompt_tokens: int
    time_taken: float  # seconds
    tokens_per_second: float
    medusa_efficiency: float
    status_code: int
    error: Optional[str] = None

class MedusaClient:
    """Client for the Medusa LLM API."""
    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        timeout: float = 120.0,
        concurrency: int = 4
    ):
        """
        Initialize the client.
        
        Args:
            api_url: Base URL for the API
            timeout: Request timeout in seconds
            concurrency: Maximum number of concurrent requests
        """
        self.api_url = api_url
        self.timeout = timeout
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)
        
        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens = 0
        self.total_time = 0.0
        
        # Create async client
        self.client = httpx.AsyncClient(
            timeout=timeout,
            base_url=api_url
        )
    
    async def close(self):
        """Close the client."""
        await self.client.aclose()
    
    async def check_health(self) -> Dict[str, Any]:
        """Check if the API is healthy."""
        response = await self.client.get("/health")
        response.raise_for_status()
        return response.json()
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        stop: Optional[List[str]] = None,
        echo: bool = False,
        request_id: Optional[str] = None
    ) -> RequestResult:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stop: Stop sequences
            echo: Whether to include prompt in output
            request_id: Optional request ID for tracking
            
        Returns:
            RequestResult object with request stats
        """
        # Create request payload
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stop": stop,
            "echo": echo
        }
        
        # Generate a request ID if not provided
        if request_id is None:
            request_id = f"req-{self.total_requests + 1}"
        
        # Initialize result with defaults
        result = RequestResult(
            request_id=request_id,
            prompt=prompt,
            response="",
            tokens_generated=0,
            prompt_tokens=0,
            time_taken=0.0,
            tokens_per_second=0.0,
            medusa_efficiency=0.0,
            status_code=0,
            error=None
        )
        
        start_time = time.time()
        self.total_requests += 1
        
        # Use semaphore to limit concurrency
        async with self.semaphore:
            try:
                # Make the request
                response = await self.client.post(
                    "/v1/completions",
                    json=payload
                )
                
                # Record status code
                result.status_code = response.status_code
                
                # Check for success
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract response data
                    result.response = data["choices"][0]["text"]
                    result.tokens_generated = data["usage"]["completion_tokens"]
                    result.prompt_tokens = data["usage"]["prompt_tokens"]
                    
                    # Extract Medusa stats if available
                    if "medusa_stats" in data:
                        result.medusa_efficiency = data["medusa_stats"]["medusa_efficiency"]
                    
                    self.successful_requests += 1
                    self.total_tokens += result.tokens_generated
                else:
                    # Handle error
                    result.error = response.text
                    self.failed_requests += 1
            
            except Exception as e:
                # Handle exceptions
                result.error = str(e)
                result.status_code = 500  # Generic error code
                self.failed_requests += 1
        
        # Calculate time taken
        end_time = time.time()
        result.time_taken = end_time - start_time
        self.total_time += result.time_taken
        
        # Calculate tokens per second
        if result.tokens_generated > 0 and result.time_taken > 0:
            result.tokens_per_second = result.tokens_generated / result.time_taken
        
        return result
    
    async def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        show_progress: bool = True
    ) -> List[RequestResult]:
        """
        Generate text for multiple prompts in parallel.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate per prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            show_progress: Whether to show progress bar
            
        Returns:
            List of RequestResult objects
        """
        # Create tasks for each prompt
        tasks = []
        for i, prompt in enumerate(prompts):
            request_id = f"batch-{i+1}"
            task = self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                request_id=request_id
            )
            tasks.append(task)
        
        # Run tasks with progress bar if requested
        if show_progress:
            results = []
            progress_bar = tqdm(total=len(tasks), desc="Generating")
            
            for task in asyncio.as_completed(tasks):
                result = await task
                results.append(result)
                
                # Update progress bar with stats
                progress_bar.set_postfix({
                    "tokens/s": f"{result.tokens_per_second:.1f}",
                    "success": result.error is None
                })
                progress_bar.update(1)
            
            progress_bar.close()
        else:
            # Run without progress bar
            results = await asyncio.gather(*tasks)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        avg_tokens_per_second = self.total_tokens / self.total_time if self.total_time > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_tokens": self.total_tokens,
            "total_time": f"{self.total_time:.2f}s",
            "average_tokens_per_second": f"{avg_tokens_per_second:.2f}",
            "average_request_time": f"{(self.total_time / self.total_requests if self.total_requests > 0 else 0):.2f}s"
        }
    
    def reset_stats(self):
        """Reset client statistics."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens = 0
        self.total_time = 0.0

def save_results(results: List[RequestResult], output_path: str):
    """
    Save results to a file.
    
    Args:
        results: List of RequestResult objects
        output_path: Path to save results
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Determine file type from extension
    if output_path.endswith('.json'):
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
            
    elif output_path.endswith('.csv'):
        # Save as CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
            writer.writeheader()
            for result in results:
                writer.writerow(asdict(result))
    else:
        # Default to text format
        with open(output_path, 'w') as f:
            for i, result in enumerate(results, 1):
                f.write(f"Request {i} ({result.request_id}):\n")
                f.write(f"  Prompt: {result.prompt[:50]}...\n")
                f.write(f"  Response: {result.response[:100]}...\n")
                f.write(f"  Tokens: {result.tokens_generated} (in {result.time_taken:.2f}s, {result.tokens_per_second:.2f} t/s)\n")
                f.write(f"  Medusa Efficiency: {result.medusa_efficiency*100:.1f}%\n")
                if result.error:
                    f.write(f"  Error: {result.error}\n")
                f.write("\n")

def print_summary(results: List[RequestResult]):
    """
    Print a summary of the results.
    
    Args:
        results: List of RequestResult objects
    """
    # Calculate summary statistics
    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]
    
    # Token statistics
    total_tokens = sum(r.tokens_generated for r in successful)
    total_time = sum(r.time_taken for r in results)
    avg_tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    
    # Calculate percentiles for latency
    if successful:
        latencies = [r.time_taken for r in successful]
        latency_p50 = np.percentile(latencies, 50)
        latency_p90 = np.percentile(latencies, 90)
        latency_p99 = np.percentile(latencies, 99)
        
        tps_values = [r.tokens_per_second for r in successful]
        tps_p50 = np.percentile(tps_values, 50)
        
        efficiency_values = [r.medusa_efficiency for r in successful]
        avg_efficiency = np.mean(efficiency_values) if efficiency_values else 0
    else:
        latency_p50 = latency_p90 = latency_p99 = tps_p50 = avg_efficiency = 0
    
    # Print summary
    print("\n=== SUMMARY ===")
    print(f"Total requests: {len(results)}")
    print(f"Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average tokens per second: {avg_tokens_per_second:.2f}")
    print(f"Medusa efficiency: {avg_efficiency*100:.1f}%")
    print("\nLatency:")
    print(f"  p50: {latency_p50:.2f}s")
    print(f"  p90: {latency_p90:.2f}s")
    print(f"  p99: {latency_p99:.2f}s")
    print(f"Throughput p50: {tps_p50:.2f} tokens/s")
    
    if failed:
        print("\nError summary:")
        error_counts = {}
        for result in failed:
            error_msg = result.error[:100] + "..." if result.error and len(result.error) > 100 else result.error
            error_counts[error_msg] = error_counts.get(error_msg, 0) + 1
        
        for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {count} requests: {error}")

async def load_prompts(path: str) -> List[str]:
    """
    Load prompts from a file.
    
    Args:
        path: Path to file with prompts
        
    Returns:
        List of prompts
    """
    prompts = []
    
    if not os.path.exists(path):
        return ["Hello, how are you?", "What is machine learning?", "Explain quantum computing."]
    
    with open(path, 'r', encoding='utf-8') as f:
        if path.endswith('.json') or path.endswith('.jsonl'):
            # JSON format
            if path.endswith('.jsonl'):
                # JSONL: one JSON object per line
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if isinstance(data, dict) and "text" in data:
                            prompts.append(data["text"])
                        elif isinstance(data, dict) and "prompt" in data:
                            prompts.append(data["prompt"])
                        elif isinstance(data, str):
                            prompts.append(data)
                    except:
                        continue
            else:
                # Regular JSON file
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "text" in item:
                                prompts.append(item["text"])
                            elif isinstance(item, dict) and "prompt" in item:
                                prompts.append(item["prompt"])
                            elif isinstance(item, str):
                                prompts.append(item)
                    elif isinstance(data, dict) and "prompts" in data:
                        prompts.extend(data["prompts"])
                except:
                    pass
        else:
            # Plain text format, one prompt per line
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    prompts.append(line)
    
    return prompts

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test client for Medusa LLM API")
    parser.add_argument("--url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--concurrency", type=int, default=4, help="Maximum concurrent requests")
    parser.add_argument("--prompts", type=str, default=None, help="Path to file with prompts")
    parser.add_argument("--num-prompts", type=int, default=10, help="Number of prompts to use")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling parameter")
    parser.add_argument("--output", type=str, default=None, help="Output file for results")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark mode")
    args = parser.parse_args()
    
    # Initialize client
    client = MedusaClient(
        api_url=args.url,
        concurrency=args.concurrency
    )
    
    try:
        # Check if API is healthy
        print(f"Checking API health at {args.url}...")
        health = await client.check_health()
        print(f"API status: {health['status']}")
        
        # Load prompts
        if args.prompts:
            prompts = await load_prompts(args.prompts)
            print(f"Loaded {len(prompts)} prompts from {args.prompts}")
        else:
            # Default prompts
            prompts = [
                "Explain the concept of neural networks in simple terms.",
                "Write a short poem about technology and nature.",
                "What are the key differences between Python and JavaScript?",
                "Describe the process of photosynthesis.",
                "What are the main challenges in artificial intelligence research?",
                "Write a brief summary of the history of space exploration.",
                "Explain the principle of quantum superposition.",
                "What are the best practices for web accessibility?",
                "Describe how blockchain technology works.",
                "What is the significance of the Turing test in AI?"
            ]
            print(f"Using {len(prompts)} default prompts")
        
        # Limit number of prompts if specified
        if args.num_prompts and args.num_prompts < len(prompts):
            prompts = random.sample(prompts, args.num_prompts)
            print(f"Randomly selected {len(prompts)} prompts")
        
        if args.benchmark:
            # Run benchmark mode
            print(f"\nRunning benchmark with {args.concurrency} concurrent requests...")
            print(f"Generating up to {args.max_tokens} tokens per prompt, temp={args.temperature}")
            
            # Generate text for all prompts
            start_time = time.time()
            results = await client.batch_generate(
                prompts=prompts,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
            total_time = time.time() - start_time
            
            # Print summary
            print_summary(results)
            print(f"\nBenchmark completed in {total_time:.2f}s")
            
            # Save results if output path is specified
            if args.output:
                save_results(results, args.output)
                print(f"Results saved to {args.output}")
        
        else:
            # Interactive mode
            print("\nEntering interactive mode. Type 'exit' to quit.")
            
            while True:
                prompt = input("\nEnter prompt: ")
                if prompt.lower() in ('exit', 'quit'):
                    break
                
                print("Generating...")
                start_time = time.time()
                
                result = await client.generate(
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p
                )
                
                elapsed = time.time() - start_time
                
                if result.error:
                    print(f"Error: {result.error}")
                else:
                    print("\nGenerated text:")
                    print(result.response)
                    print("\nStats:")
                    print(f"  Tokens: {result.tokens_generated} in {result.time_taken:.2f}s ({result.tokens_per_second:.2f} t/s)")
                    print(f"  Medusa efficiency: {result.medusa_efficiency*100:.1f}%")
    
    finally:
        # Close client
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
