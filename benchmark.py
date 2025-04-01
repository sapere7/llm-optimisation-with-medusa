#!/usr/bin/env python
"""
Benchmark script for the Medusa LLM API (v2).
"""
import argparse
import asyncio
import time
import json
import httpx
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
import csv
import os
import random
import logging

# Configure logging for the benchmark script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Benchmark")

@dataclass
class BenchmarkRequestResult:
    """Results from a single benchmark request."""
    request_id: str
    prompt_len: int
    completion_tokens: int
    time_to_first_token: Optional[float] = None # Optional: Requires streaming support in client
    total_time: float = 0.0
    tokens_per_second: float = 0.0
    success: bool = False
    status_code: Optional[int] = None
    error: Optional[str] = None
    speculative: bool = False
    medusa_efficiency: Optional[float] = None
    medusa_acceptance_rate: Optional[float] = None
    medusa_steps: Optional[int] = None

class BenchmarkClient:
    """Async client for benchmarking the API."""
    def __init__(self, api_url: str, timeout: float = 180.0):
        self.api_url = api_url
        self.client = httpx.AsyncClient(base_url=api_url, timeout=timeout)

    async def close(self):
        await self.client.aclose()

    async def send_request(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        use_speculative: bool
    ) -> BenchmarkRequestResult:
        """Sends a single non-streaming request and gathers metrics."""
        payload = {
            "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature,
            "top_p": top_p, "stream": False, "use_speculative": use_speculative
        }
        result = BenchmarkRequestResult(request_id=request_id, prompt_len=len(prompt.split()), speculative=use_speculative) # Approx prompt tokens
        start_time = time.monotonic()
        try:
            response = await self.client.post("/v1/completions", json=payload)
            result.total_time = time.monotonic() - start_time
            result.status_code = response.status_code

            if response.status_code == 200:
                data = response.json()
                result.success = True
                result.completion_tokens = data["usage"]["completion_tokens"]
                if result.total_time > 0:
                    result.tokens_per_second = result.completion_tokens / result.total_time
                if data.get("medusa_stats"):
                    # Use the correct keys based on the updated generate method
                    result.medusa_efficiency = data["medusa_stats"].get("avg_acceptance_rate") # Use acceptance rate as efficiency metric
                    result.medusa_acceptance_rate = data["medusa_stats"].get("avg_acceptance_rate")
                    result.medusa_steps = data["medusa_stats"].get("total_verification_steps")
            else:
                result.error = response.text
        except Exception as e:
            result.total_time = time.monotonic() - start_time
            result.error = str(e)
            result.success = False
        return result

async def run_benchmark_scenario(
    client: BenchmarkClient,
    prompts: List[str],
    num_requests: int,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    use_speculative: bool
) -> List[BenchmarkRequestResult]:
    """Runs a benchmark scenario with specified parameters."""
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []
    prompt_cycle = iter(lambda: random.choice(prompts), None)

    async def worker(req_id: int):
        async with semaphore:
            return await client.send_request(
                request_id=f"req_{req_id}",
                prompt=next(prompt_cycle),
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                use_speculative=use_speculative
            )

    for i in range(num_requests):
        tasks.append(worker(i))

    results: List[BenchmarkRequestResult] = []
    progress_bar = tqdm(total=num_requests, desc=f"Speculative={use_speculative}, Concurrency={concurrency}")
    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)
        progress_bar.update(1)
        progress_bar.set_postfix({"Success": result.success, "TPS": f"{result.tokens_per_second:.1f}"})
    progress_bar.close()
    return results

def analyze_results(results: List[BenchmarkRequestResult], scenario_name: str):
    """Analyzes and prints summary statistics for benchmark results."""
    successful_results = [r for r in results if r.success]
    num_requests = len(results)
    num_successful = len(successful_results)
    num_failed = num_requests - num_successful

    if not successful_results:
        logger.warning(f"No successful requests for scenario: {scenario_name}")
        print(f"\n--- Results: {scenario_name} ---")
        print(f"Total Requests: {num_requests}")
        print(f"Successful: 0 (0.0%)")
        print(f"Failed: {num_failed} (100.0%)")
        return

    total_time = sum(r.total_time for r in results) # Use total time of all requests for throughput calc
    total_completion_tokens = sum(r.completion_tokens for r in successful_results)
    avg_throughput_tps = total_completion_tokens / total_time if total_time > 0 else 0
    avg_request_latency = np.mean([r.total_time for r in successful_results])
    latencies = [r.total_time for r in successful_results]
    latency_p50 = np.percentile(latencies, 50)
    latency_p90 = np.percentile(latencies, 90)
    latency_p99 = np.percentile(latencies, 99)
    avg_tps_per_request = np.mean([r.tokens_per_second for r in successful_results])

    print(f"\n--- Results: {scenario_name} ---")
    print(f"Total Requests: {num_requests}")
    print(f"Successful: {num_successful} ({num_successful/num_requests*100:.1f}%)")
    print(f"Failed: {num_failed} ({num_failed/num_requests*100:.1f}%)")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Total Completion Tokens: {total_completion_tokens}")
    print(f"Average Throughput (System): {avg_throughput_tps:.2f} tokens/sec")
    print(f"Average TPS per Request: {avg_tps_per_request:.2f} tokens/sec")
    print(f"Average Request Latency: {avg_request_latency:.3f}s")
    print(f"Latency Percentiles: p50={latency_p50:.3f}s, p90={latency_p90:.3f}s, p99={latency_p99:.3f}s")

    if any(r.speculative for r in successful_results):
        spec_results = [r for r in successful_results if r.speculative]
        if spec_results:
            avg_acceptance = np.mean([r.medusa_acceptance_rate for r in spec_results if r.medusa_acceptance_rate is not None])
            avg_steps = np.mean([r.medusa_steps for r in spec_results if r.medusa_steps is not None])
            print(f"Average Medusa Acceptance Rate: {avg_acceptance*100:.1f}%")
            print(f"Average Medusa Verification Steps: {avg_steps:.1f}")

def save_benchmark_results(results: List[BenchmarkRequestResult], filename: str):
    """Saves benchmark results to a CSV file."""
    if not results: return
    # Ensure directory exists
    output_dir = os.path.dirname(filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    logger.info(f"Benchmark results saved to {filename}")

async def load_prompts_from_file(path: str) -> List[str]:
    """Loads prompts from a file (text, jsonl, json)."""
    if not os.path.exists(path):
        logger.error(f"Prompt file not found: {path}")
        return ["What is DeepSpeed?"] # Default fallback

    prompts = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            if path.endswith('.jsonl'):
                for line in f:
                    try: data = json.loads(line); prompts.append(data.get("prompt") or data.get("text"))
                    except: pass
            elif path.endswith('.json'):
                data = json.load(f)
                if isinstance(data, list): prompts = [item.get("prompt") or item.get("text") for item in data]
                elif isinstance(data, dict) and "prompts" in data: prompts = data["prompts"]
            else: # Plain text
                prompts = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error loading prompts from {path}: {e}")
        if not prompts: prompts = ["What is DeepSpeed?"] # Ensure there's at least one

    return [p for p in prompts if p] # Filter empty prompts

async def main():
    parser = argparse.ArgumentParser(description="Benchmark Medusa LLM API (v2)")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--prompts", default="prompts.jsonl", help="Path to prompts file (jsonl, json, txt)")
    parser.add_argument("--num-requests", type=int, default=50, help="Total number of requests to send")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent requests")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Generation top-p")
    parser.add_argument("--output-dir", default="benchmark_results", help="Directory to save output CSV files")
    args = parser.parse_args()

    client = BenchmarkClient(api_url=args.url)
    prompts = await load_prompts_from_file(args.prompts)
    if not prompts: return

    all_results = {}
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_prefix = f"{args.output_dir}/benchmark_{timestamp}"

    try:
        # --- Benchmark Standard ---
        logger.info("Starting Standard Generation Benchmark...")
        results_std = await run_benchmark_scenario(
            client, prompts, args.num_requests, args.concurrency,
            args.max_tokens, args.temperature, args.top_p, use_speculative=False
        )
        analyze_results(results_std, f"Standard (Concurrency={args.concurrency})")
        all_results["standard"] = results_std
        if results_std:
            save_benchmark_results(results_std, f"{output_prefix}_standard.csv")

        # --- Benchmark Speculative ---
        logger.info("\nStarting Speculative Generation Benchmark...")
        results_spec = await run_benchmark_scenario(
            client, prompts, args.num_requests, args.concurrency,
            args.max_tokens, args.temperature, args.top_p, use_speculative=True
        )
        analyze_results(results_spec, f"Speculative (Concurrency={args.concurrency})")
        all_results["speculative"] = results_spec
        if results_spec:
            save_benchmark_results(results_spec, f"{output_prefix}_speculative.csv")

    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
