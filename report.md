# Medusa LLM Service - Implementation Report

## 1. Objective

The goal was to implement a FastAPI service serving the `lmsys/vicuna-7b-v1.3` model, optimized using a compilation library, and accelerated with a custom Medusa head for speculative decoding. The service needed to handle concurrent requests efficiently using dynamic batching and support streaming output.

## 2. Final Architecture

The final implementation utilizes the following architecture:

*   **Base LLM:** Hugging Face `transformers` library (`AutoModelForCausalLM`). Configurable via the `MODEL_NAME` environment variable (e.g., `lmsys/vicuna-7b-v1.3`).
*   **Optimization Library:** **DeepSpeed-Inference** (`deepspeed.init_inference`). DeepSpeed optimizes the loaded Hugging Face model and provides access to hidden states required for Medusa.
*   **Medusa Head:** A custom `MedusaHead` module is implemented in `medusa_model.py`. Multiple heads are instantiated based on configuration. Logic exists to load pre-trained heads from local files or Hugging Face Hub.
*   **Speculative Decoding:** Implemented within the `MedusaModel` class (`medusa_model.py`) using efficient parallel verification (`_verify_candidates_parallel`).
*   **Dynamic Batching:** Implemented in `app.py` using the `DynamicBatcher` class (from `batch_processor.py`). It collects non-streaming requests and calls `MedusaModel.generate` once per batch for parallel processing.
*   **Streaming:** Implemented via the `generate_stream` method in `MedusaModel` and handled by the `/v1/completions` endpoint in `app.py` using `StreamingResponse` for Server-Sent Events (SSE).
*   **API:** A FastAPI application (`app.py`) exposes the `/v1/completions` endpoint, health/info endpoints, and includes rate limiting. A basic HTML UI is provided at the root (`/`).
*   **Testing:** Basic API tests are provided in `test_medusa.py`.
*   **Benchmarking:** A script (`benchmark.py`) is included for performance testing under concurrent load, comparing standard vs. speculative modes.

## 3. Implementation Details & Choices

*   **DeepSpeed Choice:** Chosen over `llama.cpp` for its easier integration with Hugging Face models and straightforward access to hidden states required by the Medusa heads.
*   **Speculative Logic:** The core loop uses parallel verification for efficiency. Drafting currently uses a simple approach (top-1 token from the first head).
*   **Batching:** Dynamic batching at the application level improves throughput for concurrent non-streaming requests by grouping them before sending to the model.
*   **Streaming:** Standard `asyncio` and `TextIteratorStreamer` (for fallback) are used to provide true token streaming.
*   **Modularity:** Code is organized into modules for the model (`medusa_model.py`), API (`app.py`), batching/rate limiting (`batch_processor.py`), HF loading (`load_medusa_from_hf.py`), testing (`test_medusa.py`), and benchmarking (`benchmark.py`).

## 4. Challenges and Considerations

*   **Hardware Requirements:** Running large models like Vicuna-7B requires significant RAM and potentially GPU VRAM, especially when combined with DeepSpeed. Ensure the deployment environment meets these requirements.
*   **Medusa Head Training:** Performance gains from speculative decoding are highly dependent on using *trained* Medusa heads compatible with the base model. This implementation provides the framework and training code (`MedusaTrainer`), but head training itself was outside the scope. Using untrained heads (the default if no pre-trained file is found) will likely degrade performance.
*   **DeepSpeed Configuration:** Fine-tuning DeepSpeed parameters (`ds_inference_kwargs`) might be necessary for optimal performance on specific hardware.

## 5. Testing and Benchmarking

*   **Unit/Integration Tests (`test_medusa.py`):** Basic tests cover API endpoint availability and functionality for non-streaming and streaming modes (both standard and speculative).
*   **Benchmark Script (`benchmark.py`):** Allows testing non-streaming performance under varying concurrency levels. Compares standard vs. speculative modes, measuring throughput (tokens/sec) and latency. Results are saved to CSV.
*   **Expected Benchmark Results:**
    *   Dynamic batching should show significantly higher throughput compared to sequential processing under concurrent load.
    *   Speculative decoding (with *trained* heads) should ideally show higher tokens/second and potentially lower latency compared to standard generation, especially for tasks allowing higher acceptance rates. The efficiency will depend heavily on the quality of the Medusa heads.

## 6. Conclusion

This implementation successfully fulfills the assignment's core requirements by providing a FastAPI service with DeepSpeed optimization, a custom Medusa head implementation featuring efficient parallel verification, dynamic batching, and streaming support. The architecture is designed to be flexible regarding the base model used and leverages DeepSpeed for performance optimization while enabling the necessary components for speculative decoding.
