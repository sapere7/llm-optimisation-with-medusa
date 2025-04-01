# Medusa LLM Service (DeepSpeed + HF Backend)

This project implements a high-performance FastAPI service for serving Large Language Models (LLMs) like `lmsys/vicuna-7b-v1.3`, optimized using DeepSpeed-Inference. It features speculative decoding acceleration via a custom Medusa head, parallel batch processing within the model, dynamic request batching for non-streaming requests, and true token streaming.

## Key Features

- **DeepSpeed Optimization**: Uses `deepspeed.init_inference` to optimize the Hugging Face base model (`medusa_model.py`), potentially leveraging kernel injection and other optimizations. Configured to provide hidden states needed for Medusa.
- **Speculative Decoding**: Employs a custom Medusa head and efficient parallel verification logic (`medusa_model.py`) to accelerate generation when `use_speculative=True`.
- **Parallel Batch Processing**: The core generation logic (`MedusaModel.generate`) processes batches of requests in parallel for improved throughput.
- **Dynamic Batching (Non-Streaming)**: Uses an asynchronous queue (`DynamicBatcher` from `batch_processor.py`, managed in `app.py`) to group incoming non-streaming requests before passing the *entire batch* to the parallel `MedusaModel.generate` method, optimizing resource utilization under concurrent load.
- **True Token Streaming**: Provides real-time token-by-token streaming via Server-Sent Events (SSE) for both standard and speculative modes using `MedusaModel.generate_stream`.
- **FastAPI Service**: Robust and scalable API built with FastAPI (`app.py`), including request validation, rate limiting, and health/info endpoints.

## Architecture & Design Decisions

This implementation uses a **single Hugging Face Transformers backend**, optimized with **DeepSpeed-Inference**.

- **Model Loading**: Loads a standard HF model (configurable via `MODEL_NAME` environment variable, e.g., `lmsys/vicuna-7b-v1.3`).
- **Optimization**: `deepspeed.init_inference` is applied to the loaded model during startup (`medusa_model.py`). DeepSpeed was chosen for its robust integration with Hugging Face models and its ability to provide hidden states needed for Medusa.
- **Standard Generation (`use_speculative=False`)**: Handled by `MedusaModel`'s `generate` (batch) or `generate_stream` (streaming) methods, falling back to standard `transformers` generation on the DeepSpeed-optimized model.
- **Speculative Generation (`use_speculative=True`)**: Handled by the same `MedusaModel` methods, activating Medusa head drafting and efficient parallel verification.

This approach provides a unified backend optimized by DeepSpeed and enables functional speculative decoding.

**Batching Implementation**:
-   **Non-Streaming Requests**: Handled by `DynamicBatcher` (`batch_processor.py`), which collects requests and dispatches them as a batch to `process_batch` in `app.py`. `process_batch` calls `MedusaModel.generate` once with the full batch.
-   **Streaming Requests**: Bypass the `DynamicBatcher`. The router (`app.py`) directly calls `MedusaModel.generate_stream` for each request.

**Streaming Implementation**:
-   Achieved using `AsyncGenerator` functions in `MedusaModel` and `app.py`, yielding Server-Sent Events (SSE) formatted data containing token deltas for both standard and speculative modes.

## How Speculative Decoding Works (Medusa - Corrected)

1.  **Drafting**: Given the current sequence batch, the base model (DeepSpeed engine) calculates hidden states. The Medusa head takes the last hidden states and predicts draft candidates in parallel for each sequence.
2.  **Verification**: The original sequences *plus* their corresponding best draft sequences are fed into the base model in a *single* parallel forward pass.
3.  **Acceptance**: Draft tokens are compared against the base model's predictions (sampled from logits) in parallel. Matches are accepted sequentially along each draft path until a mismatch occurs. The base model's token is used at the mismatch point. If all draft tokens match, the token predicted *after* the draft is also appended.
4.  **Efficiency**: By accepting multiple tokens (`k`) in a single base model forward pass and processing the batch in parallel, significant speedups can be achieved compared to standard autoregressive generation.

## Requirements

-   Python 3.8+ (Python 3.11/3.12 recommended)
-   PyTorch (with CUDA support recommended)
-   `deepspeed`, `transformers`, `fastapi`, `uvicorn`, `aiohttp`, `pydantic`, `numpy`, `protobuf`, `sentencepiece`, `accelerate`, `python-dotenv`
-   (Optional) `bitsandbytes` for quantization.
-   (Testing/Benchmarking) `pytest`, `pytest-asyncio`, `httpx`, `tqdm`.
-   Sufficient RAM and VRAM for the chosen Hugging Face model (e.g., `lmsys/vicuna-7b-v1.3`) and DeepSpeed engine.

## Installation

1.  **Clone Repository**:
    ```bash
    git clone https://github.com/sapere7/llm-optimisation-with-medusa
    cd llm-optimisation-with-medusa
    ```
2.  **Create Virtual Environment (Recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate # Linux/macOS
    # venv\Scripts\activate # Windows
    ```
3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    # Ensure deepspeed installs correctly (may require C++ build tools and CUDA toolkit)
    ```

4.  **Download Models**:
    -   **Hugging Face Model**: The code will automatically download the model specified by `MODEL_NAME` in the `.env` file (or the default in `app.py`) on first run. Ensure you are logged into Hugging Face (`huggingface-cli login`) or provide an `HF_TOKEN` in `.env` if the model requires authentication (like Vicuna). Accept model terms on the Hugging Face Hub page if necessary.
    -   **Medusa Head**: A pre-trained Medusa head compatible with the base model is needed for optimal performance (e.g., `medusa-vicuna-7b.pt`). Place it in a location accessible by the application (e.g., a `models/` directory) and set the `MEDUSA_MODEL_PATH` in the `.env` file. If the path is invalid or not set, a new, *untrained* head will be initialized (leading to poor speculative performance). Alternatively, set `MEDUSA_HF_REPO` in `.env` to attempt loading from Hugging Face Hub.

## Running the Service

1.  **Configure Environment**: Create a `.env` file in the project root or set environment variables:
    ```dotenv
    # .env example
    MODEL_NAME="lmsys/vicuna-7b-v1.3"
    MEDUSA_MODEL_PATH="./models/medusa-vicuna-7b.pt" # Adjust path as needed
    DEVICE="cuda" # Or "cpu"
    # MAX_GPU_MEMORY=20 # Optional: Limit GPU memory in GB (requires adjustment in medusa_model.py if used)
    # HF_TOKEN="hf_..." # Optional: For private models
    ```

2.  **Start the Server**:
    ```bash
    python app.py
    ```
    The server will load the model, initialize DeepSpeed, load/initialize the Medusa head, and start listening (default: `http://localhost:8000`).

## API Usage

Use any HTTP client (like `curl`) to interact with the `/v1/completions` endpoint.

### Example: Non-Streaming Request (Speculative)

```bash
curl -X POST http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
  "prompt": "Write a short history of the internet:",
  "max_tokens": 100,
  "use_speculative": true,
  "stream": false
}'
```

### Example: Streaming Request (Speculative)

```bash
curl -N -X POST http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
  "prompt": "Write a short history of the internet:",
  "max_tokens": 100,
  "use_speculative": true,
  "stream": true
}'
```

### Health / Metrics Endpoint

```bash
curl http://localhost:8000/health
```

## Benchmarking

Run the benchmark script (`benchmark.py`) after starting the server:

```bash
python benchmark.py --url http://localhost:8000 --num-requests 100 --concurrency 10 --max-tokens 128
```
This compares standard vs. speculative performance under concurrent load. Results are saved to CSV files in the `benchmark_results` directory (created automatically). Adjust arguments as needed (see `python benchmark.py --help`).

## Performance Notes

-   Speculative decoding speedup depends heavily on the quality of the trained Medusa head. An untrained head will likely result in slower performance than standard generation.
-   DeepSpeed optimization effectiveness depends on hardware and configuration.
-   Parallel batching significantly improves throughput under concurrent load for non-streaming requests.
-   True streaming provides lower time-to-first-token.

## License

MIT
