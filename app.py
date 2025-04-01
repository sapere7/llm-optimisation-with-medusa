import os
import time
import asyncio
import uuid
import json
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import logging
from dotenv import load_dotenv


from medusa_model import MedusaModel
from batch_processor import DynamicBatcher, TokenBucketRateLimiter

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MedusaAPI")

# Get settings from environment
MODEL_NAME = os.getenv("MODEL_NAME", "lmsys/vicuna-7b-v1.3") # Keep TinyLlama default
MODEL_PRECISION = os.getenv("MODEL_PRECISION", "fp16")
MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "2048"))
MEDUSA_CHOICES = [int(x) for x in os.getenv("MEDUSA_CHOICES", "3,3,3").split(",")]
MEDUSA_DEPTH = int(os.getenv("MEDUSA_DEPTH", "3"))
MEDUSA_MODEL_PATH = os.getenv("MEDUSA_MODEL_PATH", "./models/medusa-tinyllama-1.1b.pt")
MEDUSA_HF_REPO = os.getenv("MEDUSA_HF_REPO")
MEDUSA_HF_SUBFOLDER = os.getenv("MEDUSA_HF_SUBFOLDER", "")
HF_TOKEN = os.getenv("HF_TOKEN")

MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "8"))
MAX_WAIT_TIME = float(os.getenv("MAX_WAIT_TIME", "0.1"))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "50"))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "512"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
# DS_ZERO_STAGE = int(os.getenv("DS_ZERO_STAGE", "0")) # Removed - Training param
# DS_OFFLOAD_PARAM = os.getenv("DS_OFFLOAD_PARAM", "false").lower() == "true" # Removed - Training param
# DS_OFFLOAD_OPTIMIZER = os.getenv("DS_OFFLOAD_OPTIMIZER", "false").lower() == "true" # Removed - Training param
# DS_INFERENCE_THREADS = int(os.getenv("DS_INFERENCE_THREADS", "4")) # Removed - Invalid param name?
MAX_GPU_MEMORY = int(os.getenv("MAX_GPU_MEMORY", "0")) # Keep for memory constraint

# Global variables
medusa_model: Optional[MedusaModel] = None
batcher: Optional[DynamicBatcher] = None
rate_limiter: Optional[TokenBucketRateLimiter] = None

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    global medusa_model, batcher, rate_limiter
    logger.info("Starting up MedusaAPI service")
    try:
        logger.info(f"Loading model {MODEL_NAME}")
        # Corrected ds_kwargs passed to MedusaModel init
        ds_kwargs = {
             # Add any specific DS inference params needed, e.g., tensor parallel size if > 1
             # "tensor_parallel": {"tp_size": 1}, # Example, already default in MedusaModel
        }
        medusa_model = MedusaModel(
            model_name_or_path=MODEL_NAME, medusa_choices=MEDUSA_CHOICES, tree_depth=MEDUSA_DEPTH,
            precision=MODEL_PRECISION, max_context_length=MAX_CONTEXT_LENGTH,
            medusa_model_path=MEDUSA_MODEL_PATH if os.path.exists(MEDUSA_MODEL_PATH) else None,
            medusa_hf_repo=MEDUSA_HF_REPO, medusa_hf_subfolder=MEDUSA_HF_SUBFOLDER,
            max_gpu_memory=MAX_GPU_MEMORY if MAX_GPU_MEMORY > 0 else None,
            ds_inference_kwargs=ds_kwargs, hf_token=HF_TOKEN
        )
        logger.info("Model loaded successfully")

        async def process_batch(batch_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """ Processes a batch using MedusaModel.generate """
            if not batch_requests: return []
            logger.info(f"Processing batch of {len(batch_requests)} requests...")
            start_batch_proc_time = time.time()
            prompts = [req["prompt"] for req in batch_requests]
            first_req = batch_requests[0]
            max_tokens = first_req.get("max_tokens", DEFAULT_MAX_TOKENS)
            temperature = first_req.get("temperature", DEFAULT_TEMPERATURE)
            top_p = first_req.get("top_p", 0.9)
            top_k = first_req.get("top_k", 0)
            stop = first_req.get("stop")
            use_speculative = any(req.get("use_speculative", True) for req in batch_requests)
            try:
                result_batch_dict = medusa_model.generate(
                    prompt=prompts, max_tokens=max_tokens, temperature=temperature,
                    top_p=top_p, top_k=top_k, stop=stop, use_speculative=use_speculative
                )
                if "error" in result_batch_dict:
                     logger.error(f"MedusaModel.generate returned an error: {result_batch_dict['error']}")
                     return [{"id": req["id"], "error": result_batch_dict["error"]} for req in batch_requests]
                choices = result_batch_dict.get("choices", [])
                usage = result_batch_dict.get("usage", {})
                stats = result_batch_dict.get("medusa_stats")
                if len(choices) != len(batch_requests):
                     logger.error(f"Batch size mismatch: expected {len(batch_requests)}, got {len(choices)}")
                     return [{"id": req["id"], "error": "Internal error: Batch result size mismatch."} for req in batch_requests]
                final_results = []
                for i, choice in enumerate(choices):
                     res_with_id = {
                         "id": batch_requests[i]["id"], "object": result_batch_dict.get("object", "text_completion"),
                         "created": result_batch_dict.get("created", int(time.time())), "model": result_batch_dict.get("model", medusa_model.model_name),
                         "choices": [choice], "usage": usage, "medusa_stats": stats
                     }
                     final_results.append(res_with_id)
                logger.info(f"Batch processing took {time.time() - start_batch_proc_time:.2f}s")
                return final_results
            except Exception as e:
                logger.error(f"Error processing batch in process_batch: {e}", exc_info=True)
                return [{"id": req["id"], "error": f"Batch processing failed: {str(e)}"} for req in batch_requests]

        batcher = DynamicBatcher(
            batch_processor=process_batch, max_batch_size=MAX_BATCH_SIZE,
            max_wait_time=MAX_WAIT_TIME, max_queue_size=MAX_QUEUE_SIZE
        )
        await batcher.start()
        logger.info("Dynamic batcher started")
        rate_limiter = TokenBucketRateLimiter(tokens_per_second=10.0, bucket_size=30)
    except Exception as e: logger.error(f"Error during startup: {e}"); raise
    yield
    logger.info("Shutting down MedusaAPI service")
    if batcher: logger.info("Stopping dynamic batcher"); await batcher.stop(graceful=True)
    if medusa_model: logger.info("Cleaning up model resources"); del medusa_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    logger.info("Shutdown complete")

# Initialize FastAPI app
app = FastAPI(title="Medusa LLM API", description="High-performance LLM API using Medusa", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Request/Response models
class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(DEFAULT_MAX_TOKENS, ge=1, le=4096)
    temperature: float = Field(DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(0, ge=0)
    stop: Optional[List[str]] = None
    echo: bool = False
    stream: bool = False
    use_speculative: bool = True

class TokenUsage(BaseModel): prompt_tokens: int; completion_tokens: int; total_tokens: int
class MedusaStats(BaseModel):
    total_verification_steps: Optional[int] = None; total_draft_tokens: Optional[int] = None
    total_accepted_tokens: Optional[int] = None; avg_acceptance_rate: Optional[float] = None
    avg_speedup_factor: Optional[float] = None; tokens_per_second: Optional[float] = None
    elapsed_time: Optional[float] = None; error: Optional[str] = None
class GenerationChoice(BaseModel): text: str; index: int = 0; finish_reason: str
class GenerationResponse(BaseModel): id: str; object: str = "text_completion"; created: int; model: str; choices: List[GenerationChoice]; usage: TokenUsage; medusa_stats: Optional[MedusaStats] = None

# Rate limiting dependency
async def check_rate_limit(request: Request):
    client_host = request.client.host if request.client else "unknown"
    if not await rate_limiter.acquire():
        logger.warning(f"Rate limit exceeded for {client_host}")
        raise HTTPException(status_code=429, detail="Too many requests.")

# --- Streaming Response Helper ---
async def stream_generator_wrapper(request_body: GenerationRequest) -> AsyncGenerator[str, None]:
    """ Wraps the MedusaModel stream generator to format SSE chunks. """
    global medusa_model
    stream_id = f"cmpl-{str(uuid.uuid4())}"
    model_name = f"{medusa_model.model_name}-{'spec' if request_body.use_speculative else 'std'}"
    created_time = int(time.time())
    try:
        token_generator = medusa_model.generate_stream(
            prompt=request_body.prompt, max_tokens=request_body.max_tokens,
            temperature=request_body.temperature, top_p=request_body.top_p, top_k=request_body.top_k,
            stop=request_body.stop, use_speculative=request_body.use_speculative
        )
        async for delta in token_generator:
            chunk = {
                "id": stream_id, "object": "text_completion", "created": created_time, "model": model_name,
                "choices": [{"delta": {"content": delta.get("text", "")}, "index": 0, "finish_reason": delta.get("finish_reason")}]
            }
            sse_data = json.dumps(chunk)
            yield f"data: {sse_data}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Error during streaming generation: {e}", exc_info=True)
        error_chunk = {
             "id": stream_id, "object": "text_completion", "created": created_time, "model": model_name,
             "choices": [{"delta": {}, "index": 0, "finish_reason": "error"}]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"


# API endpoints
@app.post("/v1/completions", dependencies=[Depends(check_rate_limit)]) # Remove response_model for streaming
async def generate_completion(request_body: GenerationRequest):
    """ Generate text completion. Handles batching for non-streaming, streams directly otherwise. """
    global medusa_model, batcher
    if medusa_model is None or batcher is None: raise HTTPException(status_code=503, detail="Model initializing.")

    try:
        # --- Handle Streaming ---
        if request_body.stream:
            logger.info(f"Handling streaming request (speculative={request_body.use_speculative})")
            # Directly call the model's stream generator via the SSE wrapper
            return StreamingResponse(stream_generator_wrapper(request_body), media_type="text/event-stream")

        # --- Handle Non-Streaming ---
        else:
            request_id = f"medusa-{str(uuid.uuid4())}"
            content = request_body.model_dump()
            content["id"] = request_id # Add ID for batch processor mapping

            logger.info(f"Adding non-streaming request {request_id} to batcher (speculative={request_body.use_speculative})")
            result = await batcher.add_request(content, request_id)

            if isinstance(result, dict) and "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])

            # Validate and return the result from batcher
            try:
                if not isinstance(result, dict): raise TypeError(f"Expected dict result, got {type(result)}")
                result.pop("id", None) # Remove internal ID
                # Validate against Pydantic model before returning
                return GenerationResponse(**result)
            except Exception as validation_error:
                 logger.error(f"Output validation failed for request {request_id}: {validation_error}\nResult: {result}", exc_info=True)
                 raise HTTPException(status_code=500, detail=f"Output validation failed: {validation_error}")

    except HTTPException: raise
    except Exception as e: logger.error(f"Error in generate_completion: {e}"); raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    status = "ok" if medusa_model is not None and batcher is not None else "initializing"
    gpu_info = "N/A"
    if torch.cuda.is_available():
        try:
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            gpu_info = f"{gpu_memory_allocated:.2f}GB allocated, {gpu_memory_reserved:.2f}GB reserved"
        except: gpu_info = "Error getting GPU stats"
    batcher_stats = batcher._report_stats() if batcher else {}
    return {"status": status, "model": MODEL_NAME, "gpu": gpu_info, "batcher": batcher_stats, "timestamp": int(time.time())}

@app.get("/model/info")
async def model_info():
    return {"model_name": MODEL_NAME, "precision": MODEL_PRECISION, "max_context_length": MAX_CONTEXT_LENGTH, "medusa_choices": MEDUSA_CHOICES, "medusa_depth": MEDUSA_DEPTH, "max_batch_size": MAX_BATCH_SIZE, "default_max_tokens": DEFAULT_MAX_TOKENS, "default_temperature": DEFAULT_TEMPERATURE}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # (HTML UI remains the same)
    return """
    <!DOCTYPE html><html><head><title>Medusa LLM API</title><style>body{font-family:Arial,sans-serif;max-width:800px;margin:0 auto;padding:20px;line-height:1.6}textarea{width:100%;height:150px;margin-bottom:10px;padding:8px;font-family:monospace}input,select{margin-bottom:10px;padding:8px}button{padding:10px 15px;background:#4CAF50;color:white;border:none;cursor:pointer;margin-right:10px}button:hover{background:#45a049}#result{white-space:pre-wrap;background:#f5f5f5;padding:15px;margin-top:20px;border-radius:4px}#status{margin-top:20px}.controls{display:flex;gap:10px;margin-bottom:10px}.controls div{flex:1}label{display:block;margin-bottom:5px;font-weight:bold}.spinner{display:inline-block;width:20px;height:20px;border:3px solid rgba(0,0,0,.3);border-radius:50%;border-top-color:#000;animation:spin 1s ease-in-out infinite;margin-left:10px}@keyframes spin{to{transform:rotate(360deg)}}.hidden{display:none}.stats{margin-top:15px;font-size:14px;color:#666}</style></head><body><h1>Medusa LLM API Test UI</h1><div><label for="prompt">Prompt:</label><textarea id="prompt" placeholder="Enter your prompt here..."></textarea><div class="controls"><div><label for="max_tokens">Max Tokens:</label><input type="number" id="max_tokens" value="256" min="1" max="4096"></div><div><label for="temperature">Temperature:</label><input type="number" id="temperature" value="0.7" min="0" max="2" step="0.1"></div><div><label for="top_p">Top P:</label><input type="number" id="top_p" value="0.95" min="0" max="1" step="0.01"></div></div><div class="controls"><div><label for="top_k">Top K:</label><input type="number" id="top_k" value="0" min="0" step="1"></div><div><label for="echo">Echo Prompt:</label><input type="checkbox" id="echo"></div><div><label for="stop">Stop Sequences (comma separated):</label><input type="text" id="stop" placeholder="e.g. '\\n\\n', 'END'"></div><div><label for="use_speculative">Use Speculative:</label><input type="checkbox" id="use_speculative" checked></div></div><button id="generate">Generate</button><button id="clear">Clear</button><div id="spinner" class="spinner hidden"></div></div><div id="result"></div><div id="status"><h3>System Status</h3><div id="status-content">Loading...</div></div>
    <script>
        fetchStatus(); setInterval(fetchStatus, 10000);
        function fetchStatus(){fetch('/health').then(r=>r.json()).then(d=>{let h=`<p><strong>Status:</strong> ${d.status}</p><p><strong>Model:</strong> ${d.model}</p><p><strong>GPU:</strong> ${d.gpu}</p>`;if(d.batcher){h+=`<div class="stats"><p><strong>Batcher Stats:</strong></p><ul><li>Uptime: ${d.batcher.uptime}</li><li>Total requests: ${d.batcher.total_requests}</li><li>Queue size: ${d.batcher.queue_size}</li><li>Throughput: ${d.batcher.throughput}</li></ul></div>`}document.getElementById('status-content').innerHTML=h}).catch(e=>{document.getElementById('status-content').innerHTML=`<p>Error: ${e}</p>`})}
        document.getElementById('generate').addEventListener('click',async()=>{const p=document.getElementById('prompt').value;if(!p){alert('Enter prompt');return}const mt=parseInt(document.getElementById('max_tokens').value);const t=parseFloat(document.getElementById('temperature').value);const tp=parseFloat(document.getElementById('top_p').value);const tk=parseInt(document.getElementById('top_k').value);const e=document.getElementById('echo').checked;const s=document.getElementById('stop').value.split(',').map(x=>x.trim()).filter(x=>x);const us=document.getElementById('use_speculative').checked;const rD=document.getElementById('result');const sp=document.getElementById('spinner');const gB=document.getElementById('generate');rD.textContent='Generating...';sp.classList.remove('hidden');gB.disabled=true;try{const rsp=await fetch('/v1/completions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:p,max_tokens:mt,temperature:t,top_p:tp,top_k:tk,echo:e,stop:s.length>0?s:null,use_speculative:us})});const d=await rsp.json();if(rsp.ok){const txt=d.choices[0].text;const st=d.medusa_stats;let res=`Generated Text:\n${txt}\n\n`;if(st){res+=`Medusa Stats:\n- Tokens/sec: ${st.tokens_per_second?.toFixed(2) ?? 'N/A'}\n- Efficiency: ${st.medusa_efficiency != null ? (st.medusa_efficiency*100).toFixed(1)+'%' : 'N/A'}\n- Accepted tokens: ${st.accepted_tokens ?? 'N/A'}\n- Rejected tokens: ${st.rejected_tokens ?? 'N/A'}\n- Total time: ${st.elapsed_time?.toFixed(2) ?? 'N/A'}s\n`}else{res+='(Standard Generation Stats)\n'}rD.textContent=res}else{rD.textContent=`Error: ${d.detail||'Unknown'}`}}catch(err){rD.textContent=`Error: ${err.message}`}finally{sp.classList.add('hidden');gB.disabled=false}});
        document.getElementById('clear').addEventListener('click',()=>{document.getElementById('prompt').value='';document.getElementById('result').textContent=''});
    </script></body></html>
    """

# Training endpoint (remains the same)
@app.post("/train")
async def train_medusa(background_tasks: BackgroundTasks):
    from medusa_model import MedusaTrainer
    if medusa_model is None: raise HTTPException(status_code=503, detail="Model initializing.")
    async def run_training():
        try:
            logger.info("Starting Medusa head training")
            trainer = MedusaTrainer(medusa_model=medusa_model, learning_rate=5e-5, weight_decay=0.01)
            if os.path.exists("training_data.jsonl"):
                trainer.train_on_dataset(dataset_path="training_data.jsonl", num_epochs=1, save_path=MEDUSA_MODEL_PATH)
                logger.info(f"Training complete. Model saved to {MEDUSA_MODEL_PATH}")
            else: logger.error("Training data file not found")
        except Exception as e: logger.error(f"Error during training: {e}")
    background_tasks.add_task(run_training)
    return {"status": "Training started in the background"}

# Run the application
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
