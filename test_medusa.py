import pytest
import httpx
import asyncio
import json
from typing import List, Dict, Any

# Assuming the server is running at this base URL
BASE_URL = "http://localhost:8000"
# Use a longer timeout for potentially slow model responses
TIMEOUT = 180.0

@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for the module."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="module")
async def client():
    """Create an async httpx client."""
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=TIMEOUT) as client:
        yield client

@pytest.mark.asyncio
async def test_health_check(client: httpx.AsyncClient):
    """Test the health check endpoint."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["ok", "initializing"] # Allow initializing state
    assert "model" in data

@pytest.mark.asyncio
async def test_model_info(client: httpx.AsyncClient):
    """Test the model info endpoint."""
    response = await client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "precision" in data

@pytest.mark.asyncio
@pytest.mark.parametrize("use_speculative", [False, True])
async def test_non_streaming_completion(client: httpx.AsyncClient, use_speculative: bool):
    """Test non-streaming completion (standard and speculative)."""
    payload = {
        "prompt": "What is the capital of France?",
        "max_tokens": 10,
        "temperature": 0.1, # Low temp for more deterministic output
        "stream": False,
        "use_speculative": use_speculative
    }
    response = await client.post("/v1/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) == 1
    assert "text" in data["choices"][0]
    assert len(data["choices"][0]["text"]) > 0
    assert "usage" in data
    assert "prompt_tokens" in data["usage"]
    assert "completion_tokens" in data["usage"]
    if use_speculative:
        assert "medusa_stats" in data
        assert data["medusa_stats"] is not None
    else:
        assert data.get("medusa_stats") is None

@pytest.mark.asyncio
@pytest.mark.parametrize("use_speculative", [False, True])
async def test_streaming_completion(client: httpx.AsyncClient, use_speculative: bool):
    """Test streaming completion (standard and speculative)."""
    payload = {
        "prompt": "Write a short sentence about clouds.",
        "max_tokens": 15,
        "temperature": 0.1,
        "stream": True,
        "use_speculative": use_speculative
    }
    full_response_text = ""
    chunks_received = 0
    final_chunk_received = False

    async with client.stream("POST", "/v1/completions", json=payload) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        async for line in response.aiter_lines():
            line = line.strip()
            if line.startswith("data:"):
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    final_chunk_received = True
                    break
                try:
                    chunk = json.loads(data_str)
                    chunks_received += 1
                    assert "choices" in chunk
                    assert len(chunk["choices"]) == 1
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    full_response_text += content
                    finish_reason = chunk["choices"][0].get("finish_reason")
                    if finish_reason is not None:
                         # The last content chunk might also contain the finish reason
                         pass
                except json.JSONDecodeError:
                    pytest.fail(f"Failed to decode JSON chunk: {data_str}")

    assert chunks_received > 0
    assert final_chunk_received
    assert len(full_response_text) > 0

# Add more tests as needed, e.g., for error handling, specific parameters etc.
