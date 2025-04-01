import asyncio
import time
import uuid
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
import logging
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BatchProcessor")

@dataclass
class BatchRequest:
    """Representation of a request in the batching system."""
    request_id: str
    content: Dict[str, Any]
    future: asyncio.Future
    arrival_time: float = field(default_factory=time.time)
    processed: bool = False
    priority: int = 0  # Higher priority requests are processed first

class DynamicBatcher:
    """
    Dynamic batching system that collects requests and processes them in batches.
    This implementation supports priority-based processing and graceful shutdown.
    """
    def __init__(
        self, 
        batch_processor: Callable[[List[Dict[str, Any]]], Awaitable[List[Dict[str, Any]]]],
        max_batch_size: int = 8,
        max_wait_time: float = 0.1,
        max_queue_size: int = 100,
        stats_interval: int = 100,  # Report stats every N batches
        graceful_shutdown_timeout: float = 30.0  # Seconds to wait for graceful shutdown
    ):
        """
        Initialize the dynamic batching system.
        
        Args:
            batch_processor: Async function that processes a batch of requests
            max_batch_size: Maximum size of each batch
            max_wait_time: Maximum time to wait for forming a batch (in seconds)
            max_queue_size: Maximum size of the request queue
            stats_interval: Report statistics every N batches
            graceful_shutdown_timeout: Time to wait for graceful shutdown
        """
        self.batch_processor = batch_processor
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.max_queue_size = max_queue_size
        self.stats_interval = stats_interval
        self.graceful_shutdown_timeout = graceful_shutdown_timeout
        
        # Create queue and processing task
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.priority_queue = []  # For priority processing
        self.processing_task = None
        self.shutdown_event = asyncio.Event()
        
        # Stats
        self.total_requests = 0
        self.total_batches = 0
        self.total_processing_time = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.start_time = None
        
        logger.info(
            f"DynamicBatcher initialized with max_batch_size={max_batch_size}, "
            f"max_wait_time={max_wait_time}, max_queue_size={max_queue_size}"
        )
    
    async def start(self):
        """Start the batch processing task."""
        if self.processing_task is None:
            self.shutdown_event.clear()
            self.start_time = time.time()
            self.processing_task = asyncio.create_task(self._process_batches())
            logger.info("Batch processor started")
    
    async def stop(self, graceful: bool = True):
        """
        Stop the batch processing task.
        
        Args:
            graceful: If True, waits for all queued requests to complete
        """
        if self.processing_task is not None:
            logger.info(f"Stopping batch processor (graceful={graceful})")
            
            if graceful:
                # Signal shutdown but allow queue to drain
                self.shutdown_event.set()
                
                # Wait for queue to drain or timeout
                try:
                    await asyncio.wait_for(
                        self.queue.join(), 
                        timeout=self.graceful_shutdown_timeout
                    )
                    logger.info("All queued requests processed")
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Graceful shutdown timed out after {self.graceful_shutdown_timeout}s"
                    )
            
            # Cancel the processing task
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
            
            self.processing_task = None
            
            # Report final stats
            self._report_stats(final=True)
            
            logger.info("Batch processor stopped")
    
    async def add_request(
        self, 
        content: Dict[str, Any], 
        request_id: str = None,
        priority: int = 0
    ) -> Any:
        """
        Add a request to be processed.
        
        Args:
            content: The request content
            request_id: Optional request ID (generates UUID if not provided)
            priority: Higher priority requests are processed first
            
        Returns:
            Response for the request
        """
        if self.processing_task is None or self.shutdown_event.is_set():
            raise RuntimeError("Batch processor not running")
        
        # Generate request ID if not provided
        if request_id is None:
            request_id = str(uuid.uuid4())
            
        # Create future for this request
        future = asyncio.get_event_loop().create_future()
        
        # Create request object
        request = BatchRequest(
            request_id=request_id,
            content=content,
            future=future,
            priority=priority
        )
        
        try:
            # Add to queue
            if priority > 0:
                # Add to priority queue, which will be checked first
                self.priority_queue.append(request)
            else:
                # Add to regular queue
                await self.queue.put(request)
                
            self.total_requests += 1
            
            # Wait for the future to be resolved
            return await future
        except Exception as e:
            # Ensure future is cancelled if something goes wrong
            if not future.done():
                future.cancel()
            self.failed_requests += 1
            logger.error(f"Error processing request {request_id}: {e}")
            raise
    
    async def _process_batches(self):
        """Process batches of requests continuously."""
        try:
            batch_count = 0
            
            while not self.shutdown_event.is_set():
                # First check if we have priority requests
                if self.priority_queue:
                    # Process all priority requests first
                    while self.priority_queue and len(self.priority_queue) <= self.max_batch_size:
                        batch = self.priority_queue.copy()
                        self.priority_queue.clear()
                        
                        # Process this priority batch
                        await self._process_batch(batch)
                        batch_count += 1
                
                # Check if regular queue has items
                if self.queue.empty():
                    # No items, wait a bit and check again
                    await asyncio.sleep(0.01)
                    continue
                
                # Get the first request
                first_request = await self.queue.get()
                batch = [first_request]
                
                # Try to collect more requests up to max_batch_size or max_wait_time
                batch_timeout = first_request.arrival_time + self.max_wait_time
                remaining_time = max(0, batch_timeout - time.time())
                
                # Collect more requests if available
                while (
                    len(batch) < self.max_batch_size and 
                    remaining_time > 0 and 
                    not self.shutdown_event.is_set()
                ):
                    try:
                        # Wait for the next request with a timeout
                        next_request = await asyncio.wait_for(
                            self.queue.get(), 
                            timeout=remaining_time
                        )
                        batch.append(next_request)
                        
                        # Update remaining time
                        remaining_time = max(0, batch_timeout - time.time())
                    except asyncio.TimeoutError:
                        # No more requests within the wait time
                        break
                
                # Process the batch
                await self._process_batch(batch)
                
                # Report stats periodically
                batch_count += 1
                if batch_count % self.stats_interval == 0:
                    self._report_stats()
                    
        except asyncio.CancelledError:
            # Handle cancellation by failing all pending requests
            logger.info("Batch processor cancelled")
            
            # Fail all priority requests
            for request in self.priority_queue:
                if not request.future.done():
                    request.future.set_exception(RuntimeError("Batch processor shutting down"))
            self.priority_queue.clear()
            
            # Fail all regular queue requests
            while not self.queue.empty():
                try:
                    request = self.queue.get_nowait()
                    if not request.future.done():
                        request.future.set_exception(RuntimeError("Batch processor shutting down"))
                    self.queue.task_done()
                except asyncio.QueueEmpty:
                    break
            
            raise
        
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Error in batch processor: {e}")
            
            # Try to restart if not shutting down
            if not self.shutdown_event.is_set():
                logger.info("Restarting batch processor")
                self.processing_task = asyncio.create_task(self._process_batches())
    
    async def _process_batch(self, batch: List[BatchRequest]):
        """
        Process a batch of requests.
        
        Args:
            batch: List of BatchRequest objects
        """
        if not batch:
            return
            
        start_time = time.time()
        batch_size = len(batch)
        
        try:
            # Extract content from requests
            batch_content = [request.content for request in batch]
            
            # Process the batch
            results = await self.batch_processor(batch_content)
            
            # Set results for each request
            for i, request in enumerate(batch):
                if i < len(results):
                    # Set result if available
                    if not request.future.done():
                        request.future.set_result(results[i])
                    request.processed = True
                    self.successful_requests += 1
                else:
                    # No result for this request
                    if not request.future.done():
                        request.future.set_exception(
                            RuntimeError("No result returned for request")
                        )
                    self.failed_requests += 1
                    
        except Exception as e:
            # Handle batch processing errors
            logger.error(f"Error processing batch: {e}")
            
            # Set exception for all requests
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)
                self.failed_requests += 1
                
        finally:
            # Mark all queue tasks as done
            for _ in range(len(batch)):
                try:
                    # Only mark as done if from regular queue
                    # (priority queue items don't need this)
                    if not any(r.priority > 0 for r in batch):
                        self.queue.task_done()
                except Exception:
                    # Ignore errors in task_done
                    pass
            
            # Update stats
            elapsed = time.time() - start_time
            self.total_processing_time += elapsed
            self.total_batches += 1
            
            # Log batch completion
            logger.debug(
                f"Processed batch of {batch_size} requests in {elapsed:.4f}s "
                f"({batch_size/elapsed:.1f} requests/s)"
            )
    
    def _report_stats(self, final: bool = False):
        """Report statistics about the batcher."""
        if self.start_time is None:
            return
            
        uptime = time.time() - self.start_time
        avg_batch_size = self.total_requests / self.total_batches if self.total_batches > 0 else 0
        avg_processing_time = self.total_processing_time / self.total_batches if self.total_batches > 0 else 0
        throughput = self.total_requests / uptime if uptime > 0 else 0
        
        stats = {
            "uptime": f"{uptime:.1f}s",
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_batches": self.total_batches,
            "avg_batch_size": f"{avg_batch_size:.2f}",
            "avg_processing_time": f"{avg_processing_time*1000:.2f}ms",
            "throughput": f"{throughput:.2f} req/s",
            "queue_size": self.queue.qsize(),
            "priority_queue_size": len(self.priority_queue)
        }
        
        if final:
            logger.info(f"Final stats: {stats}")
        else:
            logger.info(f"Stats: {stats}")
            
        return stats

class TokenBucketRateLimiter:
    """Rate limiter using token bucket algorithm for controlling request rates."""
    def __init__(
        self, 
        tokens_per_second: float = 10.0, 
        bucket_size: int = 50
    ):
        """
        Initialize the rate limiter.
        
        Args:
            tokens_per_second: Refill rate (tokens per second)
            bucket_size: Maximum number of tokens in the bucket
        """
        self.tokens_per_second = tokens_per_second
        self.bucket_size = bucket_size
        self.tokens = bucket_size
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens were acquired, False otherwise
        """
        async with self.lock:
            # Refill tokens based on elapsed time
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(
                self.bucket_size,
                self.tokens + elapsed * self.tokens_per_second
            )
            self.last_refill = now
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            else:
                return False
    
    async def wait_for_tokens(self, tokens: int = 1, timeout: float = None) -> bool:
        """
        Wait until tokens are available.
        
        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait (seconds)
            
        Returns:
            True if tokens were acquired, False if timed out
        """
        start_time = time.time()
        
        while True:
            # Check if we have tokens available
            if await self.acquire(tokens):
                return True
            
            # Check timeout
            if timeout is not None:
                if time.time() - start_time >= timeout:
                    return False
            
            # Wait a bit before trying again
            # Adaptive wait time based on tokens needed and refill rate
            wait_time = min(
                tokens / self.tokens_per_second / 2,  # Half the expected time to refill
                0.1  # Maximum wait of 100ms
            )
            await asyncio.sleep(wait_time)
