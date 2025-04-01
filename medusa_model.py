import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple, AsyncGenerator
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, GenerationConfig
from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper
from transformers import TextIteratorStreamer
from threading import Thread
from huggingface_hub import snapshot_download, hf_hub_download
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MedusaModel")

class MedusaHead(nn.Module):
    """Medusa prediction head for speculative decoding."""
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size, bias=False)
        with torch.no_grad():
            self.linear.weight.data.normal_(mean=0.0, std=0.02)
    def forward(self, hidden_states):
        return self.linear(hidden_states)

class MedusaModel:
    def __init__(
        self,
        model_name_or_path: str,
        medusa_choices: List[int] = [3, 3, 3],
        tree_depth: int = 3,
        temperature: float = 0.0,
        precision: str = "fp16",
        max_context_length: int = 2048,
        medusa_model_path: Optional[str] = None,
        medusa_hf_repo: Optional[str] = None,
        medusa_hf_subfolder: str = "",
        device: Optional[str] = None,
        max_gpu_memory: Optional[int] = None,
        ds_inference_kwargs: Optional[Dict] = None, # Keep allowing external kwargs, but filter later
        hf_token: Optional[str] = None
    ):
        self.start_time = time.time()
        logger.info(f"Initializing MedusaModel with {model_name_or_path}")
        self.model_name = model_name_or_path
        self.medusa_choices = medusa_choices
        self.tree_depth = min(tree_depth, len(medusa_choices))
        self.temperature = temperature
        self.max_context_length = max_context_length
        self.medusa_hf_repo = medusa_hf_repo
        self.hf_token = hf_token
        self.cuda_available = torch.cuda.is_available()
        self.device = device if device else ("cuda" if self.cuda_available else "cpu")
        logger.info(f"Using device: {self.device}")

        # Determine dtype based on precision AND device
        if self.device == "cpu":
            self.dtype = torch.float32
            logger.info(f"CPU detected, forcing dtype to float32 (ignoring precision='{precision}')")
        elif precision == "fp16": self.dtype = torch.float16
        elif precision == "bf16": self.dtype = torch.bfloat16
        # Note: DeepSpeed handles int8/int4 quantization internally if configured
        elif precision in ["int8", "int4"]: self.dtype = torch.int8 # Placeholder, DS handles actual type
        else: self.dtype = torch.float32
        logger.info(f"Using precision: {precision} (Actual dtype: {self.dtype})")

        if not os.path.exists(model_name_or_path) and "/" in model_name_or_path:
            logger.info(f"Downloading model {model_name_or_path}")
            model_name_or_path = snapshot_download(repo_id=model_name_or_path, token=hf_token)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, padding_side="left", trust_remote_code=True)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token; logger.info("Set pad_token to eos_token")

        logger.info("Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=self.dtype, trust_remote_code=True, low_cpu_mem_usage=True)
        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size

        # --- Corrected DeepSpeed Inference Config ---
        # Define valid keys for DeepSpeedInferenceConfig based on documentation/source
        # Common keys: dtype, replace_with_kernel_inject, enable_cuda_graph, tensor_parallel, mp_size, max_tokens, min_tokens, max_gpu_memory, injection_policy
        valid_ds_inference_keys = [
            "dtype", "replace_with_kernel_inject", "enable_cuda_graph",
            "tensor_parallel", "mp_size", "max_tokens", "min_tokens",
            "max_gpu_memory", "injection_policy"
        ]

        # Base config required by init_inference
        ds_config = {
            "dtype": self.dtype,
            "replace_with_kernel_inject": True,
            "enable_cuda_graph": False, # Often safer to start with False
            "tensor_parallel": {"tp_size": int(os.getenv("WORLD_SIZE", "1"))},
            # Injection policy is crucial for getting hidden states
            # Using a more specific policy if possible is better
            "injection_policy": {type(self.model): {'hidden_states': True}}
        }

        # Merge external config carefully, ensuring only valid keys are passed
        if ds_inference_kwargs:
             filtered_external_kwargs = {k: v for k, v in ds_inference_kwargs.items() if k in valid_ds_inference_keys}
             ds_config.update(filtered_external_kwargs)
             if len(filtered_external_kwargs) != len(ds_inference_kwargs):
                 logger.warning(f"Filtered out invalid keys from ds_inference_kwargs: {set(ds_inference_kwargs.keys()) - set(filtered_external_kwargs.keys())}")

        # Memory constraints (optional)
        if max_gpu_memory and self.device.startswith("cuda"):
            device_id = int(self.device.split(':')[-1]) if ":" in self.device else 0
            ds_config["max_gpu_memory"] = {device_id: f"{max_gpu_memory}GiB"}

        logger.info(f"Initializing DeepSpeed inference engine with config: {ds_config}")
        self.model.to(self.device) # Ensure model is on device before init
        # Pass the config dictionary using the 'config' argument
        self.ds_engine = deepspeed.init_inference(model=self.model, config=ds_config)
        # ---------------------------------------------

        logger.info("Verifying access to hidden states...")
        test_input = self.tokenizer("Hello", return_tensors="pt").to(self.device)
        with torch.no_grad():
            test_output = self.ds_engine(input_ids=test_input["input_ids"], attention_mask=test_input["attention_mask"], output_hidden_states=True)
            hidden_states_output = None
            if isinstance(test_output, tuple):
                 for item in test_output:
                      if isinstance(item, (list, tuple)) and item and isinstance(item[-1], torch.Tensor) and item[-1].dim() == 3:
                           hidden_states_output = item[-1]; break
            elif hasattr(test_output, "hidden_states"): hidden_states_output = test_output.hidden_states[-1]
            elif isinstance(test_output, dict) and "hidden_states" in test_output: hidden_states_output = test_output["hidden_states"][-1] # Check dict output

            if hidden_states_output is None: raise ValueError("Cannot access hidden states with current DeepSpeed configuration")
            logger.info(f"Hidden states accessible. Shape: {hidden_states_output.shape}")

        logger.info("Initializing Medusa heads...")
        self.num_medusa_heads = sum(self.medusa_choices[:self.tree_depth])
        self.medusa_heads = nn.ModuleList([MedusaHead(self.hidden_size, self.vocab_size) for _ in range(self.num_medusa_heads)]).to(self.device).to(self.dtype)

        if self.medusa_hf_repo:
            logger.info(f"Loading Medusa heads from Hugging Face repo: {self.medusa_hf_repo}")
            try:
                from load_medusa_from_hf import load_medusa_heads_from_huggingface
                self._init_medusa_heads_from_huggingface(repo_id=self.medusa_hf_repo, subfolder=medusa_hf_subfolder)
            except Exception as e: logger.error(f"Error loading Medusa heads from Hugging Face: {e}"); logger.warning("Falling back to local initialization."); self._init_medusa_heads(medusa_model_path)
        else: self._init_medusa_heads(medusa_model_path)

        logger.info("Warming up model...");
        try:
             _ = self.generate("Hello", max_tokens=1, use_speculative=False, _is_warmup=True)
             if self.num_medusa_heads > 0: _ = self.generate("Hello", max_tokens=1, use_speculative=True, _is_warmup=True)
        except Exception as e: logger.error(f"Warmup failed: {e}")
        logger.info(f"Model initialization complete. Time taken: {time.time() - self.start_time:.2f}s")

    def _init_medusa_heads(self, medusa_model_path: Optional[str] = None):
        # (Implementation remains the same)
        self.num_medusa_heads = sum(self.medusa_choices[:self.tree_depth])
        if len(self.medusa_heads) != self.num_medusa_heads:
             logger.info(f"Recreating Medusa heads: {self.num_medusa_heads} heads")
             self.medusa_heads = nn.ModuleList([MedusaHead(self.hidden_size, self.vocab_size) for _ in range(self.num_medusa_heads)]).to(self.device).to(self.dtype)
        if medusa_model_path and os.path.exists(medusa_model_path): logger.info(f"Loading pre-trained Medusa heads from {medusa_model_path}"); self.load_medusa_heads(medusa_model_path)
        else: logger.info("Initializing new Medusa heads (if not loaded from HF)")
        self.medusa_heads.eval()

    def _init_medusa_heads_from_huggingface(self, repo_id: str, subfolder: str = ""):
        # (Implementation remains the same)
        from load_medusa_from_hf import load_medusa_heads_from_huggingface, get_compatible_medusa_repos
        if repo_id.lower() == 'auto':
            compatible_repos = get_compatible_medusa_repos(self.model_name)
            if compatible_repos: repo_id = compatible_repos[0]; logger.info(f"Automatically selected Medusa repository: {repo_id}")
            else: raise ValueError("Could not find a compatible Medusa repository")
        medusa_weights = load_medusa_heads_from_huggingface(repo_id=repo_id, model_id=self.model_name, subfolder=subfolder, device=self.device, token=self.hf_token)
        metadata = medusa_weights.get("metadata", {}); state_dict = medusa_weights.get("state_dict", medusa_weights)
        if "medusa_choices" in metadata: self.medusa_choices = metadata["medusa_choices"]; logger.info(f"Updated medusa_choices to {self.medusa_choices}")
        if "tree_depth" in metadata: self.tree_depth = metadata["tree_depth"]; logger.info(f"Updated tree_depth to {self.tree_depth}")
        if "num_heads" in metadata: self.num_medusa_heads = metadata["num_heads"]; logger.info(f"Updated num_medusa_heads to {self.num_medusa_heads}")
        if not hasattr(self, 'medusa_heads') or len(self.medusa_heads) != self.num_medusa_heads:
             logger.info(f"Recreating Medusa heads with loaded config: {self.num_medusa_heads} heads")
             self.medusa_heads = nn.ModuleList([MedusaHead(self.hidden_size, self.vocab_size) for _ in range(self.num_medusa_heads)]).to(self.device).to(self.dtype)
        adapted_state_dict = {};
        if len(state_dict) == 1 and isinstance(list(state_dict.values())[0], dict) and "linear.weight" in list(state_dict.values())[0]:
            for i, (_, head_dict) in enumerate(state_dict.items()):
                if i < len(self.medusa_heads): adapted_state_dict[f"{i}.linear.weight"] = head_dict["linear.weight"]
        elif any("medusa_heads" in k for k in state_dict.keys()):
            import re
            for k, v in state_dict.items():
                if "linear.weight" in k:
                    idx_match = re.search(r'medusa_heads\.(\d+)', k);
                    if idx_match: idx = int(idx_match.group(1));
                    if idx < len(self.medusa_heads): adapted_state_dict[f"{idx}.linear.weight"] = v
        else:
            head_weights = [v for k, v in state_dict.items() if isinstance(v, torch.Tensor) and len(v.shape) == 2]
            for i, weight in enumerate(head_weights):
                if i < len(self.medusa_heads):
                    if weight.shape == (self.vocab_size, self.hidden_size): adapted_state_dict[f"{i}.linear.weight"] = weight
                    elif weight.shape == (self.hidden_size, self.vocab_size): adapted_state_dict[f"{i}.linear.weight"] = weight.t()
        try: self.medusa_heads.load_state_dict(adapted_state_dict); logger.info(f"Successfully loaded Medusa heads from {repo_id}")
        except Exception as e: logger.error(f"Error loading Medusa weights: {e}");
        try: self.medusa_heads.load_state_dict(adapted_state_dict, strict=False); logger.info(f"Loaded Medusa heads with strict=False")
        except Exception as e2: logger.error(f"Error loading Medusa weights with strict=False: {e2}"); raise ValueError(f"Failed to load Medusa heads from {repo_id}")
        self.medusa_heads.eval()

    @torch.no_grad()
    def _generate_medusa_candidates(self, hidden_states: torch.Tensor, temperature: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # (Implementation remains the same)
        self.medusa_heads.eval()
        last_hidden = hidden_states[:, -1, :]
        candidate_logits = torch.stack([head(last_hidden) for head in self.medusa_heads], dim=1)
        temp = temperature if temperature is not None else self.temperature
        if temp == 0: candidate_ids = torch.argmax(candidate_logits, dim=-1)
        else:
            probs = F.softmax(candidate_logits / temp, dim=-1)
            probs_flat = probs.view(-1, self.vocab_size)
            sampled_ids_flat = torch.multinomial(probs_flat, num_samples=1)
            candidate_ids = sampled_ids_flat.view(probs.shape[0], probs.shape[1])
        return candidate_logits, candidate_ids

    @torch.no_grad()
    def _verify_candidates_parallel(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                                   draft_token_ids_batch: List[List[int]], temperature: float,
                                   top_p: float, top_k: int) -> Tuple[List[List[int]], List[int]]:
        # (Implementation remains the same)
        self.model.eval()
        batch_size = input_ids.shape[0]
        if batch_size == 0: return [], []
        if batch_size != len(draft_token_ids_batch): raise ValueError("Batch size mismatch")
        indices_with_empty_drafts = {i for i, d in enumerate(draft_token_ids_batch) if not d}
        active_indices = [i for i, d in enumerate(draft_token_ids_batch) if d]
        verified_tokens_batch = [[] for _ in range(batch_size)]
        num_accepted_batch = [0] * batch_size
        if indices_with_empty_drafts:
             empty_draft_indices_list = list(indices_with_empty_drafts)
             if empty_draft_indices_list:
                 outputs_empty = self.ds_engine(input_ids=input_ids[empty_draft_indices_list], attention_mask=attention_mask[empty_draft_indices_list])
                 next_token_logits_empty = outputs_empty.logits[:, -1, :]
                 sampled_token_ids_empty = self._sample_logits(next_token_logits_empty, temperature, top_k, top_p)
                 for i, original_idx in enumerate(empty_draft_indices_list):
                     token_id = sampled_token_ids_empty[i].item()
                     verified_tokens_batch[original_idx] = [token_id] if token_id != self.tokenizer.eos_token_id else []
                     num_accepted_batch[original_idx] = 0
        if not active_indices: return verified_tokens_batch, num_accepted_batch
        active_prefix_ids = input_ids[active_indices]; active_prefix_attn_mask = attention_mask[active_indices]
        active_drafts = [draft_token_ids_batch[i] for i in active_indices]
        max_draft_len = max(len(d) for d in active_drafts) if active_drafts else 0
        if max_draft_len == 0: return verified_tokens_batch, num_accepted_batch
        padded_drafts, draft_attn_mask_list, original_draft_lens = [], [], []
        for draft in active_drafts:
            original_len = len(draft); original_draft_lens.append(original_len)
            padding_len = max_draft_len - original_len
            padded_drafts.append(draft + [self.tokenizer.pad_token_id] * padding_len)
            draft_attn_mask_list.append([1] * original_len + [0] * padding_len)
        draft_tensor = torch.tensor(padded_drafts, device=self.device, dtype=torch.long)
        draft_attn_mask = torch.tensor(draft_attn_mask_list, device=self.device, dtype=torch.long)
        verify_input_ids = torch.cat([active_prefix_ids, draft_tensor], dim=1)
        verify_attn_mask = torch.cat([active_prefix_attn_mask, draft_attn_mask], dim=1)
        outputs = self.ds_engine(input_ids=verify_input_ids, attention_mask=verify_attn_mask)
        verify_logits = outputs.logits[:, active_prefix_ids.shape[1]-1:, :]
        num_active = len(active_indices)
        logits_flat = verify_logits.view(-1, verify_logits.shape[-1])
        base_model_sampled_ids_flat = self._sample_logits(logits_flat, temperature, top_k, top_p)
        base_model_sampled_ids = base_model_sampled_ids_flat.view(num_active, max_draft_len + 1)
        matches = (draft_tensor == base_model_sampled_ids[:, :max_draft_len]) & draft_attn_mask.bool()
        verified_tokens_active, num_accepted_active = [], []
        for i in range(num_active):
            item_accepted_tokens, item_num_accepted = [], 0
            original_len = original_draft_lens[i]
            for j in range(original_len):
                if matches[i, j]:
                    token_to_add = draft_tensor[i, j].item()
                    if token_to_add != self.tokenizer.eos_token_id: item_accepted_tokens.append(token_to_add); item_num_accepted += 1
                    else: break
                else:
                    token_to_add = base_model_sampled_ids[i, j].item()
                    if token_to_add != self.tokenizer.eos_token_id: item_accepted_tokens.append(token_to_add)
                    break
            else: # All draft accepted
                 next_token_id = base_model_sampled_ids[i, original_len].item()
                 if next_token_id != self.tokenizer.eos_token_id: item_accepted_tokens.append(next_token_id)
            verified_tokens_active.append(item_accepted_tokens)
            num_accepted_active.append(item_num_accepted)
        for i, original_idx in enumerate(active_indices):
            verified_tokens_batch[original_idx] = verified_tokens_active[i]
            num_accepted_batch[original_idx] = num_accepted_active[i]
        return verified_tokens_batch, num_accepted_batch

    # Helper for sampling (similar to v1)
    def _sample_logits(self, logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
        """ Samples token IDs from logits tensor. """
        batch_size, vocab_size = logits.shape
        if batch_size == 0: return torch.empty((0, 1), dtype=torch.long, device=logits.device)
        if temperature == 0: return torch.argmax(logits, dim=-1, keepdim=True)
        logits = logits / temperature
        logits_flat = logits.view(-1, vocab_size)
        if top_k > 0: logits_flat = TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=1)(None, logits_flat)
        if top_p < 1.0: logits_flat = TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1)(None, logits_flat)
        probs_flat = F.softmax(logits_flat, dim=-1)
        probs_flat = torch.nan_to_num(probs_flat, nan=0.0)
        probs_flat = probs_flat / probs_flat.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        sampled_ids_flat = torch.multinomial(probs_flat, num_samples=1)
        return sampled_ids_flat.view(batch_size, 1)

    # --- Main Generate Method (Corrected) ---
    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_tokens: int = 128,
        temperature: Optional[float] = None,
        top_p: float = 0.9,
        top_k: int = 0,
        stop: Optional[List[str]] = None,
        echo: bool = False, # Not implemented
        return_full_text: bool = False, # Not implemented
        use_speculative: bool = True,
        _is_warmup: bool = False # Internal flag to prevent recursion during warmup
    ) -> Dict[str, Any]:
        """
        Generate text using Medusa-enhanced speculative decoding (if use_speculative=True)
        or standard DeepSpeed/HF generation. Handles batch input using parallel verification.
        """
        if _is_warmup:
             if isinstance(prompt, list): prompt = prompt[0]
             inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
             _ = self.ds_engine(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
             return {}

        start_time = time.time()
        temp = temperature if temperature is not None else self.temperature

        if isinstance(prompt, str): prompts = [prompt]; is_single_prompt = True
        else: prompts = prompt; is_single_prompt = False
        batch_size = len(prompts)
        if batch_size == 0: return {}

        logger.info(f"Generating for {batch_size} prompts (speculative={use_speculative})")

        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt", truncation=True, max_length=self.max_context_length - max_tokens).to(self.device)
        input_ids = inputs["input_ids"]; attention_mask = inputs["attention_mask"]
        prompt_lens = attention_mask.sum(dim=1).tolist()

        if not use_speculative:
            logger.info("Using standard DeepSpeed/HF generation...")
            try:
                gen_config = GenerationConfig(
                    max_new_tokens=max_tokens, temperature=temp if temp > 0 else None, top_p=top_p if temp > 0 else None,
                    top_k=top_k if temp > 0 else None, do_sample=temp > 0,
                    pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
                )
                gen_outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=gen_config)
            except Exception as e: logger.error(f"Standard generation failed: {e}", exc_info=True); return {"error": f"Standard generation failed: {e}"}
            generated_ids_batch = [out[prompt_lens[i]:].tolist() for i, out in enumerate(gen_outputs)]
            total_verification_steps = 0; total_draft_tokens = 0; total_accepted_tokens = 0
        else:
            self.model.eval(); self.medusa_heads.eval()
            generated_ids_batch = [[] for _ in range(batch_size)]
            is_done = [False] * batch_size; num_done = 0; max_new_tokens = max_tokens
            total_draft_tokens = 0; total_accepted_tokens = 0; total_verification_steps = 0
            current_input_ids = input_ids; current_attention_mask = attention_mask
            current_total_len = current_input_ids.shape[1]

            while num_done < batch_size:
                all_max_len_reached = all(len(generated_ids_batch[i]) >= max_new_tokens for i in range(batch_size))
                if all_max_len_reached: break
                if total_verification_steps >= max_new_tokens * 1.5: logger.warning("Speculative safety break"); break

                active_indices = torch.tensor([i for i, done in enumerate(is_done) if not done], device=self.device)
                if len(active_indices) == 0: break

                active_input_ids = current_input_ids[active_indices, :current_total_len]
                active_attn_mask = current_attention_mask[active_indices, :current_total_len]

                outputs = self.ds_engine(input_ids=active_input_ids, attention_mask=active_attn_mask, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]

                _, candidate_ids = self._generate_medusa_candidates(hidden_states, temperature=0.6)
                best_drafts_active = [[ids[0].item()] if ids.numel() > 0 else [] for ids in candidate_ids[:, 0:1]]
                total_draft_tokens += sum(len(d) for d in best_drafts_active)

                verified_tokens_active, num_accepted_active = self._verify_candidates_parallel(
                    active_input_ids, active_attn_mask, best_drafts_active,
                    temperature=temp, top_p=top_p, top_k=top_k
                )
                total_verification_steps += 1
                total_accepted_tokens += sum(num_accepted_active)

                needs_repadding = False; max_len_after_step = current_total_len
                temp_is_done = list(is_done)
                for i, original_idx in enumerate(active_indices.tolist()):
                    if temp_is_done[original_idx]: continue
                    verified_ids = verified_tokens_active[i]
                    current_generated_len = len(generated_ids_batch[original_idx])
                    for token_id in verified_ids:
                        if current_generated_len < max_new_tokens: generated_ids_batch[original_idx].append(token_id); current_generated_len += 1
                        else: temp_is_done[original_idx] = True; break
                    if not temp_is_done[original_idx] and stop:
                         check_ids = generated_ids_batch[original_idx][max(0, current_generated_len - len(verified_ids) - 5):]
                         if check_ids: last_tokens_text = self.tokenizer.decode(check_ids, skip_special_tokens=True);
                         if any(s in last_tokens_text for s in stop): temp_is_done[original_idx] = True
                    current_seq_total_len = prompt_lens[original_idx] + len(generated_ids_batch[original_idx])
                    if current_seq_total_len > max_len_after_step: max_len_after_step = current_seq_total_len; needs_repadding = True
                is_done = temp_is_done; num_done = sum(is_done)

                if num_done < batch_size:
                    if max_len_after_step > current_total_len or needs_repadding:
                        new_input_ids_list, new_attn_mask_list = [], []
                        original_prompt_ids = inputs["input_ids"]
                        for i in range(batch_size):
                             prompt_part = original_prompt_ids[i, :prompt_lens[i]]
                             generated_part = torch.tensor(generated_ids_batch[i], device=self.device, dtype=torch.long)
                             full_seq = torch.cat([prompt_part, generated_part])
                             seq_len = full_seq.shape[0]; padding_len = max_len_after_step - seq_len
                             padded_seq = F.pad(full_seq, (0, max(0, padding_len)), value=self.tokenizer.pad_token_id)
                             new_input_ids_list.append(padded_seq)
                             attn_mask = torch.ones(seq_len, device=self.device, dtype=torch.long)
                             padded_attn_mask = F.pad(attn_mask, (0, max(0, padding_len)), value=0)
                             new_attn_mask_list.append(padded_attn_mask)
                        current_input_ids = torch.stack(new_input_ids_list)
                        current_attention_mask = torch.stack(new_attn_mask_list)
                        current_total_len = max_len_after_step

        output_texts = []; completion_tokens_list = []; finish_reasons = []
        for i in range(batch_size):
            final_ids = generated_ids_batch[i]
            output_texts.append(self.tokenizer.decode(final_ids, skip_special_tokens=True))
            completion_tokens_list.append(len(final_ids))
            finish_reasons.append("length" if len(final_ids) >= max_new_tokens else "stop")

        elapsed_time = time.time() - start_time
        total_completion_tokens = sum(completion_tokens_list)
        tokens_per_second = total_completion_tokens / elapsed_time if elapsed_time > 0 else 0

        medusa_stats = None
        if use_speculative:
             avg_acceptance_rate = total_accepted_tokens / max(1, total_draft_tokens) if total_draft_tokens > 0 else 0
             total_generated = sum(len(g) for g in generated_ids_batch)
             avg_speedup_factor = total_generated / max(1, total_verification_steps * batch_size) if total_verification_steps > 0 else 0
             medusa_stats = {
                 "total_verification_steps": total_verification_steps, "total_draft_tokens": total_draft_tokens,
                 "total_accepted_tokens": total_accepted_tokens, "avg_acceptance_rate": avg_acceptance_rate,
                 "avg_speedup_factor": avg_speedup_factor, "tokens_per_second": tokens_per_second,
                 "elapsed_time": elapsed_time
             }

        response = {
            "id": f"medusa-{int(time.time())}", "object": "text_completion", "created": int(time.time()),
            "model": f"medusa-{self.model_name}" if use_speculative else self.model_name,
            "choices": [{"text": output_texts[i], "index": i, "finish_reason": finish_reasons[i]} for i in range(batch_size)],
            "usage": {"prompt_tokens": sum(prompt_lens), "completion_tokens": total_completion_tokens, "total_tokens": sum(prompt_lens) + total_completion_tokens},
            "medusa_stats": medusa_stats
        }
        if is_single_prompt:
            response["choices"] = [response["choices"][0]]
            response["usage"]["prompt_tokens"] = prompt_lens[0]; response["usage"]["completion_tokens"] = completion_tokens_list[0]; response["usage"]["total_tokens"] = prompt_lens[0] + completion_tokens_list[0]
        return response

    # --- Streaming Generate Method ---
    @torch.no_grad()
    async def generate_stream(
        self,
        prompt: str, max_tokens: int = 256, temperature: float = 0.7,
        top_p: float = 0.95, top_k: int = 40, stop: Optional[List[str]] = None,
        use_speculative: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        logger.info(f"Starting streaming generation (speculative={use_speculative}).")
        temp = temperature

        if not use_speculative:
            logger.info("Using standard HF streaming fallback.")
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
            gen_config = GenerationConfig(
                max_new_tokens=max_tokens, temperature=temp if temp > 0 else None, top_p=top_p if temp > 0 else None,
                top_k=top_k if temp > 0 else None, do_sample=temp > 0,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
            )
            generation_kwargs = dict(inputs, streamer=streamer, generation_config=gen_config)
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            finish_reason = "stop"; generated_count = 0
            for new_text in streamer:
                yield {"text": new_text, "finish_reason": None}
                await asyncio.sleep(0.01)
                generated_count += 1
                if generated_count >= max_tokens: finish_reason = "length"; break
            thread.join()
            yield {"text": "", "finish_reason": finish_reason}
            logger.info(f"Finished standard HF streaming.")
            return

        if self.tokenizer.pad_token_id is None: self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]; attention_mask = inputs["attention_mask"]
        prompt_len = attention_mask.sum().item()

        self.model.eval(); self.medusa_heads.eval()
        generated_ids = []; total_generated_len = 0; max_new_tokens = max_tokens
        total_verification_steps = 0

        while total_generated_len < max_new_tokens:
            if total_verification_steps >= max_new_tokens * 1.5: break
            current_input_ids = input_ids; current_attn_mask = attention_mask
            outputs = self.ds_engine(input_ids=current_input_ids, attention_mask=current_attn_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            _, candidate_ids = self._generate_medusa_candidates(hidden_states, temperature=0.6)
            best_draft = [candidate_ids[0,0].item()] if candidate_ids.numel() > 0 else []
            verified_tokens_list, _ = self._verify_candidates_parallel(
                current_input_ids, current_attn_mask, [best_draft],
                temperature=temp, top_p=top_p, top_k=top_k
            )
            verified_ids = verified_tokens_list[0]; total_verification_steps += 1
            newly_generated_this_step = []; stop_sequence_found = False
            for token_id in verified_ids:
                if total_generated_len < max_new_tokens:
                    generated_ids.append(token_id); newly_generated_this_step.append(token_id); total_generated_len += 1
                    delta_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                    yield {"text": delta_text, "finish_reason": None}
                    await asyncio.sleep(0.01)
                    if stop:
                         full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                         if any(full_text.endswith(s) for s in stop): logger.debug(f"Stream finished (Stop sequence)."); stop_sequence_found = True; break
                else: logger.debug(f"Stream finished (Max length)."); stop_sequence_found = True; break
            if stop_sequence_found or total_generated_len >= max_new_tokens: break
            if not newly_generated_this_step: logger.warning("No new tokens generated in stream step, breaking."); break
            new_ids_tensor = torch.tensor([newly_generated_this_step], device=self.device, dtype=torch.long)
            input_ids = torch.cat([input_ids, new_ids_tensor], dim=1)
            attention_mask = F.pad(attention_mask, (0, len(newly_generated_this_step)), value=1)

        finish_reason = "stop" if total_generated_len < max_new_tokens else "length"
        yield {"text": "", "finish_reason": finish_reason}
        logger.info(f"Finished streaming speculative generation in {total_verification_steps} steps.")

    def save_medusa_heads(self, output_path: str):
        # (Implementation remains the same)
        state_dict = self.medusa_heads.state_dict()
        metadata = {"medusa_choices": self.medusa_choices, "tree_depth": self.tree_depth, "num_heads": self.num_medusa_heads, "vocab_size": self.vocab_size, "hidden_size": self.hidden_size, "model_name": self.model_name}
        torch.save({"state_dict": state_dict, "metadata": metadata}, output_path)
        logger.info(f"Saved Medusa heads to {output_path}")

    def load_medusa_heads(self, checkpoint_path: str):
        # (Implementation remains the same)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "state_dict" in checkpoint: self.medusa_heads.load_state_dict(checkpoint["state_dict"])
        else: self.medusa_heads.load_state_dict(checkpoint)
        if "metadata" in checkpoint:
            metadata = checkpoint["metadata"]; logger.info(f"Loaded Medusa heads metadata: {metadata}")
            if metadata.get("vocab_size") != self.vocab_size: logger.warning(f"Vocabulary size mismatch: loaded {metadata.get('vocab_size')}, current {self.vocab_size}")
        else: logger.info("No metadata found in checkpoint")
        logger.info(f"Loaded Medusa heads from {checkpoint_path}")
        self.medusa_heads.eval()
        return self

# For training Medusa heads
class MedusaTrainer:
    # ... (Trainer code remains largely the same) ...
    def __init__(
        self,
        medusa_model: MedusaModel,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0
    ):
        self.medusa_model = medusa_model
        self.device = medusa_model.device
        self.medusa_model.medusa_heads.train()
        self.optimizer = torch.optim.AdamW(self.medusa_model.medusa_heads.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.max_grad_norm = max_grad_norm
        self.global_step = 0
        self.warmup_steps = warmup_steps
        logger.info(f"Initialized MedusaTrainer with lr={learning_rate}, weight_decay={weight_decay}")

    def get_next_tokens(self, prompt: str, n_tokens: int) -> List[int]:
        inputs = self.medusa_model.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            try: outputs = self.medusa_model.ds_engine.generate(input_ids=inputs["input_ids"], max_new_tokens=n_tokens, do_sample=False)
            except AttributeError: outputs = self.medusa_model.model.generate(input_ids=inputs["input_ids"], max_new_tokens=n_tokens, do_sample=False)
            prompt_length = inputs["input_ids"].shape[1]
            next_tokens = outputs[0, prompt_length:].tolist()
            return next_tokens

    def train_on_prompt(self, prompt: str, max_positions: int = 10) -> float:
        full_inputs = self.medusa_model.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = full_inputs["input_ids"]; seq_length = input_ids.shape[1]
        if seq_length < 8: logger.warning(f"Prompt too short: {seq_length}"); return 0.0
        min_pos = max(2, seq_length // 4); max_pos = min(seq_length - 2, seq_length * 3 // 4)
        if max_pos <= min_pos: positions = [min_pos]
        else: num_positions = min(max_positions, max_pos - min_pos + 1); positions = np.random.choice(range(min_pos, max_pos + 1), size=num_positions, replace=False).tolist()
        total_loss = 0.0; num_training_steps = 0
        for position in positions:
            context_ids = input_ids[:, :position]
            with torch.no_grad():
                outputs = self.medusa_model.ds_engine(input_ids=context_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                target_length = min(self.medusa_model.num_medusa_heads, seq_length - position)
                if target_length <= 0: continue
                if position + target_length <= seq_length: target_tokens = input_ids[0, position:position+target_length].tolist()
                else: context_text = self.medusa_model.tokenizer.decode(context_ids[0], skip_special_tokens=True); target_tokens = self.get_next_tokens(context_text, target_length)
                if len(target_tokens) < self.medusa_model.num_medusa_heads: padding = [self.medusa_model.tokenizer.pad_token_id] * (self.medusa_model.num_medusa_heads - len(target_tokens)); target_tokens = target_tokens + padding
            last_hidden = hidden_states[:, -1:, :]
            head_idx = 0; step_loss = 0.0
            for level in range(self.medusa_model.tree_depth):
                if level >= len(self.medusa_model.medusa_choices): break
                for choice in range(self.medusa_model.medusa_choices[level]):
                    if head_idx >= self.medusa_model.num_medusa_heads: break
                    target_token = target_tokens[head_idx]; target_tensor = torch.tensor([target_token], device=self.device)
                    head = self.medusa_model.medusa_heads[head_idx]; logits = head(last_hidden).squeeze(1)
                    if target_token != self.medusa_model.tokenizer.pad_token_id:
                        loss = F.cross_entropy(logits, target_tensor)
                        step_loss += loss; self.optimizer.zero_grad(); loss.backward(retain_graph=True)
                        torch.nn.utils.clip_grad_norm_(head.parameters(), self.max_grad_norm); self.optimizer.step()
                    head_idx += 1
            if head_idx > 0:
                step_loss = step_loss / head_idx; total_loss += step_loss.item(); num_training_steps += 1
                if self.global_step < self.warmup_steps: lr_scale = min(1.0, float(self.global_step + 1) / self.warmup_steps);
                for param_group in self.optimizer.param_groups: param_group['lr'] *= lr_scale
                self.global_step += 1
        return total_loss / num_training_steps if num_training_steps > 0 else 0.0

    def train_on_dataset(self, dataset_path: str, num_epochs: int = 3, batch_size: int = 1, save_path: str = None, save_every: int = 1000, eval_every: int = 500, eval_prompts: List[str] = None) -> Dict[str, Any]:
        logger.info(f"Loading dataset from {dataset_path}"); prompts = []
        if dataset_path.endswith('.jsonl'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        text = item.get('text', '')
                        if text and len(text) > 10:
                            prompts.append(text)
                    except Exception as e:
                        logger.warning(f"Error parsing line: {e}")
        else:
            with open(dataset_path, 'r', encoding='utf-8') as f: text = f.read(); paragraphs = text.split('\n\n');
            for p in paragraphs: p = p.strip();
            if p and len(p) > 10: prompts.append(p)
        if not prompts: logger.error("No valid prompts found"); return {"error": "No valid prompts"}
        logger.info(f"Loaded {len(prompts)} prompts"); stats = {"epochs": [], "total_loss": 0.0, "steps": 0, "step_losses": [], "evaluations": []}
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch+1}/{num_epochs}"); epoch_loss = 0.0; np.random.shuffle(prompts)
            for i, prompt in enumerate(prompts):
                try:
                    loss = self.train_on_prompt(prompt); epoch_loss += loss; stats["total_loss"] += loss; stats["steps"] += 1
                    stats["step_losses"].append({"step": stats["steps"], "loss": loss})
                    if (i + 1) % 10 == 0: avg_loss = epoch_loss / (i + 1); logger.info(f"Epoch {epoch+1}, Step {i+1}/{len(prompts)}, Loss: {avg_loss:.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                    if eval_prompts and stats["steps"] % eval_every == 0: eval_results = self.evaluate(eval_prompts); stats["evaluations"].append({"step": stats["steps"], "results": eval_results}); logger.info(f"Evaluation at step {stats['steps']}: {eval_results}")
                    if save_path and stats["steps"] % save_every == 0: checkpoint_path = f"{save_path}_step{stats['steps']}.pt"; self.medusa_model.save_medusa_heads(checkpoint_path)
                except Exception as e: logger.error(f"Error training on prompt: {e}")
            epoch_avg_loss = epoch_loss / len(prompts); stats["epochs"].append({"epoch": epoch + 1, "avg_loss": epoch_avg_loss}); logger.info(f"Epoch {epoch+1} complete, Average Loss: {epoch_avg_loss:.4f}")
            if save_path: checkpoint_path = f"{save_path}_epoch{epoch+1}.pt"; self.medusa_model.save_medusa_heads(checkpoint_path)
        if save_path: self.medusa_model.save_medusa_heads(f"{save_path}_final.pt")
        return stats

    def evaluate(self, prompts: List[str]) -> Dict[str, Any]:
        self.medusa_model.medusa_heads.eval(); start_time = time.time(); tokens_generated = 0; medusa_tokens = 0
        # Evaluate needs to use the correct generate method
        for prompt in prompts:
             # Assuming generate handles single prompt correctly and returns expected structure
             result = self.medusa_model.generate(prompt, max_tokens=50, temperature=0.0, use_speculative=True) # Force speculative for eval
             if "usage" in result and "medusa_stats" in result and result["medusa_stats"] is not None: # Check medusa_stats is not None
                 tokens_generated += result["usage"].get("completion_tokens", 0)
                 # Ensure accepted_tokens exists before accessing
                 medusa_tokens += result["medusa_stats"].get("total_accepted_tokens", 0) # Use correct key from generate method
             else:
                 logger.warning(f"Unexpected result structure during evaluation: {result}")

        elapsed_time = time.time() - start_time; tokens_per_second = tokens_generated / elapsed_time if elapsed_time > 0 else 0; medusa_efficiency = medusa_tokens / tokens_generated if tokens_generated > 0 else 0
        self.medusa_model.medusa_heads.train()
        return {"tokens_per_second": tokens_per_second, "medusa_efficiency": medusa_efficiency, "total_tokens": tokens_generated, "medusa_tokens": medusa_tokens, "elapsed_time": elapsed_time}

if __name__ == "__main__":
    # Example usage - Requires a compatible base model and potentially trained Medusa heads
    # model = MedusaModel(model_name_or_path="lmsys/vicuna-7b-v1.5", precision="fp16") # Example
    # result = model.generate("Once upon a time in a distant land,", max_tokens=100)
    # print(result["choices"][0]["text"])
    # print(f"Efficiency: {result['medusa_stats']['medusa_efficiency']:.2f}")
    # print(f"Tokens/sec: {result['medusa_stats']['tokens_per_second']:.2f}")
    logger.info("MedusaModel example usage (commented out). Requires model setup.")
