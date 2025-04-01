"""
Module for loading pre-trained Medusa heads from Hugging Face.
"""
import os
import torch
import logging
from huggingface_hub import snapshot_download, hf_hub_download
from typing import Dict, List, Optional, Union

logger = logging.getLogger("MedusaLoader")

def load_medusa_heads_from_huggingface(
    repo_id: str,
    model_id: Optional[str] = None,
    subfolder: str = "",
    medusa_key: str = "medusa_head",
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None
) -> Dict:
    """
    Load pre-trained Medusa heads from Hugging Face Hub.
    
    Args:
        repo_id: Hugging Face Hub repository ID containing Medusa weights
        model_id: Base model ID (for finding compatible heads)
        subfolder: Subfolder within repo containing Medusa weights
        medusa_key: Key for identifying medusa weights in model files
        device: Device to load weights on ('cuda', 'cpu', etc.)
        cache_dir: Directory to cache downloaded files
        token: Hugging Face API token for private repositories
        
    Returns:
        Dictionary containing Medusa weights and configuration
    """
    logger.info(f"Loading Medusa heads from {repo_id}")
    
    # Determine repository structure
    try:
        # Try to get file info from the repository
        if not subfolder:
            file_paths = []
            # Check for common filenames
            possible_files = [
                "medusa_head.bin", "medusa_heads.bin", 
                "medusa_head.pt", "medusa_heads.pt",
                "medusa_lm_head.bin", "medusa_lm_head.pt",
                "medusa_config.json"
            ]
            
            for filename in possible_files:
                try:
                    file_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder,
                        cache_dir=cache_dir,
                        token=token
                    )
                    file_paths.append(file_path)
                except:
                    # File not found, try next one
                    continue
            
            if not file_paths:
                # No specific files found, download the whole repository
                logger.info("No specific Medusa files found, downloading whole repository")
                repo_path = snapshot_download(
                    repo_id=repo_id,
                    cache_dir=cache_dir,
                    token=token
                )
                
                # Look for Medusa-related files in the repository
                for root, _, files in os.walk(repo_path):
                    for filename in files:
                        if "medusa" in filename.lower() and (filename.endswith(".bin") or filename.endswith(".pt")):
                            file_paths.append(os.path.join(root, filename))
            
            if not file_paths:
                raise ValueError(f"No Medusa-related files found in repository {repo_id}")
            
            logger.info(f"Found Medusa files: {file_paths}")
        else:
            # Use the specified subfolder
            repo_path = snapshot_download(
                repo_id=repo_id,
                subfolder=subfolder,
                cache_dir=cache_dir,
                token=token
            )
            
            # Look for Medusa-related files in the subfolder
            file_paths = []
            for root, _, files in os.walk(repo_path):
                for filename in files:
                    if "medusa" in filename.lower() and (filename.endswith(".bin") or filename.endswith(".pt")):
                        file_paths.append(os.path.join(root, filename))
            
            if not file_paths:
                raise ValueError(f"No Medusa-related files found in subfolder {subfolder} of repository {repo_id}")
        
        # Determine which files to load
        medusa_weights_file = None
        medusa_config_file = None
        
        for file_path in file_paths:
            if file_path.endswith(".bin") or file_path.endswith(".pt"):
                if "config" not in file_path.lower():
                    medusa_weights_file = file_path
            elif file_path.endswith(".json") and "config" in file_path.lower():
                medusa_config_file = file_path
        
        # Load the weights
        if medusa_weights_file:
            logger.info(f"Loading Medusa weights from {medusa_weights_file}")
            weights = torch.load(medusa_weights_file, map_location="cpu")
            
            # Check if we need to extract the Medusa weights from a larger state dict
            if isinstance(weights, dict) and not any("medusa" in k.lower() for k in weights.keys()):
                # Look for Medusa heads in the weights
                medusa_state_dict = {}
                
                # Check common patterns for Medusa weights
                for pattern in [medusa_key, "medusa_head", "medusa.head", "medusa.heads", "medusa_projection"]:
                    matched_keys = [k for k in weights.keys() if pattern in k.lower()]
                    if matched_keys:
                        # Extract Medusa weights
                        for k in matched_keys:
                            medusa_state_dict[k] = weights[k]
                
                if not medusa_state_dict:
                    # If we couldn't find Medusa weights using patterns, assume this is the direct Medusa state dict
                    medusa_state_dict = weights
                
                weights = {"state_dict": medusa_state_dict}
            
            # Load config if available
            if medusa_config_file:
                import json
                logger.info(f"Loading Medusa config from {medusa_config_file}")
                with open(medusa_config_file, "r") as f:
                    config = json.load(f)
                weights["metadata"] = config
            
            # Ensure we have a proper metadata structure
            if "metadata" not in weights:
                # Try to infer metadata from weights structure
                metadata = infer_medusa_metadata(weights, model_id)
                weights["metadata"] = metadata
            
            if device:
                # Move weights to the specified device
                for k, v in weights.get("state_dict", {}).items():
                    weights["state_dict"][k] = v.to(device)
            
            return weights
        else:
            raise ValueError(f"No Medusa weights file found in repository {repo_id}")
            
    except Exception as e:
        logger.error(f"Error loading Medusa heads from Hugging Face: {e}")
        raise

def infer_medusa_metadata(weights: Dict, model_id: Optional[str] = None) -> Dict:
    """
    Infer Medusa configuration metadata from weights structure.
    
    Args:
        weights: Loaded weights dictionary
        model_id: Base model ID for compatibility
        
    Returns:
        Inferred metadata dictionary
    """
    metadata = {}
    
    # Get weights dictionary
    state_dict = weights.get("state_dict", weights)
    
    # Infer number of heads from weight keys
    head_keys = [k for k in state_dict.keys() if "head" in k.lower()]
    if not head_keys:
        # If no "head" in keys, just count the number of weight tensors that look like projection layers
        projection_keys = [
            k for k in state_dict.keys() 
            if len(state_dict[k].shape) == 2  # Matrix shape for projection
            and "weight" in k.lower()
        ]
        num_heads = len(projection_keys)
    else:
        # Try to extract head numbers from keys
        import re
        head_numbers = []
        for key in head_keys:
            matches = re.findall(r'head_(\d+)|heads\.(\d+)|projection\.(\d+)', key.lower())
            if matches:
                # Flatten and filter the matches
                numbers = [int(num) for match in matches for num in match if num]
                if numbers:
                    head_numbers.extend(numbers)
        
        if head_numbers:
            # Add 1 because head indices are typically 0-based
            num_heads = max(head_numbers) + 1
        else:
            # If we can't find head numbers, count the unique "base" key names
            base_keys = set()
            for key in head_keys:
                base_key = re.sub(r'\.\d+\.', '.', key)  # Replace .0., .1., etc with .
                base_keys.add(base_key)
            num_heads = len(base_keys)
    
    # Infer other metadata
    metadata["num_heads"] = num_heads
    
    # Infer vocab size and hidden size from weights shape
    projection_shapes = [
        state_dict[k].shape 
        for k in state_dict.keys() 
        if len(state_dict[k].shape) == 2
    ]
    
    if projection_shapes:
        # Most projection weights should have shape (vocab_size, hidden_size) or (hidden_size, vocab_size)
        hidden_size = min(projection_shapes[0])
        vocab_size = max(projection_shapes[0])
        
        metadata["hidden_size"] = hidden_size
        metadata["vocab_size"] = vocab_size
    
    # Infer tree structure (guessing based on common configurations)
    if num_heads <= 2:
        metadata["medusa_choices"] = [num_heads]
        metadata["tree_depth"] = 1
    elif num_heads <= 5:
        metadata["medusa_choices"] = [2, 3]
        metadata["tree_depth"] = 2
    elif num_heads <= 9:
        metadata["medusa_choices"] = [3, 3, 3]
        metadata["tree_depth"] = 3
    else:
        # For more heads, distribute them evenly
        import math
        tree_depth = min(3, math.ceil(math.log2(num_heads + 1)))
        choices = [math.ceil(num_heads / tree_depth)] * tree_depth
        metadata["medusa_choices"] = choices
        metadata["tree_depth"] = tree_depth
    
    # Set model name if provided
    if model_id:
        metadata["model_name"] = model_id
    
    return metadata

def get_compatible_medusa_repos(model_name: str) -> List[str]:
    """
    Get a list of Hugging Face repositories with compatible Medusa heads for a model.
    
    Args:
        model_name: Base model name (e.g., "lmsys/vicuna-7b-v1.5")
        
    Returns:
        List of compatible repository IDs
    """
    # Extract model family and size from name
    model_name_lower = model_name.lower()
    
    # Map model names to potential compatible Medusa repositories
    compatibility_map = {
        "vicuna-7b": [
            "microsoft/Medusa-Vicuna-7B-v1.3",
            "liuhaotian/LLaVA-Medusa-Vicuna-7B-v1.3"
        ],
        "llama-2-7b": [
            "TIGER-Lab/MAmmoTH-Medusa-Llama2-7B",
            "LoneStriker/Llama-2-7b-chat-hf-Medusa"
        ],
        "llama-3-8b": [
            "cjpais/llama-3-8b-instruct-medusa-1",
            "neuralmagic/llama-3-8b-medusa"
        ],
        "mistral-7b": [
            "LoneStriker/Medusa-Mistral-7B-v0.1", 
            "cognitivecomputations/mistral-7b-medusa-alpha"
        ],
        "tinyllama": [
            "TinyLlama/Medusa-1.1B"
        ],
        "phi-2": [
            "microsoft/Medusa-Phi-2"
        ]
    }
    
    compatible_repos = []
    
    # Check each model family
    for family, repos in compatibility_map.items():
        if family in model_name_lower:
            compatible_repos.extend(repos)
    
    if not compatible_repos:
        # Return some generic options if no specific match
        return [
            "microsoft/Medusa-Vicuna-7B-v1.3",  # Most widely compatible
            "LoneStriker/Medusa-Mistral-7B-v0.1",  # Good alternative
            "TinyLlama/Medusa-1.1B"  # Lightweight option
        ]
    
    return compatible_repos
