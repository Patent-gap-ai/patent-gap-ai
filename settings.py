"""
settings.py
-----------
Central configuration for the Patent Gap AI project.

This file holds:
  1. LLM_RUNTIME            -> how we talk to the local LLM server (Ollama)
  2. Memory tuning values   -> safety margins used when choosing a model
  3. LOCAL_LLM_MODELS       -> JSON-style dictionary of all supported local models

To add a new model, just add a new entry to LOCAL_LLM_MODELS. No code changes
are needed in model_loader.py.
"""

import os

# ---------------------------------------------------------------------------
# 1. Runtime configuration (Ollama is used to serve the local models)
# ---------------------------------------------------------------------------
LLM_RUNTIME = {
    "backend": "ollama",
    # Ollama server address. Override with the OLLAMA_HOST environment variable.
    "host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    # If a selected model is not downloaded yet, pull it automatically.
    "auto_pull": True,
    # How long Ollama keeps the model in memory after the last request.
    "keep_alive": "10m",
}

# ---------------------------------------------------------------------------
# 2. Memory tuning
# ---------------------------------------------------------------------------
# RAM (in GB) always kept free for the OS, the API server, MongoDB, etc.
RAM_SAFETY_MARGIN_GB = 1.5

# On Apple Silicon, CPU and GPU share one memory pool ("unified memory").
# macOS only lets the GPU use a part of it, so we treat ~65% of RAM as "VRAM".
APPLE_UNIFIED_MEMORY_RATIO = 0.65

# Smallest model, used if nothing else fits the machine.
FALLBACK_MODEL_KEY = "qwen2.5-0.5b"

# ---------------------------------------------------------------------------
# 3. Local LLM catalogue (JSON-style dictionary)
# ---------------------------------------------------------------------------
# Field meaning:
#   display_name   : human-friendly name shown in list output
#   ollama_tag     : exact model tag used by Ollama (`ollama pull <tag>`)
#   params_billion : model size in billions of parameters (higher = smarter)
#   size_gb        : approximate download / memory size (4-bit quantised)
#   min_ram_gb     : minimum system RAM needed to run on CPU
#   min_vram_gb    : minimum GPU memory needed to run fully on GPU
#   context_length : maximum context window in tokens
#   tier           : rough category (tiny / small / medium / large / xlarge)
#   description    : short note on what the model is good for
LOCAL_LLM_MODELS = {
    "qwen2.5-0.5b": {
        "display_name": "Qwen 2.5 0.5B",
        "ollama_tag": "qwen2.5:0.5b",
        "params_billion": 0.5,
        "size_gb": 0.4,
        "min_ram_gb": 4,
        "min_vram_gb": 1,
        "context_length": 32768,
        "tier": "tiny",
        "description": "Ultra light model for very low-spec machines.",
    },
    "qwen2.5-1.5b": {
        "display_name": "Qwen 2.5 1.5B",
        "ollama_tag": "qwen2.5:1.5b",
        "params_billion": 1.5,
        "size_gb": 1.0,
        "min_ram_gb": 4,
        "min_vram_gb": 2,
        "context_length": 32768,
        "tier": "tiny",
        "description": "Small and fast, basic validation tasks.",
    },
    "llama3.2-3b": {
        "display_name": "Llama 3.2 3B",
        "ollama_tag": "llama3.2:3b",
        "params_billion": 3,
        "size_gb": 2.0,
        "min_ram_gb": 8,
        "min_vram_gb": 3,
        "context_length": 131072,
        "tier": "small",
        "description": "Good balance of speed and quality on laptops.",
    },
    "phi3-mini": {
        "display_name": "Phi-3 Mini 3.8B",
        "ollama_tag": "phi3:mini",
        "params_billion": 3.8,
        "size_gb": 2.3,
        "min_ram_gb": 8,
        "min_vram_gb": 3,
        "context_length": 4096,
        "tier": "small",
        "description": "Strong reasoning for its size, short context.",
    },
    "mistral-7b": {
        "display_name": "Mistral 7B Instruct",
        "ollama_tag": "mistral:7b",
        "params_billion": 7,
        "size_gb": 4.1,
        "min_ram_gb": 8,
        "min_vram_gb": 5,
        "context_length": 32768,
        "tier": "medium",
        "description": "Reliable general model for JSON validation and analysis.",
    },
    "llama3.1-8b": {
        "display_name": "Llama 3.1 8B",
        "ollama_tag": "llama3.1:8b",
        "params_billion": 8,
        "size_gb": 4.9,
        "min_ram_gb": 16,
        "min_vram_gb": 6,
        "context_length": 131072,
        "tier": "medium",
        "description": "Long context, good for reading full patent text.",
    },
    "qwen2.5-14b": {
        "display_name": "Qwen 2.5 14B",
        "ollama_tag": "qwen2.5:14b",
        "params_billion": 14,
        "size_gb": 9.0,
        "min_ram_gb": 16,
        "min_vram_gb": 10,
        "context_length": 32768,
        "tier": "large",
        "description": "High quality analysis and structured output.",
    },
    "qwen2.5-32b": {
        "display_name": "Qwen 2.5 32B",
        "ollama_tag": "qwen2.5:32b",
        "params_billion": 32,
        "size_gb": 20.0,
        "min_ram_gb": 32,
        "min_vram_gb": 22,
        "context_length": 32768,
        "tier": "xlarge",
        "description": "Workstation-class model for deep patent analysis.",
    },
    "llama3.1-70b": {
        "display_name": "Llama 3.1 70B",
        "ollama_tag": "llama3.1:70b",
        "params_billion": 70,
        "size_gb": 40.0,
        "min_ram_gb": 64,
        "min_vram_gb": 44,
        "context_length": 131072,
        "tier": "xlarge",
        "description": "Server-class model, needs a lot of memory.",
    },
}