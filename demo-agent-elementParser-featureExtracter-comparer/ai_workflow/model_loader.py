"""
model_loader.py
---------------
Reads the model catalogue from settings.py, inspects the current machine
(RAM, GPU, GUI), and loads the best local LLM that fits.

Main functions
    get_system_info()            -> RAM / GPU / GUI details of this machine
    list_models()                -> print/return every model and whether it fits
    select_best_model()          -> pick the largest model that fits
    load_model(model_key=None)   -> make sure the model exists, warm it up,
                                    and return a LoadedModel you can chat with

Run directly to check what your machine supports:
    python -m ai_workflow.model_loader
"""

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

import psutil

from settings import (
    APPLE_UNIFIED_MEMORY_RATIO,
    FALLBACK_MODEL_KEY,
    LLM_RUNTIME,
    LOCAL_LLM_MODELS,
    RAM_SAFETY_MARGIN_GB,
)

# `ollama` is imported lazily so that list_models() works even if the
# package is not installed yet.
try:
    import ollama
except ImportError:  # pragma: no cover
    ollama = None


# ---------------------------------------------------------------------------
# System detection
# ---------------------------------------------------------------------------
def _detect_gpu() -> dict:
    """
    Detect the GPU type and its usable memory.

    Returns a dict:  {"type": "cuda" | "mps" | "none", "name": str, "vram_gb": float}
      - cuda : NVIDIA GPU (detected through the `nvidia-smi` command)
      - mps  : Apple Silicon GPU (Metal), uses unified memory
      - none : no usable GPU, models will run on CPU
    """
    # --- NVIDIA GPU -------------------------------------------------------
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=5,
            ).strip().splitlines()
            # Take the first GPU. Output example: "NVIDIA RTX 4070, 12282"
            name, mem_mb = [x.strip() for x in out[0].split(",")]
            return {"type": "cuda", "name": name, "vram_gb": round(float(mem_mb) / 1024, 1)}
        except Exception:
            pass  # fall through to the next check

    # --- Apple Silicon (M1/M2/M3/M4...) ---------------------------------
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        total_gb = psutil.virtual_memory().total / 1024 ** 3
        return {
            "type": "mps",
            "name": "Apple Silicon (Metal)",
            # Only part of the unified memory can be used by the GPU.
            "vram_gb": round(total_gb * APPLE_UNIFIED_MEMORY_RATIO, 1),
        }

    # --- No GPU -----------------------------------------------------------
    return {"type": "none", "name": "None", "vram_gb": 0.0}


def _has_gui() -> bool:
    """True if the machine has a desktop environment (useful to enable the dashboard)."""
    system = platform.system()
    if system in ("Darwin", "Windows"):
        return True
    # Linux: a desktop session sets DISPLAY (X11) or WAYLAND_DISPLAY
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def get_system_info() -> dict:
    """Collect everything we need to decide which model this machine can run."""
    mem = psutil.virtual_memory()
    gpu = _detect_gpu()
    return {
        "os": platform.system(),
        "cpu_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(mem.total / 1024 ** 3, 1),
        "ram_available_gb": round(mem.available / 1024 ** 3, 1),
        "gpu_type": gpu["type"],
        "gpu_name": gpu["name"],
        "vram_gb": gpu["vram_gb"],
        "has_gui": _has_gui(),
    }


# ---------------------------------------------------------------------------
# Compatibility checks
# ---------------------------------------------------------------------------
def _check_compatibility(cfg: dict, info: dict) -> dict:
    """
    Decide whether one model fits the machine.

    Returns {"compatible": bool, "mode": "gpu" | "cpu" | None, "reason": str}
    GPU mode is preferred because it is much faster. CPU mode is the fallback.
    """
    # 1) Try GPU (NVIDIA VRAM or Apple unified memory)
    if info["gpu_type"] in ("cuda", "mps") and info["vram_gb"] >= cfg["min_vram_gb"]:
        return {"compatible": True, "mode": "gpu",
                "reason": f"fits GPU memory ({info['vram_gb']} GB >= {cfg['min_vram_gb']} GB)"}

    # 2) Try CPU, keeping a safety margin of RAM free for the rest of the system
    usable_ram = info["ram_total_gb"] - RAM_SAFETY_MARGIN_GB
    if usable_ram >= cfg["min_ram_gb"]:
        return {"compatible": True, "mode": "cpu",
                "reason": f"fits system RAM ({usable_ram:.1f} GB usable >= {cfg['min_ram_gb']} GB)"}

    # 3) Does not fit
    return {"compatible": False, "mode": None,
            "reason": f"needs {cfg['min_ram_gb']} GB RAM or {cfg['min_vram_gb']} GB GPU memory"}


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------
def list_models(show_incompatible: bool = True, print_output: bool = True) -> list:
    """
    List all models in settings.py with their compatibility on THIS machine.

    Args:
        show_incompatible: if False, only models that fit are listed.
        print_output:      if True, prints a formatted table.

    Returns:
        A list of dicts (one per model) so other code can use the result too.
    """
    info = get_system_info()
    rows = []

    for key, cfg in LOCAL_LLM_MODELS.items():
        check = _check_compatibility(cfg, info)
        if not check["compatible"] and not show_incompatible:
            continue
        rows.append({
            "key": key,
            "name": cfg["display_name"],
            "tag": cfg["ollama_tag"],
            "params_b": cfg["params_billion"],
            "tier": cfg["tier"],
            "compatible": check["compatible"],
            "mode": check["mode"],
            "reason": check["reason"],
        })

    if print_output:
        print("\n=== System ===")
        print(f"OS: {info['os']} | CPU cores: {info['cpu_cores']} | "
              f"RAM: {info['ram_total_gb']} GB (free {info['ram_available_gb']} GB)")
        print(f"GPU: {info['gpu_name']} ({info['gpu_type']}) | "
              f"GPU memory: {info['vram_gb']} GB | GUI: {info['has_gui']}")

        print("\n=== Models ===")
        print(f"{'KEY':<15}{'NAME':<22}{'SIZE':<7}{'FITS':<6}{'MODE':<6}REASON")
        for r in rows:
            print(f"{r['key']:<15}{r['name']:<22}{str(r['params_b']) + 'B':<7}"
                  f"{'yes' if r['compatible'] else 'no':<6}{(r['mode'] or '-'):<6}{r['reason']}")

        best = select_best_model(info)
        print(f"\nRecommended model: {best['key']} ({best['mode']} mode)\n")

    return rows


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def select_best_model(info: Optional[dict] = None) -> dict:
    """
    Pick the biggest (most capable) model that fits this machine.

    Returns {"key": str, "mode": "gpu" | "cpu"}.
    If nothing fits, the fallback model from settings.py is returned on CPU.
    """
    info = info or get_system_info()
    best_key, best_mode, best_size = None, None, -1

    for key, cfg in LOCAL_LLM_MODELS.items():
        check = _check_compatibility(cfg, info)
        if check["compatible"] and cfg["params_billion"] > best_size:
            best_key, best_mode, best_size = key, check["mode"], cfg["params_billion"]

    if best_key is None:
        return {"key": FALLBACK_MODEL_KEY, "mode": "cpu"}
    return {"key": best_key, "mode": best_mode}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@dataclass
class LoadedModel:
    """A ready-to-use model handle that the agents can call."""
    key: str
    tag: str
    mode: str                 # "gpu" or "cpu"
    client: "ollama.Client"

    def _options(self) -> dict:
        # num_gpu=0 forces CPU only. Otherwise Ollama chooses GPU automatically.
        return {"num_gpu": 0} if self.mode == "cpu" else {}

    def chat(self, messages: list, **kwargs) -> str:
        """
        Send chat messages and return the reply text.
        Example: model.chat([{"role": "user", "content": "Hello"}])
        """
        response = self.client.chat(
            model=self.tag,
            messages=messages,
            options={**self._options(), **kwargs.pop("options", {})},
            keep_alive=LLM_RUNTIME["keep_alive"],
            **kwargs,
        )
        return response["message"]["content"]


def _installed_tags(client) -> set:
    """Return the set of model tags already downloaded in Ollama."""
    resp = client.list()
    models = resp["models"] if isinstance(resp, dict) else resp.models
    tags = set()
    for m in models:
        # Older ollama versions use "name", newer ones use "model"
        tag = m.get("model") or m.get("name") if isinstance(m, dict) else getattr(m, "model", None)
        if tag:
            tags.add(tag)
    return tags


def load_model(model_key: Optional[str] = None) -> LoadedModel:
    """
    Load a local LLM.

    Args:
        model_key: key from LOCAL_LLM_MODELS. If None, the best model for this
                   machine is chosen automatically.

    Steps:
        1. Pick the model (auto or manual) and check it fits the machine.
        2. Connect to the Ollama server.
        3. Download the model if missing (when auto_pull is True).
        4. Warm up (load into memory) so the first real request is fast.
    """
    if ollama is None:
        raise RuntimeError("The 'ollama' package is missing. Run: pip install ollama")

    info = get_system_info()

    # 1) Choose the model
    if model_key is None:
        choice = select_best_model(info)
        model_key, mode = choice["key"], choice["mode"]
    else:
        if model_key not in LOCAL_LLM_MODELS:
            raise ValueError(f"Unknown model '{model_key}'. Available: {list(LOCAL_LLM_MODELS)}")
        check = _check_compatibility(LOCAL_LLM_MODELS[model_key], info)
        if not check["compatible"]:
            raise RuntimeError(f"Model '{model_key}' does not fit this machine: {check['reason']}")
        mode = check["mode"]

    tag = LOCAL_LLM_MODELS[model_key]["ollama_tag"]

    # 2) Connect to Ollama
    client = ollama.Client(host=LLM_RUNTIME["host"])
    try:
        installed = _installed_tags(client)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {LLM_RUNTIME['host']}. "
            f"Is it installed and running? (`ollama serve`) -> {exc}"
        )

    # 3) Download if needed
    if tag not in installed:
        if not LLM_RUNTIME["auto_pull"]:
            raise RuntimeError(f"Model '{tag}' is not downloaded. Run: ollama pull {tag}")
        print(f"Downloading {tag} (first time only)...")
        client.pull(tag)

    # 4) Warm up: an empty prompt just loads the model into memory
    client.generate(
        model=tag,
        prompt="",
        options={"num_gpu": 0} if mode == "cpu" else {},
        keep_alive=LLM_RUNTIME["keep_alive"],
    )

    print(f"Loaded {model_key} ({tag}) in {mode.upper()} mode")
    return LoadedModel(key=model_key, tag=tag, mode=mode, client=client)


# ---------------------------------------------------------------------------
# Quick manual check:  python -m ai_workflow.model_loader
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    list_models()