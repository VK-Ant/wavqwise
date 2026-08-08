"""
WavqWise Runtime Engine
========================
Auto-detects and uses the best available hardware:
  1. TensorRT (NVIDIA GPU - fastest)
  2. ONNX Runtime GPU (NVIDIA/AMD/DirectML)
  3. ONNX Runtime CPU (optimized CPU inference)
  4. PyTorch CUDA (NVIDIA GPU)
  5. PyTorch MPS (Apple Silicon)
  6. CPU fallback (always works)

Usage:
    from wavqwise.runtime.engine import RuntimeEngine

    engine = RuntimeEngine()         # Auto-detect best runtime
    engine = RuntimeEngine("cuda")   # Force CUDA
    engine = RuntimeEngine("onnx")   # Force ONNX Runtime

    print(engine.device)             # "cuda:0", "onnx-gpu", "cpu"
    print(engine.summary())          # Full hardware report
"""

import logging
import os
import platform
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RuntimeBackend(Enum):
    TENSORRT = "tensorrt"
    ONNX_GPU = "onnx-gpu"
    ONNX_CPU = "onnx-cpu"
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


@dataclass
class DeviceInfo:
    backend: RuntimeBackend
    device_name: str
    memory_gb: Optional[float] = None
    compute_capability: Optional[str] = None
    provider: Optional[str] = None


class RuntimeEngine:
    """Auto-detect and manage compute runtime.

    Priority order:
    1. TensorRT (if available + NVIDIA GPU)
    2. ONNX Runtime GPU (CUDAExecutionProvider / TensorrtExecutionProvider)
    3. PyTorch CUDA
    4. PyTorch MPS (Apple Silicon)
    5. ONNX Runtime CPU
    6. NumPy/CPU fallback

    The engine is a singleton per pipeline - initialize once, reuse everywhere.
    """

    _instance = None

    def __init__(self, preferred: Optional[str] = None, device_id: int = 0):
        self._preferred = preferred
        self._device_id = device_id
        self._backend: Optional[RuntimeBackend] = None
        self._device_info: Optional[DeviceInfo] = None
        self._torch_device = None
        self._onnx_session_options = None
        self._onnx_providers = None

        self._detect()

    def _detect(self):
        """Auto-detect best available runtime."""
        available = self._scan_available()

        if self._preferred:
            # User forced a specific runtime
            preferred_map = {
                "tensorrt": RuntimeBackend.TENSORRT,
                "trt": RuntimeBackend.TENSORRT,
                "onnx": RuntimeBackend.ONNX_GPU,
                "onnx-gpu": RuntimeBackend.ONNX_GPU,
                "onnx-cpu": RuntimeBackend.ONNX_CPU,
                "cuda": RuntimeBackend.CUDA,
                "gpu": RuntimeBackend.CUDA,
                "mps": RuntimeBackend.MPS,
                "cpu": RuntimeBackend.CPU,
            }
            requested = preferred_map.get(self._preferred.lower())
            if requested and requested in available:
                self._backend = requested
            else:
                logger.warning(
                    f"Requested runtime '{self._preferred}' not available. "
                    f"Available: {[b.value for b in available]}. Falling back to auto."
                )
                self._backend = self._pick_best(available)
        else:
            self._backend = self._pick_best(available)

        self._device_info = self._get_device_info()
        self._setup_runtime()

        logger.info(f"Runtime: {self._backend.value} | {self._device_info.device_name}")

    def _scan_available(self) -> List[RuntimeBackend]:
        """Scan what's installed and available."""
        available = [RuntimeBackend.CPU]  # Always available

        # Check PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                available.append(RuntimeBackend.CUDA)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                available.append(RuntimeBackend.MPS)
        except ImportError:
            pass

        # Check ONNX Runtime
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()

            if "TensorrtExecutionProvider" in providers:
                available.append(RuntimeBackend.TENSORRT)
            if "CUDAExecutionProvider" in providers:
                available.append(RuntimeBackend.ONNX_GPU)
            if "DmlExecutionProvider" in providers:
                available.append(RuntimeBackend.ONNX_GPU)  # DirectML (AMD/Intel on Windows)

            available.append(RuntimeBackend.ONNX_CPU)
        except ImportError:
            pass

        # Check TensorRT standalone
        if RuntimeBackend.TENSORRT not in available:
            try:
                import tensorrt
                available.append(RuntimeBackend.TENSORRT)
            except ImportError:
                pass

        return available

    def _pick_best(self, available: List[RuntimeBackend]) -> RuntimeBackend:
        """Pick the fastest available runtime."""
        priority = [
            RuntimeBackend.TENSORRT,
            RuntimeBackend.ONNX_GPU,
            RuntimeBackend.CUDA,
            RuntimeBackend.MPS,
            RuntimeBackend.ONNX_CPU,
            RuntimeBackend.CPU,
        ]
        for backend in priority:
            if backend in available:
                return backend
        return RuntimeBackend.CPU

    def _get_device_info(self) -> DeviceInfo:
        """Get hardware details for the selected backend."""
        if self._backend in (RuntimeBackend.CUDA, RuntimeBackend.TENSORRT):
            try:
                import torch
                name = torch.cuda.get_device_name(self._device_id)
                mem = torch.cuda.get_device_properties(self._device_id).total_mem / (1024**3)
                cap = torch.cuda.get_device_capability(self._device_id)
                return DeviceInfo(
                    backend=self._backend,
                    device_name=name,
                    memory_gb=round(mem, 1),
                    compute_capability=f"{cap[0]}.{cap[1]}",
                )
            except Exception:
                pass

        if self._backend == RuntimeBackend.ONNX_GPU:
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in providers else providers[0]
                return DeviceInfo(
                    backend=self._backend,
                    device_name=f"ONNX Runtime ({provider})",
                    provider=provider,
                )
            except Exception:
                pass

        if self._backend == RuntimeBackend.MPS:
            return DeviceInfo(
                backend=self._backend,
                device_name=f"Apple Silicon ({platform.processor() or 'M-series'})",
            )

        if self._backend == RuntimeBackend.ONNX_CPU:
            return DeviceInfo(
                backend=self._backend,
                device_name=f"ONNX Runtime CPU ({platform.processor() or 'x86_64'})",
                provider="CPUExecutionProvider",
            )

        return DeviceInfo(
            backend=RuntimeBackend.CPU,
            device_name=f"CPU ({platform.processor() or 'unknown'})",
        )

    def _setup_runtime(self):
        """Initialize runtime-specific settings."""
        if self._backend in (RuntimeBackend.CUDA, RuntimeBackend.TENSORRT):
            try:
                import torch
                self._torch_device = torch.device(f"cuda:{self._device_id}")
                # Enable TF32 for Ampere+
                if torch.cuda.get_device_capability(self._device_id)[0] >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass

        if self._backend == RuntimeBackend.MPS:
            try:
                import torch
                self._torch_device = torch.device("mps")
            except Exception:
                pass

        if self._backend in (RuntimeBackend.ONNX_GPU, RuntimeBackend.ONNX_CPU, RuntimeBackend.TENSORRT):
            try:
                import onnxruntime as ort
                self._onnx_session_options = ort.SessionOptions()
                self._onnx_session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._onnx_session_options.intra_op_num_threads = os.cpu_count()

                if self._backend == RuntimeBackend.TENSORRT:
                    self._onnx_providers = [
                        ("TensorrtExecutionProvider", {
                            "device_id": self._device_id,
                            "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,
                            "trt_fp16_enable": True,
                        }),
                        "CUDAExecutionProvider",
                        "CPUExecutionProvider",
                    ]
                elif self._backend == RuntimeBackend.ONNX_GPU:
                    self._onnx_providers = [
                        ("CUDAExecutionProvider", {"device_id": self._device_id}),
                        "CPUExecutionProvider",
                    ]
                else:
                    self._onnx_providers = ["CPUExecutionProvider"]
            except Exception:
                pass

    # === Public API ===

    @property
    def device(self) -> str:
        """Device string for display/logging."""
        if self._backend == RuntimeBackend.CUDA:
            return f"cuda:{self._device_id}"
        return self._backend.value

    @property
    def backend(self) -> RuntimeBackend:
        return self._backend

    @property
    def is_gpu(self) -> bool:
        return self._backend in (
            RuntimeBackend.CUDA, RuntimeBackend.MPS,
            RuntimeBackend.ONNX_GPU, RuntimeBackend.TENSORRT,
        )

    @property
    def torch_device(self):
        """PyTorch device object. Falls back to CPU."""
        if self._torch_device:
            return self._torch_device
        try:
            import torch
            return torch.device("cpu")
        except ImportError:
            return None

    def get_onnx_session(self, model_path: str):
        """Create an ONNX Runtime InferenceSession with optimal providers."""
        try:
            import onnxruntime as ort
            return ort.InferenceSession(
                model_path,
                sess_options=self._onnx_session_options,
                providers=self._onnx_providers or ["CPUExecutionProvider"],
            )
        except ImportError:
            raise ImportError("pip install onnxruntime-gpu  # or onnxruntime for CPU")

    def to_device(self, tensor):
        """Move a PyTorch tensor to the active device."""
        if self._torch_device and hasattr(tensor, "to"):
            return tensor.to(self._torch_device)
        return tensor

    def summary(self) -> str:
        """Full hardware/runtime report."""
        lines = [
            "WavqWise Runtime Engine",
            "=" * 40,
            f"Backend:  {self._backend.value}",
            f"Device:   {self._device_info.device_name}",
            f"GPU:      {'Yes' if self.is_gpu else 'No'}",
        ]
        if self._device_info.memory_gb:
            lines.append(f"VRAM:     {self._device_info.memory_gb} GB")
        if self._device_info.compute_capability:
            lines.append(f"Compute:  SM {self._device_info.compute_capability}")
        if self._device_info.provider:
            lines.append(f"Provider: {self._device_info.provider}")

        # Available runtimes
        available = self._scan_available()
        lines.append(f"Available: {', '.join(b.value for b in available)}")

        return "\n".join(lines)

    def __repr__(self):
        return f"RuntimeEngine(backend={self._backend.value}, device={self._device_info.device_name})"
