"""
ONNX Export & Inference
========================
Export trained WavqWise models to ONNX for fast inference.
Supports TensorRT optimization on NVIDIA GPUs.

Usage:
    from wavqwise.runtime.onnx_inference import ONNXExporter, ONNXPredictor

    # Export trained sklearn/torch model
    exporter = ONNXExporter()
    exporter.export_sklearn(trained_model, "model.onnx", n_features=13)

    # Fast inference
    predictor = ONNXPredictor("model.onnx")
    result = predictor.predict(input_array)
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ONNXExporter:
    """Export trained models to ONNX format."""

    def export_sklearn(self, model, output_path: str, n_features: int,
                       opset_version: int = 13):
        """Export sklearn model to ONNX."""
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType

            initial_type = [("input", FloatTensorType([None, n_features]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type,
                                          target_opset=opset_version)

            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())

            logger.info(f"Exported sklearn model to {output_path}")
            return output_path

        except ImportError:
            raise ImportError("pip install skl2onnx")

    def export_pytorch(self, model, output_path: str, input_shape: tuple,
                       opset_version: int = 13):
        """Export PyTorch model to ONNX."""
        try:
            import torch

            model.eval()
            dummy_input = torch.randn(*input_shape)

            torch.onnx.export(
                model, dummy_input, output_path,
                opset_version=opset_version,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
            )

            logger.info(f"Exported PyTorch model to {output_path}")
            return output_path

        except ImportError:
            raise ImportError("pip install torch")

    def optimize_for_tensorrt(self, onnx_path: str, output_path: Optional[str] = None,
                               fp16: bool = True, max_workspace_gb: int = 2):
        """Optimize ONNX model with TensorRT."""
        try:
            import tensorrt as trt

            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, TRT_LOGGER)

            with open(onnx_path, "rb") as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        logger.error(f"TensorRT parse error: {parser.get_error(i)}")
                    raise RuntimeError("TensorRT ONNX parsing failed")

            config = builder.create_builder_config()
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE,
                max_workspace_gb * (1 << 30)
            )

            if fp16:
                config.set_flag(trt.BuilderFlag.FP16)

            engine = builder.build_serialized_network(network, config)
            out = output_path or onnx_path.replace(".onnx", ".trt")

            with open(out, "wb") as f:
                f.write(engine)

            logger.info(f"TensorRT engine saved to {out} (FP16={fp16})")
            return out

        except ImportError:
            raise ImportError("pip install tensorrt")


class ONNXPredictor:
    """Fast inference using ONNX Runtime with auto GPU detection."""

    def __init__(self, model_path: str, device: Optional[str] = None):
        from wavqwise.runtime.engine import RuntimeEngine

        self._engine = RuntimeEngine(preferred=device)
        self._session = self._engine.get_onnx_session(model_path)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        logger.info(
            f"ONNX Predictor: {model_path} on {self._engine.device}"
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference. Input: numpy array. Output: numpy array."""
        if X.dtype != np.float32:
            X = X.astype(np.float32)

        result = self._session.run(
            [self._output_name],
            {self._input_name: X},
        )
        return result[0]

    def predict_batch(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Batched inference for large datasets."""
        results = []
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size]
            results.append(self.predict(batch))
        return np.concatenate(results, axis=0)

    @property
    def device(self) -> str:
        return self._engine.device

    def benchmark(self, X: np.ndarray, n_runs: int = 100) -> Dict[str, float]:
        """Benchmark inference speed."""
        import time

        if X.dtype != np.float32:
            X = X.astype(np.float32)

        # Warmup
        for _ in range(10):
            self.predict(X)

        # Benchmark
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            self.predict(X)
            times.append(time.perf_counter() - start)

        times = np.array(times) * 1000  # ms
        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "p50_ms": float(np.percentile(times, 50)),
            "p99_ms": float(np.percentile(times, 99)),
            "throughput_per_sec": float(len(X) / (np.mean(times) / 1000)),
            "device": self.device,
        }
