import os
import numpy as np
import torch
import logging
from voice_changer.embedder.Embedder import Embedder

logger = logging.getLogger(__name__)


class CoreMLEmbedder(Embedder):

    def load_model(self, file: str) -> Embedder:
        import coremltools as ct

        mlpackage_path = os.path.splitext(file)[0] + '.mlpackage'

        if not os.path.exists(mlpackage_path) or \
                os.path.getmtime(file) > os.path.getmtime(mlpackage_path):
            logger.info('Converting embedder ONNX → CoreML (one-time, ~30s)...')
            mlmodel = ct.convert(
                file,
                minimum_deployment_target=ct.target.macOS13,
                compute_units=ct.ComputeUnit.ALL,
            )
            mlmodel.save(mlpackage_path)
            logger.info(f'CoreML model cached at {mlpackage_path}')

        self.mlmodel = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.ALL)
        # Discover actual output names in case coremltools renamed them
        spec = self.mlmodel.get_spec()
        self._output_names = [o.name for o in spec.description.output]
        logger.info(f'CoreML embedder outputs: {self._output_names}')

        self.fp_dtype_t = torch.float32
        super().set_props(self.embedderType, file)
        return self

    def extract_features(
        self, feats: torch.Tensor, embOutputLayer=9, useFinalProj=True
    ) -> torch.Tensor:
        audio_np = feats.detach().cpu().numpy().astype(np.float32)
        result = self.mlmodel.predict({'audio': audio_np})

        # Prefer known output names; fall back to position if names changed after conversion
        if embOutputLayer == 9:
            output = result.get('units9', result[self._output_names[0]])
        else:
            idx = min(1, len(self._output_names) - 1)
            output = result.get('unit12', result[self._output_names[idx]])

        return torch.as_tensor(output, dtype=self.fp_dtype_t, device=feats.device)
