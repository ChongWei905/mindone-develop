# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import mindspore
from mindspore import Tensor
import mindspore.ops as ops

from .feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from typing import Any, List, Optional, Sequence, Tuple, Union
from mindspore.nn import Metric
from mindspore.nn import Cell
import logging

logger = logging.getLogger("mindone.inception")


class InceptionScore(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    features: List
    inception: Cell
    feature_network: str = "inception"

    def __init__(
            self,
            # todo: support custom feature extractor
            feature: Union[str, int] = "logits_unbiased",
            deactivate_randperm: bool = False,
            splits: int = 10,
            normalize: bool = False,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        logger.warning(
            "Metric `InceptionScore` will save all extracted features in buffer."
            " For large datasets this may lead to large memory footprint."
        )

        # todo: find a way to get pre-trained inceptionV3 weights
        valid_inputs = ("logits_unbiased", 64, 192, 768, 2048)
        if feature not in valid_inputs:
            raise ValueError(
                f"Input to argument `feature` must be one of {valid_inputs}, but got {feature}."
            )

        data_type = self._dtype if hasattr(self, "_dtype") else mindspore.float32
        self.inception = FeatureExtractorInceptionV3(
            name="inecption-v3-compat",
            request_feature=str(feature),
            custom_dtype=data_type
        )

        if not isinstance(normalize, bool):
            raise ValueError("Argument `normalize` expected to be a bool")
        self.normalize = normalize

        self.splits = splits

        self.deactivate_randperm = deactivate_randperm

        self.features = []
        # self.add_state("features", [], dist_reduce_fx=None)

    def update(self, imgs: Tensor) -> None:
        imgs = (imgs * 255).byte() if self.normalize else imgs
        out = self.inception(imgs)
        features = out.reshape(imgs.shape[0], -1)
        self.features.append(features)

    def eval(self) -> Tuple[Tensor, Tensor]:
        """Compute metric."""
        if isinstance(self.features, mindspore.Tensor):
            features = self.features
        else:
            features = [y.unsqueeze(0) if y.numel() == 1 and y.ndim == 0 else y for y in self.features]
            if not features:  # empty list
                raise ValueError("No samples to concatenate")
            features = ops.cat(features, axis=0)

        # random permute the features
        if not self.deactivate_randperm:
            idx = ops.randperm(features.shape[0])
            features = features[idx]

        # calculate probs and logits
        prob = ops.softmax(features, axis=1)
        log_prob = ops.log_softmax(features, axis=1)

        # split into groups

        prob = ops.chunk(prob, chunks=self.splits, axis=0)
        log_prob = ops.chunk(log_prob, chunks=self.splits, axis=0)

        # calculate score per split
        mean_prob = [ops.mean(p, axis=0, keep_dims=True) for p in prob]
        kl_ = [p * (log_p - m_p.log()) for p, log_p, m_p in zip(prob, log_prob, mean_prob)]
        kl_ = [ops.sum(k, dim=1).mean().exp() for k in kl_]
        kl = ops.stack(kl_)

        # return mean and std
        return kl.mean(), kl.std()

    def clear(self):
        # todo: finish clear
        return
