# Copyright 2024 The HuggingFace Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import pytest
import torch
from ddt import data, ddt, unpack

import mindspore as ms

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    PipelineTesterMixin,
    get_module,
    get_pipeline_components,
)

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
    {"mode": ms.GRAPH_MODE, "dtype": "float32"},
    {"mode": ms.GRAPH_MODE, "dtype": "float16"},
]


@ddt
class LTXPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.models.transformers.transformer_ltx.LTXVideoTransformer3DModel",
            "mindone.diffusers.models.transformers.transformer_ltx.LTXVideoTransformer3DModel",
            dict(
                in_channels=8,
                out_channels=8,
                patch_size=1,
                patch_size_t=1,
                num_attention_heads=4,
                attention_head_dim=8,
                cross_attention_dim=32,
                num_layers=1,
                caption_channels=32,
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl_ltx.AutoencoderKLLTXVideo",
            "mindone.diffusers.models.autoencoders.autoencoder_kl_ltx.AutoencoderKLLTXVideo",
            dict(
                in_channels=3,
                out_channels=3,
                latent_channels=8,
                block_out_channels=(8, 8, 8, 8),
                decoder_block_out_channels=(8, 8, 8, 8),
                layers_per_block=(1, 1, 1, 1, 1),
                decoder_layers_per_block=(1, 1, 1, 1, 1),
                spatio_temporal_scaling=(True, True, False, False),
                decoder_spatio_temporal_scaling=(True, True, False, False),
                decoder_inject_noise=(False, False, False, False, False),
                upsample_residual=(False, False, False, False),
                upsample_factor=(1, 1, 1, 1),
                timestep_conditioning=False,
                patch_size=1,
                patch_size_t=1,
                encoder_causal=True,
                decoder_causal=False,
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler",
            "mindone.diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler",
            dict(),
        ],
        [
            "text_encoder",
            "transformers.models.t5.modeling_t5.T5EncoderModel",
            "mindone.transformers.models.t5.modeling_t5.T5EncoderModel",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5",
                revision="refs/pr/1",
            ),
        ],
        [
            "tokenizer",
            "transformers.models.auto.tokenization_auto.AutoTokenizer",
            "transformers.models.auto.tokenization_auto.AutoTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5",
            ),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "transformer",
                "vae",
                "scheduler",
                "text_encoder",
                "tokenizer",
            ]
        }

        pt_components, ms_components = get_pipeline_components(components, self.pipeline_config)

        pt_components["vae"].use_framewise_encoding = False
        ms_components["vae"].use_framewise_encoding = False
        pt_components["vae"].use_framewise_decoding = False
        ms_components["vae"].use_framewise_decoding = False

        return pt_components, ms_components

    def get_dummy_inputs(self, seed=0):
        inputs = {
            "prompt": "dance monkey",
            "negative_prompt": "",
            "num_inference_steps": 2,
            "guidance_scale": 3.0,
            "height": 32,
            "width": 32,
            # 8 * k + 1 is the recommendation
            "num_frames": 9,
            "max_sequence_length": 16,
            "output_type": "np",
        }

        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        if dtype == "float32":
            pytest.skip(" FP32 is not supported since HunyuanVideoPipeline contains nn.Conv3d")

        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.ltx.LTXPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.ltx.LTXPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_image = pt_pipe(**inputs)
        torch.manual_seed(0)
        ms_image = ms_pipe(**inputs)

        pt_image_slice = pt_image.frames[0, -3:, -3:, -1]
        ms_image_slice = ms_image[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.max(np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice)) < threshold
