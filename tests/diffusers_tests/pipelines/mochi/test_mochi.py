import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack

import mindspore as ms

from mindone.diffusers.utils.testing_utils import export_to_gif, load_numpy, slow

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    THRESHOLD_PIXEL,
    PipelineTesterMixin,
    get_module,
    get_pipeline_components,
)

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
    {"mode": ms.GRAPH_MODE, "dtype": "float16"},
]


@ddt
class MochiPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.models.transformers.transformer_mochi.MochiTransformer3DModel",
            "mindone.diffusers.models.transformers.transformer_mochi.MochiTransformer3DModel",
            dict(
                patch_size=2,
                num_attention_heads=2,
                attention_head_dim=8,
                num_layers=2,
                pooled_projection_dim=16,
                in_channels=12,
                out_channels=None,
                qk_norm="rms_norm",
                text_embed_dim=32,
                time_embed_dim=4,
                activation_fn="swiglu",
                max_sequence_length=16,
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl_mochi.AutoencoderKLMochi",
            "mindone.diffusers.models.autoencoders.autoencoder_kl_mochi.AutoencoderKLMochi",
            dict(
                latent_channels=12,
                out_channels=3,
                encoder_block_out_channels=(32, 32, 32, 32),
                decoder_block_out_channels=(32, 32, 32, 32),
                layers_per_block=(1, 1, 1, 1, 1),
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
            dict(pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5", revision="refs/pr/1"),
        ],
        [
            "tokenizer",
            "transformers.models.auto.tokenization_auto.AutoTokenizer",
            "transformers.models.auto.tokenization_auto.AutoTokenizer",
            dict(pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5"),
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

        return pt_components, ms_components

    def get_dummy_inputs(self, seed=0):
        generator = torch.manual_seed(seed)

        inputs = {
            "prompt": "dance monkey",
            "negative_prompt": "",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 4.5,
            "height": 16,
            "width": 16,
            # 6 * k + 1 is the recommendation
            "num_frames": 7,
            "max_sequence_length": 16,
            "output_type": "np",
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.mochi.pipeline_mochi.MochiPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.mochi.pipeline_mochi.MochiPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        ms_video = ms_pipe(**inputs)[0]

        pt_generated_video = load_numpy(
            f"mochi_{dtype}_fasttest.npy",
        )
        ms_generated_video = ms_video[0]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_generated_video - ms_generated_video) / np.linalg.norm(pt_generated_video) < threshold


@slow
@ddt
class MochiPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_mochi(sself, mode, dtype):
        ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)
        ms_dtype = getattr(ms, dtype)
        pipe_cls = get_module("mindone.diffusers.pipelines.mochi.pipeline_mochi.MochiPipeline")
        pipe = pipe_cls.from_pretrained("genmo/mochi-1-preview", mindspore_dtype=ms_dtype)

        prompt = "A painting of a squirrel eating a burger."

        torch.manual_seed(0)
        videos = pipe(
            prompt=prompt,
            height=480,
            width=848,
            num_frames=19,
            num_inference_steps=2,
        )[0]
        video = videos[0]
        export_to_gif(video, f"mochi_{dtype}_{mode}_generate.gif")

        expected_video = load_numpy(
            f"mochi_{dtype}.npy",
        )
        assert np.mean(np.abs(np.array(video, dtype=np.float32) - expected_video)) < THRESHOLD_PIXEL
