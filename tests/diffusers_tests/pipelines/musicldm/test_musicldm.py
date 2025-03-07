import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack

import mindspore as ms

import mindone.transformers.models.speecht5.modeling_speecht5
from mindone.diffusers.utils.testing_utils import load_downloaded_numpy_from_hf_hub, slow

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    THRESHOLD_PIXEL,
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
class MusicLDMPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "unet",
            "diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            "mindone.diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            dict(
                block_out_channels=(32, 64),
                layers_per_block=2,
                sample_size=32,
                in_channels=4,
                out_channels=4,
                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                cross_attention_dim=(32, 64),
                class_embed_type="simple_projection",
                projection_class_embeddings_input_dim=32,
                class_embeddings_concat=True,
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_ddim.DDIMScheduler",
            "mindone.diffusers.schedulers.scheduling_ddim.DDIMScheduler",
            dict(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            "mindone.diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            dict(
                block_out_channels=[32, 64],
                in_channels=1,
                out_channels=1,
                down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
                up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
                latent_channels=4,
            ),
        ],
        [
            "tokenizer",
            "transformers.models.roberta.tokenization_roberta.RobertaTokenizer",
            "mindone.transformers.models.roberta.tokenization_roberta.RobertaTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-roberta",
                model_max_length=77,
            ),
        ],
        [
            "feature_extractor",
            "transformers.models.clap.feature_extraction_clap.ClapFeatureExtractor",
            "mindone.transformers.models.clap.feature_extraction_clap.ClapFeatureExtractor",
            dict(pretrained_model_name_or_path="hf-internal-testing/tiny-random-ClapModel", hop_length=7900),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "unet",
                "scheduler",
                "vae",
                "tokenizer",
                "feature_extractor",
            ]
        }

        pt_components, ms_components = get_pipeline_components(components, self.pipeline_config)

        clap_text_config_cls = get_module("transformers.models.clap.configuration_clap.ClapTextConfig")
        text_branch_config = clap_text_config_cls(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=16,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=2,
            num_hidden_layers=2,
            pad_token_id=1,
            vocab_size=1000,
        )
        clap_audio_config_cls = get_module("transformers.models.clap.configuration_clap.ClapAudioConfig")
        audio_branch_config = clap_audio_config_cls(
            spec_size=64,
            window_size=4,
            num_mel_bins=64,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            depths=[2, 2],
            num_attention_heads=[2, 2],
            num_hidden_layers=2,
            hidden_size=192,
            patch_size=2,
            patch_stride=2,
            patch_embed_input_channels=4,
        )
        clap_config_cls = get_module("transformers.models.clap.configuration_clap.ClapConfig")
        text_encoder_config = clap_config_cls.from_text_audio_configs(
            text_config=text_branch_config, audio_config=audio_branch_config, projection_dim=32
        )
        pt_text_encoder_cls = get_module("transformers.models.clap.modeling_clap.ClapModel")
        pt_text_encoder = pt_text_encoder_cls(text_encoder_config)
        ms_text_encoder_cls = get_module("mindone.transformers.models.clap.modeling_clap.ClapModel")
        ms_text_encoder = ms_text_encoder_cls(text_encoder_config)

        vocoder_config_cls = get_module("transformers.models.speecht5.configuration_speecht5.SpeechT5HifiGanConfig")
        vocoder_config = vocoder_config_cls(
            model_in_dim=8,
            sampling_rate=16000,
            upsample_initial_channel=16,
            upsample_rates=[2, 2],
            upsample_kernel_sizes=[4, 4],
            resblock_kernel_sizes=[3, 7],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]],
            normalize_before=False,
        )
        pt_vocoder_cls = get_module("transformers.models.speecht5.modeling_speecht5.SpeechT5HifiGan")
        pt_vocoder = pt_vocoder_cls(vocoder_config)
        ms_vocoder_cls = get_module(mindone.transformers.models.speecht5.modeling_speecht5.SpeechT5HifiGan)
        ms_vocoder = ms_vocoder_cls(vocoder_config)

        pt_components["text_encoder"] = pt_text_encoder
        pt_components["vocoder"] = pt_vocoder
        ms_components["text_encoder"] = ms_text_encoder
        ms_components["vocoder"] = ms_vocoder

        return pt_components, ms_components

    def get_dummy_inputs(self, seed=0):
        generator = torch.manual_seed(seed)

        inputs = {
            "prompt": "A hammer hitting a wooden surface",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.musicldm.pipeline_musicldm.MusicLDMPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.musicldm.pipeline_musicldm.MusicLDMPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_audio = pt_pipe(**inputs).audios
        torch.manual_seed(0)
        ms_audio = ms_pipe(**inputs)[0]

        pt_generated_audio_slice = pt_audio[0][8680:8690]
        ms_generated_audio_slice = ms_audio[0][8680:8690]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert (
            np.linalg.norm(pt_generated_audio_slice - ms_generated_audio_slice)
            / np.linalg.norm(pt_generated_audio_slice)
            < threshold
        )


@slow
@ddt
class MusicLDMPipelineNightlyTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)
        pipe_cls = get_module("mindone.diffusers.pipelines.musicldm.pipeline_musicldm.MusicLDMPipeline")
        pipe = pipe_cls.from_pretrained("cvssp/musicldm", mindspore_dtype=ms_dtype)
        torch.manual_seed(0)
        latents = np.random.RandomState(0).standard_normal((1, 8, 128, 16))
        latents = ms.Tensor(latents).to(ms_dtype)
        prompt = "A hammer hitting a wooden surface"
        audios = pipe(
            prompt=prompt,
            latents=latents,
            num_inference_steps=3,
            guidance_scale=2.5,
        )[0]
        audio = audios[0]

        expected_audio = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"musicldm_{dtype}.npy",
            subfolder="mochi",
        )
        assert np.mean(np.abs(np.array(audio, dtype=np.float32) - expected_audio)) < THRESHOLD_PIXEL
