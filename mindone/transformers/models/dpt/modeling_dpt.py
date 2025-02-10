# coding=utf-8
# Copyright 2022 Intel Labs, OpenMMLab and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch DPT (Dense Prediction Transformers) model.

This implementation is heavily inspired by OpenMMLab's implementation, found here:
https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/decode_heads/dpt_head.py.

"""

import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union

from transformers.models.dpt.configuration_dpt import DPTConfig
from transformers.utils import ModelOutput, logging

import mindspore as ms
from mindspore import nn, ops, mint

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, DepthEstimatorOutput
from ...modeling_utils import MSPreTrainedModel
from ..bit import BitBackbone

logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "DPTConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "Intel/dpt-large"
_EXPECTED_OUTPUT_SHAPE = [1, 577, 1024]


@dataclass
class BaseModelOutputWithIntermediateActivations(ModelOutput):
    """
    Base class for model's outputs that also contains intermediate activations that can be used at later stages. Useful
    in the context of Vision models.:

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        intermediate_activations (`tuple(torch.FloatTensor)`, *optional*):
            Intermediate activations that can be used to compute hidden states of the model at various layers.
    """

    last_hidden_states: ms.Tensor = None
    intermediate_activations: Optional[Tuple[ms.Tensor, ...]] = None


@dataclass
class BaseModelOutputWithPoolingAndIntermediateActivations(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states as well as intermediate
    activations that can be used by the model at later stages.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        intermediate_activations (`tuple(torch.FloatTensor)`, *optional*):
            Intermediate activations that can be used to compute hidden states of the model at various layers.
    """

    last_hidden_state: ms.Tensor = None
    pooler_output: ms.Tensor = None
    hidden_states: Optional[Tuple[ms.Tensor, ...]] = None
    attentions: Optional[Tuple[ms.Tensor, ...]] = None
    intermediate_activations: Optional[Tuple[ms.Tensor, ...]] = None


class DPTViTHybridEmbeddings(nn.Cell):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config, feature_size=None):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])

        # TODO: AutoBackbone of transformers need to be implemented
        backbone_config = getattr(config, "backbone_config", None)
        if backbone_config.model_type == "bit":
            self.backbone = BitBackbone(backbone_config).set_train(False)
        else:
            raise NotImplementedError(f"Backbone is not supported except bit, but got {backbone_config.model_type}")
        feature_dim = self.backbone.channels[-1]
        if len(self.backbone.channels) != 3:
            raise ValueError(f"Expected backbone to have 3 output features, got {len(self.backbone.channels)}")
        self.residual_feature_map_index = [0, 1]  # Always take the output of the first and second backbone stage

        if feature_size is None:
            feat_map_shape = config.backbone_featmap_shape
            feature_size = feat_map_shape[-2:]
            feature_dim = feat_map_shape[1]
        else:
            feature_size = (
                feature_size if isinstance(feature_size, collections.abc.Iterable) else (feature_size, feature_size)
            )
            feature_dim = self.backbone.channels[-1]

        self.image_size = image_size
        self.patch_size = patch_size[0]
        self.num_channels = num_channels

        self.projection = mint.nn.Conv2d(feature_dim, hidden_size, kernel_size=1, bias=True)

        self.cls_token = ms.Parameter(mint.zeros((1, 1, config.hidden_size)))
        self.position_embeddings = ms.Parameter(mint.zeros((1, num_patches + 1, config.hidden_size)))
        self.posemb_shape_n = ms.Tensor(self.position_embeddings.shape[1], ms.float32)

    def _resize_pos_embed(self, posemb, posemb_shape, grid_size_height, grid_size_width, start_index=1):
        posemb_tok = posemb[:, :start_index]
        posemb_grid = posemb[0, start_index:]

        old_grid_size = int(mint.sqrt(posemb_shape - start_index))

        posemb_grid = mint.permute(mint.reshape(posemb_grid, (1, old_grid_size, old_grid_size, -1)), (0, 3, 1, 2))
        posemb_grid = ops.interpolate(posemb_grid, size=(grid_size_height, grid_size_width), mode="bilinear")
        posemb_grid = mint.reshape(mint.permute(posemb_grid, (0, 2, 3, 1)), (1, grid_size_height * grid_size_width, -1))

        posemb = mint.cat([posemb_tok, posemb_grid], dim=1)

        return posemb

    def construct(
        self, pixel_values: ms.Tensor, interpolate_pos_encoding: bool = False, return_dict: bool = False
    ) -> ms.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )

        position_embeddings = self._resize_pos_embed(
            self.position_embeddings, self.posemb_shape_n, height // self.patch_size, width // self.patch_size
        )

        backbone_output = self.backbone(pixel_values)

        features = backbone_output[0][-1]

        # Retrieve also the intermediate activations to use them at later stages
        output_hidden_states = [backbone_output[0][index] for index in self.residual_feature_map_index]

        embeddings = mint.swapaxes(mint.flatten(self.projection(features), start_dim=2), 1, 2)

        cls_tokens = mint.broadcast_to(self.cls_token, (batch_size, -1, -1))
        embeddings = mint.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + position_embeddings

        if not return_dict:
            return (embeddings, output_hidden_states)

        # Return hidden states and intermediate activations
        return BaseModelOutputWithIntermediateActivations(
            last_hidden_states=embeddings,
            intermediate_activations=output_hidden_states,
        )


class DPTViTEmbeddings(nn.Cell):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, config):
        super().__init__()

        self.cls_token = ms.Parameter(mint.zeros((1, 1, config.hidden_size)))
        self.patch_embeddings = DPTViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = ms.Parameter(mint.zeros((1, num_patches + 1, config.hidden_size)))
        self.dropout = mint.nn.Dropout(p=config.hidden_dropout_prob)
        self.config = config
        self.patch_size = self.config.patch_size

    def _resize_pos_embed(self, posemb, grid_size_height, grid_size_width, start_index=1):
        posemb_tok = posemb[:, :start_index]
        posemb_grid = posemb[0, start_index:]

        old_grid_size = int(posemb_grid.shape[0] ** 0.5)

        posemb_grid = mint.permute(mint.reshape(posemb_grid, (1, old_grid_size, old_grid_size, -1)), (0, 3, 1, 2))
        posemb_grid = ops.interpolate(posemb_grid, size=(grid_size_height, grid_size_width), mode="bilinear")
        posemb_grid = mint.reshape(mint.permute(posemb_grid, (0, 2, 3, 1)), (1, grid_size_height * grid_size_width, -1))

        posemb = mint.cat([posemb_tok, posemb_grid], dim=1)

        return posemb

    def construct(self, pixel_values, return_dict=False):
        batch_size, num_channels, height, width = pixel_values.shape

        # possibly interpolate position encodings to handle varying image sizes
        patch_size = self.patch_size
        position_embeddings = self._resize_pos_embed(
            self.position_embeddings, height // patch_size, width // patch_size
        )

        embeddings = self.patch_embeddings(pixel_values)

        batch_size, seq_len, _ = embeddings.shape

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = mint.broadcast_to(self.cls_token, (batch_size, -1, -1))
        embeddings = mint.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + position_embeddings

        embeddings = self.dropout(embeddings)

        if not return_dict:
            return (embeddings,)

        return BaseModelOutputWithIntermediateActivations(last_hidden_states=embeddings)


class DPTViTPatchEmbeddings(nn.Cell):
    """
    Image to Patch Embedding.

    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = mint.nn.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=True
        )

    def construct(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        embeddings = mint.swapaxes(mint.flatten(self.projection(pixel_values), start_dim=2), 1, 2)
        return embeddings


# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention with ViT->DPT
class DPTViTSelfAttention(nn.Cell):
    def __init__(self, config: DPTConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.attention_head_size_tensor = ms.Tensor(self.attention_head_size, ms.float32)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = mint.nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = mint.nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = mint.nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = mint.nn.Dropout(p=config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: ms.Tensor) -> ms.Tensor:
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return mint.permute(x, (0, 2, 1, 3))

    def construct(
        self, hidden_states, head_mask: Optional[ms.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[ms.Tensor, ms.Tensor], Tuple[ms.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = mint.matmul(query_layer, mint.swapaxes(key_layer, -1, -2))

        attention_scores = attention_scores / mint.sqrt(self.attention_head_size_tensor.to(attention_scores.dtype))

        # Normalize the attention scores to probabilities.
        attention_probs = mint.nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = mint.matmul(attention_probs, value_layer)

        context_layer = mint.permute(context_layer, (0, 2, 1, 3))
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->DPT
class DPTViTSelfOutput(nn.Cell):
    """
    The residual connection is defined in DPTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: DPTConfig) -> None:
        super().__init__()
        self.dense = mint.nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = mint.nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: ms.Tensor, input_tensor: ms.Tensor) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], ms.Tensor]:
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.

    Args:
        heads (`List[int]`): List of the indices of heads to prune.
        n_heads (`int`): The number of heads in the model.
        head_size (`int`): The size of each head.
        already_pruned_heads (`Set[int]`): A set of already pruned heads.

    Returns:
        `Tuple[Set[int], torch.LongTensor]`: A tuple with the indices of heads to prune taking `already_pruned_heads`
        into account and the indices of rows/columns to keep in the layer weight.
    """
    mask = mint.ones((n_heads, head_size))
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mint.eq(mask.view(-1), 1)
    index: ms.Tensor = mint.arange(len(mask))[mask].long()
    return heads, index


# TODO: fix
def prune_linear_layer(layer: mint.nn.Linear, index: ms.Tensor, dim: int = 0) -> mint.nn.Linear:
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`torch.nn.Linear`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        `torch.nn.Linear`: The pruned layer as a new layer with `requires_grad=True`.
    """
    W = mint.clone(mint.index_select(layer.weight, dim, index)).detach()
    if layer.bias is not None:
        if dim == 1:
            b = mint.clone(layer.bias).detach()
        else:
            b = mint.clone(layer.bias[index]).detach()
    new_size = list(layer.weight.shape)
    new_size[dim] = len(index)
    new_layer = mint.nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W)
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b)
        new_layer.bias.requires_grad = True
    return new_layer


class DPTViTAttention(nn.Cell):
    def __init__(self, config: DPTConfig) -> None:
        super().__init__()
        self.attention = DPTViTSelfAttention(config)
        self.output = DPTViTSelfOutput(config)
        self.pruned_heads = set()

    # Copied from transformers.models.vit.modeling_vit.ViTAttention.prune_heads
    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # Copied from transformers.models.vit.modeling_vit.ViTAttention.forward
    def construct(
        self,
        hidden_states: ms.Tensor,
        head_mask: Optional[ms.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[ms.Tensor, ms.Tensor], Tuple[ms.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTIntermediate with ViT->DPT
class DPTViTIntermediate(nn.Cell):
    def __init__(self, config: DPTConfig) -> None:
        super().__init__()
        self.dense = mint.nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTOutput with ViT->DPT
class DPTViTOutput(nn.Cell):
    def __init__(self, config: DPTConfig) -> None:
        super().__init__()
        self.dense = mint.nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = mint.nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: ms.Tensor, input_tensor: ms.Tensor) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


# copied from transformers.models.vit.modeling_vit.ViTLayer with ViTConfig->DPTConfig, ViTAttention->DPTViTAttention,
# ViTIntermediate->DPTViTIntermediate, ViTOutput->DPTViTOutput
class DPTViTLayer(nn.Cell):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: DPTConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = DPTViTAttention(config)
        self.intermediate = DPTViTIntermediate(config)
        self.output = DPTViTOutput(config)
        self.layernorm_before = mint.nn.LayerNorm((config.hidden_size,), eps=config.layer_norm_eps)
        self.layernorm_after = mint.nn.LayerNorm((config.hidden_size,), eps=config.layer_norm_eps)

    def construct(
        self,
        hidden_states: ms.Tensor,
        head_mask: Optional[ms.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[ms.Tensor, ms.Tensor], Tuple[ms.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


# copied from transformers.models.vit.modeling_vit.ViTEncoder with ViTConfig -> DPTConfig, ViTLayer->DPTViTLayer
class DPTViTEncoder(nn.Cell):
    def __init__(self, config: DPTConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.CellList([DPTViTLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def construct(
        self,
        hidden_states: ms.Tensor,
        head_mask: Optional[ms.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class DPTReassembleStage(nn.Cell):
    """
    This class reassembles the hidden states of the backbone into image-like feature representations at various
    resolutions.

    This happens in 3 stages:
    1. Map the N + 1 tokens to a set of N tokens, by taking into account the readout ([CLS]) token according to
       `config.readout_type`.
    2. Project the channel dimension of the hidden states according to `config.neck_hidden_sizes`.
    3. Resizing the spatial dimensions (height, width).

    Args:
        config (`[DPTConfig]`):
            Model configuration class defining the model architecture.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.readout_type = self.config.readout_type
        self.layers = nn.CellList()
        if config.is_hybrid:
            self._init_reassemble_dpt_hybrid(config)
        else:
            self._init_reassemble_dpt(config)

        self.neck_ignore_stages = config.neck_ignore_stages

    def _init_reassemble_dpt_hybrid(self, config):
        r""" "
        For DPT-Hybrid the first 2 reassemble layers are set to `mint.nn.Identity()`, please check the official
        implementation: https://github.com/isl-org/DPT/blob/f43ef9e08d70a752195028a51be5e1aff227b913/dpt/vit.py#L438
        for more details.
        """
        for i, factor in zip(range(len(config.neck_hidden_sizes)), config.reassemble_factors):
            if i <= 1:
                self.layers.append(mint.nn.Identity())
            elif i > 1:
                self.layers.append(DPTReassembleLayer(config, channels=config.neck_hidden_sizes[i], factor=factor))

        if config.readout_type != "project":
            raise ValueError(f"Readout type {config.readout_type} is not supported for DPT-Hybrid.")

        # When using DPT-Hybrid the readout type is set to "project". The sanity check is done on the config file
        self.readout_projects = nn.CellList()
        hidden_size = _get_backbone_hidden_size(config)
        for i in range(len(config.neck_hidden_sizes)):
            if i <= 1:
                self.readout_projects.append(nn.SequentialCell(mint.nn.Identity()))
            elif i > 1:
                self.readout_projects.append(
                    nn.SequentialCell(mint.nn.Linear(2 * hidden_size, hidden_size), ACT2FN[config.hidden_act])
                )

    def _init_reassemble_dpt(self, config):
        for i, factor in zip(range(len(config.neck_hidden_sizes)), config.reassemble_factors):
            self.layers.append(DPTReassembleLayer(config, channels=config.neck_hidden_sizes[i], factor=factor))

        if config.readout_type == "project":
            self.readout_projects = nn.CellList()
            hidden_size = _get_backbone_hidden_size(config)
            for _ in range(len(config.neck_hidden_sizes)):
                self.readout_projects.append(
                    nn.SequentialCell(mint.nn.Linear(2 * hidden_size, hidden_size), ACT2FN[config.hidden_act])
                )

    def construct(self, hidden_states: List[ms.Tensor], patch_height=None, patch_width=None) -> List[ms.Tensor]:
        """
        Args:
            hidden_states (`List[ms.Tensor]`, each of shape `(batch_size, sequence_length + 1, hidden_size)`):
                List of hidden states from the backbone.
        """
        out = []

        for i, hidden_state in enumerate(hidden_states):
            if i not in self.neck_ignore_stages:
                # reshape to (batch_size, num_channels, height, width)
                cls_token, hidden_state = hidden_state[:, 0], hidden_state[:, 1:]
                batch_size, sequence_length, num_channels = hidden_state.shape
                if patch_height is not None and patch_width is not None:
                    hidden_state = mint.reshape(hidden_state, (batch_size, patch_height, patch_width, num_channels))
                else:
                    size = int(math.sqrt(sequence_length))
                    hidden_state = mint.reshape(hidden_state, (batch_size, size, size, num_channels))
                hidden_state = mint.permute(hidden_state, (0, 3, 1, 2))

                feature_shape = hidden_state.shape
                if self.readout_type == "project":
                    # reshape to (batch_size, height*width, num_channels)
                    hidden_state = mint.permute(mint.flatten(hidden_state, start_dim=2), (0, 2, 1))
                    readout = mint.unsqueeze(cls_token, 1).expand_as(hidden_state)
                    # concatenate the readout token to the hidden states and project
                    hidden_state = self.readout_projects[i](mint.cat((hidden_state, readout), -1))
                    # reshape back to (batch_size, num_channels, height, width)
                    hidden_state = mint.reshape(mint.permute(hidden_state, (0, 2, 1)), feature_shape)
                elif self.readout_type == "add":
                    hidden_state = mint.flatten(hidden_state, start_dim=2) + mint.unsqueeze(cls_token, -1)
                    hidden_state = mint.reshape(hidden_state, feature_shape)
                hidden_state = self.layers[i](hidden_state)
            out.append(hidden_state)

        return out


def _get_backbone_hidden_size(config):
    if config.backbone_config is not None and config.is_hybrid is False:
        return config.backbone_config.hidden_size
    else:
        return config.hidden_size


class DPTReassembleLayer(nn.Cell):
    def __init__(self, config, channels, factor):
        super().__init__()
        # projection
        hidden_size = _get_backbone_hidden_size(config)
        self.projection = mint.nn.Conv2d(
            in_channels=hidden_size, out_channels=channels, kernel_size=1, bias=True
        )

        # up/down sampling depending on factor
        if factor > 1:
            # TODO: nn.Conv2dTranspose 已支持，已收录
            self.resize = nn.Conv2dTranspose(
                channels, channels, kernel_size=factor, stride=factor, pad_mode="pad", padding=0, has_bias=True
            )
        elif factor == 1:
            self.resize = mint.nn.Identity()
        elif factor < 1:
            # so should downsample
            self.resize = mint.nn.Conv2d(
                channels, channels, kernel_size=3, stride=int(1 / factor), padding=1, bias=True
            )

    def construct(self, hidden_state):
        hidden_state = self.projection(hidden_state)
        hidden_state = self.resize(hidden_state)
        return hidden_state


class DPTFeatureFusionStage(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.CellList()
        for _ in range(len(config.neck_hidden_sizes)):
            self.layers.append(DPTFeatureFusionLayer(config))

    def construct(self, hidden_states):
        # reversing the hidden_states, we start from the last
        hidden_states = hidden_states[::-1]

        fused_hidden_states = []
        # first layer only uses the last hidden_state
        fused_hidden_state = self.layers[0](hidden_states[0])
        fused_hidden_states.append(fused_hidden_state)
        # looping from the last layer to the second
        for hidden_state, layer in zip(hidden_states[1:], self.layers[1:]):
            fused_hidden_state = layer(fused_hidden_state, hidden_state)
            fused_hidden_states.append(fused_hidden_state)

        return fused_hidden_states


class DPTPreActResidualLayer(nn.Cell):
    """
    ResidualConvUnit, pre-activate residual unit.

    Args:
        config (`[DPTConfig]`):
            Model configuration class defining the model architecture.
    """

    def __init__(self, config):
        super().__init__()

        self.use_batch_norm = config.use_batch_norm_in_fusion_residual
        use_bias_in_fusion_residual = (
            config.use_bias_in_fusion_residual
            if config.use_bias_in_fusion_residual is not None
            else not self.use_batch_norm
        )

        self.activation1 = mint.nn.ReLU()
        self.convolution1 = mint.nn.Conv2d(
            config.fusion_hidden_size,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias_in_fusion_residual
        )

        self.activation2 = mint.nn.ReLU()
        self.convolution2 = mint.nn.Conv2d(
            config.fusion_hidden_size,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias_in_fusion_residual
        )

        if self.use_batch_norm:
            self.batch_norm1 = mint.nn.BatchNorm2d(config.fusion_hidden_size)
            self.batch_norm2 = mint.nn.BatchNorm2d(config.fusion_hidden_size)

    def construct(self, hidden_state: ms.Tensor) -> ms.Tensor:
        residual = hidden_state
        hidden_state = self.activation1(hidden_state)

        hidden_state = self.convolution1(hidden_state)

        if self.use_batch_norm:
            hidden_state = self.batch_norm1(hidden_state)

        hidden_state = self.activation2(hidden_state)
        hidden_state = self.convolution2(hidden_state)

        if self.use_batch_norm:
            hidden_state = self.batch_norm2(hidden_state)

        return hidden_state + residual


class DPTFeatureFusionLayer(nn.Cell):
    """Feature fusion layer, merges feature maps from different stages.

    Args:
        config (`[DPTConfig]`):
            Model configuration class defining the model architecture.
        align_corners (`bool`, *optional*, defaults to `True`):
            The align_corner setting for bilinear upsample.
    """

    def __init__(self, config, align_corners=True):
        super().__init__()

        self.align_corners = align_corners

        self.projection = mint.nn.Conv2d(
            config.fusion_hidden_size, config.fusion_hidden_size, kernel_size=1, bias=True
        )

        self.residual_layer1 = DPTPreActResidualLayer(config)
        self.residual_layer2 = DPTPreActResidualLayer(config)

    def construct(self, hidden_state, residual=None):
        if residual is not None:
            if hidden_state.shape != residual.shape:
                residual = ops.interpolate(
                    residual, size=(hidden_state.shape[2], hidden_state.shape[3]), mode="bilinear", align_corners=False
                )
            hidden_state = hidden_state + self.residual_layer1(residual)

        hidden_state = self.residual_layer2(hidden_state)
        hidden_state = ops.interpolate(
            hidden_state,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=self.align_corners,
            recompute_scale_factor=True,
        )
        hidden_state = self.projection(hidden_state)

        return hidden_state


class DPTPreTrainedModel(MSPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DPTConfig
    base_model_prefix = "dpt"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        pass


class DPTModel(DPTPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.is_hybrid = self.config.is_hybrid
        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states
        self.use_return_dict = self.config.use_return_dict
        self.num_hidden_layers = self.config.num_hidden_layers

        # vit encoder
        if config.is_hybrid:
            self.embeddings = DPTViTHybridEmbeddings(config)
        else:
            self.embeddings = DPTViTEmbeddings(config)
        self.encoder = DPTViTEncoder(config)

        self.layernorm = mint.nn.LayerNorm((config.hidden_size,), eps=config.layer_norm_eps)
        self.pooler = DPTViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        if self.is_hybrid:
            return self.embeddings
        else:
            return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def construct(
        self,
        pixel_values: ms.Tensor,
        head_mask: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndIntermediateActivations]:
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.num_hidden_layers)

        embedding_output = self.embeddings(pixel_values, return_dict=return_dict)

        embedding_last_hidden_states = embedding_output[0] if not return_dict else embedding_output.last_hidden_states

        encoder_outputs = self.encoder(
            embedding_last_hidden_states,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:] + embedding_output[1:]

        return BaseModelOutputWithPoolingAndIntermediateActivations(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            intermediate_activations=embedding_output.intermediate_activations,
        )


# Copied from transformers.models.vit.modeling_vit.ViTPooler with ViT->DPT
class DPTViTPooler(nn.Cell):
    def __init__(self, config: DPTConfig):
        super().__init__()
        self.dense = mint.nn.Linear(config.hidden_size, config.hidden_size)
        # TODO: nn.Tanh()? 未收录，不支持
        self.activation = nn.Tanh()

    def construct(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class DPTNeck(nn.Cell):
    """
    DPTNeck. A neck is a module that is normally used between the backbone and the head. It takes a list of tensors as
    input and produces another list of tensors as output. For DPT, it includes 2 stages:

    * DPTReassembleStage
    * DPTFeatureFusionStage.

    Args:
        config (dict): config dict.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.neck_hidden_sizes = self.config.neck_hidden_sizes

        # postprocessing: only required in case of a non-hierarchical backbone (e.g. ViT, BEiT)
        if config.backbone_config is not None and config.backbone_config.model_type in ["swinv2"]:
            self.reassemble_stage = None
        else:
            self.reassemble_stage = DPTReassembleStage(config)

        self.convs = nn.CellList()
        for channel in config.neck_hidden_sizes:
            self.convs.append(
                mint.nn.Conv2d(channel, config.fusion_hidden_size, kernel_size=3, padding=1, bias=False)
            )

        # fusion
        self.fusion_stage = DPTFeatureFusionStage(config)

    def construct(self, hidden_states: List[ms.Tensor], patch_height=None, patch_width=None) -> List[ms.Tensor]:
        """
        Args:
            hidden_states (`List[ms.Tensor]`, each of shape `(batch_size, sequence_length, hidden_size)` or `(batch_size, hidden_size, height, width)`):
                List of hidden states from the backbone.
        """
        if not isinstance(hidden_states, (tuple, list)):
            raise ValueError("hidden_states should be a tuple or list of tensors")

        if len(hidden_states) != len(self.neck_hidden_sizes):
            raise ValueError("The number of hidden states should be equal to the number of neck hidden sizes.")

        # postprocess hidden states
        if self.reassemble_stage is not None:
            hidden_states = self.reassemble_stage(hidden_states, patch_height, patch_width)

        features = [self.convs[i](feature) for i, feature in enumerate(hidden_states)]

        # fusion blocks
        output = self.fusion_stage(features)

        return output


class DPTDepthEstimationHead(nn.Cell):
    """
    Output head head consisting of 3 convolutional layers. It progressively halves the feature dimension and upsamples
    the predictions to the input resolution after the first convolutional layer (details can be found in the paper's
    supplementary material).
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.head_in_index = self.config.head_in_index

        self.projection = None
        if config.add_projection:
            self.projection = mint.nn.Conv2d(
                256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True
            )

        features = config.fusion_hidden_size
        # TODO: nn.Upsample 已收录，不支持
        self.head = nn.SequentialCell(
            mint.nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=True, recompute_scale_factor=True),
            mint.nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1, bias=True),
            mint.nn.ReLU(),
            mint.nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=True),
            mint.nn.ReLU(),
        )

    def construct(self, hidden_states: List[ms.Tensor]) -> ms.Tensor:
        # use last features
        hidden_states = hidden_states[self.head_in_index]

        if self.projection is not None:
            hidden_states = self.projection(hidden_states)
            hidden_states = mint.nn.ReLU()(hidden_states)

        predicted_depth = self.head(hidden_states)

        predicted_depth = mint.squeeze(predicted_depth, dim=1)

        return predicted_depth


class DPTForDepthEstimation(DPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.backbone = None
        if config.is_hybrid is False and (config.backbone_config is not None or config.backbone is not None):
            backbone_config = getattr(config, "backbone_config", None)
            self.backbone = BitBackbone(backbone_config).set_train(False)
        else:
            self.dpt = DPTModel(config, add_pooling_layer=False)

        # Neck
        self.neck = DPTNeck(config)

        # Depth estimation head
        self.head = DPTDepthEstimationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

        # config
        self.use_return_dict = self.config.use_return_dict
        self.output_hidden_states = self.config.output_hidden_states
        self.output_attentions = self.config.output_attentions
        self.is_hybrid = self.config.is_hybrid
        self.backbone_out_indices = self.config.backbone_out_indices
        if self.config.backbone_config is not None:
            self.backbone_config = True
        else:
            self.backbone_config = None
        if hasattr(self.config.backbone_config, "patch_size"):
            self.backbone_patch_size = self.config.backbone_config.patch_size

    def construct(
        self,
        pixel_values: ms.Tensor,
        head_mask: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple[ms.Tensor], DepthEstimatorOutput]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth depth estimation maps for computing the loss.

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor
        >>> from mindone.transformers import DPTForDepthEstimation
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
        >>> model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="np")
        >>> inputs["pixel_values"] = ms.Tensor(inputs["pixel_values"])

        >>> outputs = model(**inputs)
        >>> predicted_depth = outputs[0]

        >>> # interpolate to original size
        >>> prediction = ms.ops.interpolate(
        ...     predicted_depth.unsqueeze(1),
        ...     size=image.size[::-1],
        ...     mode="bicubic",
        ...     align_corners=False,
        ... )

        >>> # visualize the prediction
        >>> output = prediction.squeeze().asnumpy()
        >>> formatted = (output * 255 / np.max(output)).astype("uint8")
        >>> depth = Image.fromarray(formatted)
        ```"""
        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not implemented yet")

        return_dict = return_dict if return_dict is not None else self.use_return_dict
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions

        if self.backbone is not None:
            outputs = self.backbone.forward_with_filtered_kwargs(
                pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
            )
            hidden_states = outputs[0]
        else:
            outputs = self.dpt(
                pixel_values,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=True,  # we need the intermediate hidden states
                return_dict=return_dict,
            )
            hidden_states = outputs.hidden_states if return_dict else outputs[1]
            # only keep certain features based on config.backbone_out_indices
            # note that the hidden_states also include the initial embeddings
            if not self.is_hybrid:
                hidden_states = [
                    feature for idx, feature in enumerate(hidden_states[1:]) if idx in self.backbone_out_indices
                ]
            else:
                backbone_hidden_states = outputs.intermediate_activations if return_dict else list(outputs[-1])
                backbone_hidden_states.extend(
                    feature for idx, feature in enumerate(hidden_states[1:]) if idx in self.backbone_out_indices[2:]
                )

                hidden_states = backbone_hidden_states

        patch_height, patch_width = None, None
        if self.backbone_config and self.is_hybrid is False:
            _, _, height, width = pixel_values.shape
            patch_size = self.backbone_patch_size
            patch_height = height // patch_size
            patch_width = width // patch_size

        hidden_states = self.neck(hidden_states, patch_height, patch_width)

        predicted_depth = self.head(hidden_states)

        if not return_dict:
            if output_hidden_states:
                output = (predicted_depth,) + outputs[1:]
            else:
                output = (predicted_depth,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return DepthEstimatorOutput(
            loss=loss,
            predicted_depth=predicted_depth,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
