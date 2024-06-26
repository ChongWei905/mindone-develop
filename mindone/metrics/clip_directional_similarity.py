from mindspore.nn import Cell
from mindspore import ops, Tensor
from typing_extensions import Literal
from typing import Any, Union, List
import mindspore as ms
from mindone.metrics.functional.clip_score import _get_clip_model_and_processor
from mindone.metrics.functional.clip_directional_similarity import _clip_directional_similarity_update


class ClipDirectionalSimilarity(Cell):
    r"""Calculates `CLIP Directional Similarity`_ to measure the consistency of the change between the two images
    (in CLIP space) with the change between the two image captions.
    The metric was originally proposed in `clip_dir_sim ref1`_.

    .. math::
        \text{CLIPDirectionalSimilarity(I1, I2, C1, C2)} = max(100 * cos(E_I1 - E_I2, E_C1 - E_C2), 0)

    which corresponds to the cosine similarity between the difference of visual `CLIP`_ embeddings of two images and
    textual CLIP embeddings of two texts. The higher the CLIP directional similarity, the better it is.

    .. note:: Clip Directional Similarity metric does not support GRAPH_MODE

    As input to ``construct`` and ``update`` the metric accepts the following input

    - ``origin_image``: Tensor, origin image tensor
    - ``generated_image``: Tensor, generated image tensor
    - ``origin_text``: Union[str, List[str]], origin text string
    - ``edited_text``: Union[str, List[str]], edited text string

    As output of `construct` and `compute` the metric returns the following output

    - ``clip_directional_similarity`` (:class:`~mindspore.Tensor`): float scalar tensor with mean CLIP directional
    similarity over samples

    Args:
        model_name_or_path: string indicating the version of the CLIP model to use. Available models are:

            - `"clip_vit_b_16"`
            - `"clip_vit_b_32"`
            - `"clip_vit_l_14@336"`
            - `"clip_vit_l_14"`

        kwargs: Additional keyword arguments, passed to parent class mindspore.nn.Cell directly.

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindone.metrics.clip_directional_similarity import ClipDirectionalSimilarity
        >>> np.random.seed(123)
        >>> origin_image_np = np.random.randint(0, 255, (3, 224, 224))
        >>> generated_image_np = np.random.randint(0, 255, (3, 224, 224))
        >>> origin_text = "a photo of cat"
        >>> edited_text = "a photo of dog"
        >>> original_image_ms = ms.Tensor(origin_image_np).to(ms.uint8)
        >>> original_caption_ms = origin_text
        >>> edited_image_ms = ms.Tensor(generated_image_np).to(ms.uint8)
        >>> modified_caption_ms = edited_text
        >>> metric = ClipDirectionalSimilarity()
        >>> metric.update(original_image_ms, edited_image_ms, original_caption_ms, modified_caption_ms)
        >>> metric.compute()
        -0.028463412
        note: the output may be different since features extracted from clip model are different. We're trying to
        fix this problem with mindnlp developers.

        """

    def __init__(
        self,
        pretrained_model: Literal[
            "clip_vit_b_16",
            "clip_vit_b_32",
            "clip_vit_l_14@336",
            "clip_vit_l_14",
        ] = "clip_vit_l_14",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.model, self.processor = _get_clip_model_and_processor(pretrained_model)
        self.sim_direction = []
        if ms.communication.GlobalComm.INITED:
            self.all_gather = ops.AllGather()

    def update(
        self,
        origin_image: Tensor,
        generated_image: Tensor,
        origin_text: Union[str, List[str]],
        edited_text: Union[str, List[str]],
    ):

        img_feat_one, img_feat_two, text_feat_one, text_feat_two = _clip_directional_similarity_update(
            origin_image, generated_image, origin_text, edited_text, self.model, self.processor
        )
        self.sim_direction.append(ops.cosine_similarity(img_feat_two - img_feat_one, text_feat_two - text_feat_one))

    def compute(self):
        return ops.mean(ops.stack(self.sim_direction))

    def construct(
        self,
        origin_image: Tensor,
        generated_image: Tensor,
        origin_text: Union[str, List[str]],
        edited_text: Union[str, List[str]],
    ):
        processed_input1 = self.processor(origin_image, origin_text)
        image1_features = self.model.get_image_features(processed_input1['image'])
        image1_features = image1_features / ops.norm(image1_features, ord=2, dim=-1, keepdim=True)
        text1_features = self.model.get_text_features(processed_input1['text'])
        text1_features = text1_features / ops.norm(text1_features, ord=2, dim=-1, keepdim=True)

        processed_input2 = self.processor(generated_image, edited_text)
        image2_features = self.model.get_image_features(processed_input2['image'])
        image2_features = image2_features / ops.norm(image2_features, ord=2, dim=-1, keepdim=True)
        text2_features = self.model.get_text_features(processed_input2['text'])
        text2_features = text2_features / ops.norm(text2_features, ord=2, dim=-1, keepdim=True)

        sim_direction = ops.cosine_similarity(image1_features - image2_features, text1_features - text2_features)
        if ms.communication.GlobalComm.INITED:
            sim_direction = self.all_gather(sim_direction)

        return ops.mean(sim_direction)

    def reset(self):
        self.sim_direction = []
