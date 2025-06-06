from typing import List, Optional, Union

from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import ImageInput, get_image_size, to_numpy_array
from transformers.models.video_llava.processing_video_llava import VideoLlavaProcessor
from transformers.tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType

from optimum.utils import logging


logger = logging.get_logger(__name__)


class GaudiVideoLlavaProcessor(VideoLlavaProcessor):
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "patch_size", "vision_feature_select_strategy", "image_token", "video_token"]
    image_processor_class = "VideoLlavaImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size=None,
        vision_feature_select_strategy=None,
        image_token="<image>",  # set the default and let users change if they have peculiar special tokens in rare cases
        video_token="<video>",
        chat_template=None,
        **kwargs,
    ):
        r"""
        Copied from https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/video_llava/processing_video_llava.py#L61
        except move super().__init__ to the top of the function so that it will not change the default value for self.patch_size and self.vision_feature_select_strategy
        """
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.patch_size = patch_size
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_token = image_token
        self.video_token = video_token

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        videos: ImageInput = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        r"""
        Copied from https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/models/video_llava/processing_video_llava.py#L78
        Here we use the processor logit in transformers 4.45.2 to avoid the performance regression for op `masked_scatter` in transformers 4.49 on Gaudi3
        """
        data = {}
        if images is not None or videos is not None:
            encoded_images = self.image_processor(images=images, videos=videos, return_tensors=return_tensors)
            data.update(encoded_images)

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        prompt_strings = text
        if encoded_images is not None and (self.patch_size is None or self.vision_feature_select_strategy is None):
            logger.warning_once(
                "Expanding inputs for image tokens in Video-LLaVa should be done in processing. "
                "Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly "
                "with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. "
                "Using processors without these attributes in the config is deprecated and will throw an error in v4.44."
            )
        # Replace the image/video tokens with the expanded token sequence
        elif encoded_images is not None:
            if "pixel_values_images" in encoded_images.keys():
                height, width = get_image_size(to_numpy_array(encoded_images.get("pixel_values_images")[0]))
                num_frames = 1

            if "pixel_values_videos" in encoded_images.keys():
                one_video = to_numpy_array(encoded_images.get("pixel_values_videos")[0])
                height, width = get_image_size(one_video[0])
                num_frames = one_video.shape[0]  # frame dim is always after batch dim

            num_image_tokens = (height // self.patch_size) * (width // self.patch_size) + 1
            num_video_tokens = num_image_tokens * num_frames

            num_image_tokens = (height // self.patch_size) * (width // self.patch_size) + 1
            num_video_tokens = num_image_tokens * num_frames
            if self.vision_feature_select_strategy == "default":
                num_image_tokens -= 1

            prompt_strings = []
            for sample in text:
                sample = sample.replace(self.image_token, self.image_token * num_image_tokens)
                sample = sample.replace(self.video_token, self.video_token * num_video_tokens)
                prompt_strings.append(sample)

        text_inputs = self.tokenizer(
            prompt_strings,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )
        data.update(text_inputs)

        return BatchFeature(data=data)
