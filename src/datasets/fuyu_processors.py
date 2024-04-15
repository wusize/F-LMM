from transformers import LlamaTokenizerFast, FuyuImageProcessor
from typing import Dict, Optional, Union
from transformers.image_transforms import to_channel_dimension_format
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    to_numpy_array,
    validate_preprocess_arguments,
)
from transformers.utils import TensorType
from transformers.models.fuyu.image_processing_fuyu import make_list_of_list_of_images, logger, FuyuBatchFeature
from transformers.image_transforms import resize
import numpy as np


fuyu_template = dict(
    SYSTEM='',
    INSTRUCTION='{input}\n\x04',
    SEP='\n')


class CustomFuyuTokenizer(LlamaTokenizerFast):
    @classmethod
    def from_pretrained(cls, *args, **kwargs,):
        tokenizer = super().from_pretrained(*args, **kwargs)
        tokenizer.bos_token = "<s>"
        tokenizer.pad_token_id = tokenizer.eos_token_id  # 0
        tokenizer.padding_side = 'right'  # 'left'

        special_tokens_dict = {'additional_special_tokens': ['<image>']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        assert num_added_toks == 1
        tokenizer.update_post_processor()

        return tokenizer


class CustomFuyuImageProcessor(FuyuImageProcessor):
    def preprocess(
        self,
        images,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: Optional[PILImageResampling] = None,
        do_pad: Optional[bool] = None,
        padding_value: Optional[float] = None,
        padding_mode: Optional[str] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[float] = None,
        image_std: Optional[float] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        patch_size: Optional[Dict[str, int]] = None,
        data_format: Optional[Union[str, ChannelDimension]] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        return_tensors: Optional[TensorType] = None,
    ):
        """

        Utility function to preprocess the images and extract necessary information about original formats.

        Args:
            images (`ImageInput`):
                Images to preprocess. Expects a single image, a list or images or a list of lists of images. Pixel
                values range from 0 to 255, or between 0 and 1 if `do_rescale` is `False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image to `size`.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
            do_pad (`bool`, *optional*, defaults to `self.do_pad`):
                Whether to pad the image to `size`.
            padding_value (`float`, *optional*, defaults to `self.padding_value`):
                The value to pad the image with.
            padding_mode (`str`, *optional*, defaults to `self.padding_mode`):
                The padding mode to use when padding the image.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float`, *optional*, defaults to `self.image_mean`):
                The mean to use when normalizing the image.
            image_std (`float`, *optional*, defaults to `self.image_std`):
                The standard deviation to use when normalizing the image.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                The factor to use when rescaling the image.
            patch_size (`Dict[str, int]`, *optional*, defaults to `self.patch_size`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the patches.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format of the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
        """

        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        resample = resample if resample is not None else self.resample
        do_pad = do_pad if do_pad is not None else self.do_pad
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        padding_value = padding_value if padding_value is not None else self.padding_value
        padding_mode = padding_mode if padding_mode is not None else self.padding_mode
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        patch_size = patch_size if patch_size is not None else self.patch_size

        if isinstance(images, list) and any(isinstance(elem, list) and len(elem) >= 2 for elem in images):
            raise ValueError("Multiple images for a single sample are not yet supported.")

        batch_images = make_list_of_list_of_images(images)

        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_pad=do_pad,
            size_divisibility=size,  # There is no pad divisibility in this processor, but pad requires the size arg.
            do_resize=do_resize,
            size=size,
            resample=resample,
        )
        # All transformations expect numpy arrays.
        batch_images = [[to_numpy_array(image) for image in images] for images in batch_images]

        if is_scaled_image(batch_images[0][0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(batch_images[0][0])

        original_image_sizes = [get_image_size(images[0], channel_dim=input_data_format) for images in batch_images]

        if do_resize:
            batch_images = [
                [self.resize(image, size=size, input_data_format=input_data_format) for image in images]
                for images in batch_images
            ]

        image_sizes = [get_image_size(images[0], channel_dim=input_data_format) for images in batch_images]
        image_unpadded_heights = [[image_size[0]] for image_size in image_sizes]
        image_unpadded_widths = [[image_size[1]] for image_size in image_sizes]

        # scale_h is the same as scale_w
        image_scale_factors = [
            [resized_size[0] / original_size[0]]
            for original_size, resized_size in zip(original_image_sizes, image_sizes)
        ]

        if do_pad:
            batch_images = [
                [
                    self.pad_image(
                        image,
                        size=size,
                        mode=padding_mode,
                        constant_values=padding_value,
                        input_data_format=input_data_format,
                    )
                    for image in images
                ]
                for images in batch_images
            ]

        if do_rescale:
            batch_images = [
                [self.rescale(image, scale=rescale_factor, input_data_format=input_data_format) for image in images]
                for images in batch_images
            ]

        if do_normalize:
            batch_images = [
                [
                    self.normalize(image, mean=image_mean, std=image_std, input_data_format=input_data_format)
                    for image in images
                ]
                for images in batch_images
            ]

        if data_format is not None:
            batch_images = [
                [to_channel_dimension_format(image, data_format, input_data_format) for image in images]
                for images in batch_images
            ]

        data = {
            "images": batch_images,
            "image_unpadded_heights": image_unpadded_heights,
            "image_unpadded_widths": image_unpadded_widths,
            "image_scale_factors": image_scale_factors,
        }
        return FuyuBatchFeature(data=data, tensor_type=return_tensors)


    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BILINEAR`.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        Returns:
            `np.ndarray`: The resized image.
        """
        image_height, image_width = get_image_size(image, input_data_format)
        target_height, target_width = size["height"], size["width"]

        if image_width <= target_width and image_height <= target_height:
            return image

        height_scale_factor = target_height / image_height
        width_scale_factor = target_width / image_width
        optimal_scale_factor = min(height_scale_factor, width_scale_factor)

        new_height = int(image_height * optimal_scale_factor)
        new_width = int(image_width * optimal_scale_factor)

        scaled_image = resize(
            image=image,
            size=(new_height, new_width),
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
            **kwargs,
        )
        return scaled_image


if __name__ == "__main__":
    tokenizer = CustomFuyuTokenizer.from_pretrained(pretrained_model_name_or_path='adept/fuyu-8b')
    image_processor = CustomFuyuImageProcessor.from_pretrained(pretrained_model_name_or_path='adept/fuyu-8b',
                                                               size={'height': 960, 'width': 960})
    encoded = tokenizer.encode('<image>what is in this image?\n\x04')
    for enc in encoded:
        print(tokenizer.decode(enc), flush=True)
    from PIL import Image
    image = Image.open('src/datasets/000000000139.jpg').convert('RGB')
    image_data = image_processor.preprocess(image)

