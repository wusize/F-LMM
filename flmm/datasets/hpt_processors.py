from typing import Dict, List, Optional, Union
import numpy as np
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (
    resize,
    convert_to_rgb,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
    validate_kwargs,
    validate_preprocess_arguments,
)
from transformers.utils import TensorType
from transformers.models.clip.image_processing_clip import logger
from flmm.utils import multi_apply
from flmm.datasets.llava_processors import CustomLlavaImageProcessor
from transformers import SiglipImageProcessor
CustomHPTImageProcessor = CustomLlavaImageProcessor


class CustomHPT15ImageProcessor(SiglipImageProcessor):
    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs,
    ):
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        size = get_size_dict(size, param_name="size", default_to_square=False)
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        images = make_list_of_images(images)

        validate_kwargs(captured_kwargs=kwargs.keys(), valid_processor_keys=self._valid_processor_keys)

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )
        validate_preprocess_arguments(
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )

        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        image_sizes = [get_image_size(image, channel_dim=input_data_format) for image in images]

        if do_resize:
            tar_h, tar_w = size["height"], size["width"]
            tmp_images = []
            for image in images:
                cur_h, cur_w = get_image_size(image, channel_dim=input_data_format)
                if tar_h / cur_h < tar_w / cur_w:
                    image = resize(image=image, size=(tar_h, int(cur_w * tar_h / cur_h)),
                                   resample=resample, input_data_format=input_data_format)
                else:
                    image = resize(image=image, size=(int(cur_h * tar_w / cur_w), tar_w),
                                   resample=resample, input_data_format=input_data_format)
                tmp_images.append(image)
            images = tmp_images

        images, meta_datas = multi_apply(self.pad, images)

        if do_rescale:
            images = [
                self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
                for image in images
            ]

        if do_normalize:
            images = [
                self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
                for image in images
            ]

        images = [
            to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format) for image in images
        ]

        data = {"pixel_values": images, "image_sizes": image_sizes, "meta_datas": meta_datas}

        return BatchFeature(data=data, tensor_type=return_tensors)

    def pad(self, image):
        pad_value = np.array(tuple(int(x * 255) for x in self.image_mean), dtype=image.dtype)
        assert isinstance(image, np.ndarray)
        h, w, _ = image.shape
        size = max(h, w)
        new_image = np.ones((size, size, 3), dtype=image.dtype) * pad_value

        pad_height, pad_width = size - h, size - w
        before_height, before_width = pad_height // 2, pad_width // 2
        after_height, after_width = pad_height - before_height, pad_width - before_width

        new_image[before_height:size - after_height, before_width:size - after_width] = image

        meta = dict(padding=dict(before_height=before_height, after_height=after_height,
                                 before_width=before_width, after_width=after_width),
                    image_shape=dict(height=h, width=w),
                    padded_shape=dict(height=size, width=size))

        return new_image, meta


if __name__ == "__main__":
    from PIL import Image
    image_size = 588
    image_processor = CustomHPTImageProcessor.from_pretrained(
        pretrained_model_name_or_path='HyperGAI/HPT',
        subfolder='visual_encoder',
        size={"shortest_edge": image_size},
        crop_size={"height": image_size, "width": image_size}
    )
    image = Image.open('src/datasets/000000000139.jpg')
    image_data = image_processor.preprocess(image)
    print(image_data.keys())
