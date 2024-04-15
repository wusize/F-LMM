from transformers import LlavaNextImageProcessor
from typing import Dict, List, Optional, Union

import numpy as np

from transformers.image_processing_utils import BatchFeature, get_size_dict, select_best_resolution
from transformers.image_transforms import (
    convert_to_rgb,
    pad,
    resize,
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
    validate_preprocess_arguments,
)

from transformers.models.llava_next.image_processing_llava_next import (
    divide_to_patches, _get_patch_output_size, logger)
from transformers.utils import TensorType


class CustomLlavaNextImageProcessor(LlavaNextImageProcessor):
    def get_image_patches(
        self,
        image: np.array,
        grid_pinpoints,
        size: tuple,
        patch_size: int,
        resample: PILImageResampling,
        data_format: ChannelDimension,
        input_data_format: ChannelDimension,
    ):
        """
        Process an image with variable resolutions by dividing it into patches.

        Args:
            image (np.array):
                The input image to be processed.
            grid_pinpoints (List):
                A string representation of a list of possible resolutions.
            size (`tuple`):
                Size to resize the original image to.
            patch_size (`int`):
                Size of the patches to divide the image into.
            resample (`PILImageResampling`):
                Resampling filter to use if resizing the image.
            data_format (`ChannelDimension` or `str`):
                The channel dimension format for the output image.
            input_data_format (`ChannelDimension` or `str`):
                The channel dimension format of the input image.

        Returns:
            List[np.array]: A list of NumPy arrays containing the processed image patches.
        """
        if not isinstance(grid_pinpoints, list):
            raise ValueError("grid_pinpoints must be a list of possible resolutions.")

        possible_resolutions = grid_pinpoints

        image_size = get_image_size(image, channel_dim=input_data_format)
        best_resolution = select_best_resolution(image_size, possible_resolutions)
        resized_image = self._resize_for_patching(
            image, best_resolution, resample=resample, input_data_format=input_data_format
        )
        padded_image, meta_data = self._pad_for_patching(resized_image, best_resolution,
                                                         input_data_format=input_data_format)

        patches = divide_to_patches(padded_image, patch_size=patch_size, input_data_format=input_data_format)
        meta_data.update(grid_shape=self.get_grid_shape(image=padded_image, patch_size=patch_size,
                                                        input_data_format=input_data_format),
                         ori_shape=dict(height=image_size[0], width=image_size[1]))


        for patch in patches:
            assert get_image_size(patch, channel_dim=input_data_format) == (patch_size, patch_size)

        # make sure that all patches are in the input data format
        patches = [
            to_channel_dimension_format(patch, channel_dim=data_format, input_channel_dim=input_data_format)
            for patch in patches
        ]

        resized_original_image = resize(
            image,
            size=size,
            resample=resample,
            data_format=data_format,
            input_data_format=input_data_format,
        )

        image_patches = [resized_original_image] + patches

        return image_patches, meta_data


    def _pad_for_patching(
        self, image: np.array, target_resolution: tuple, input_data_format: ChannelDimension
    ) -> np.array:
        """
        Pad an image to a target resolution while maintaining aspect ratio.
        """
        target_height, target_width = target_resolution
        new_height, new_width = _get_patch_output_size(image, target_resolution, input_data_format)

        pad_height = target_height - new_height
        pad_width = target_width - new_width

        # there is a bug in the official implementation

        before_height, before_width = pad_height // 2, pad_width // 2
        after_height, after_width = pad_height - before_height, pad_width - before_width

        padded_image = pad(image, padding=((before_height, after_height), (before_width, after_width)))

        return padded_image, dict(padding=dict(before_height=before_height, after_height=after_height,
                                               before_width=before_width, after_width=after_width),
                                  image_shape=dict(height=new_height, width=new_width),
                                  padded_shape=dict(height=target_height, width=target_width))


    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        size: Dict[str, int] = None,
        image_grid_pinpoints: List = None,
        resample: PILImageResampling = None,
        do_center_crop: bool = None,
        crop_size: int = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. Shortest edge of the image is resized to size["shortest_edge"], with
                the longest edge resized to keep the input aspect ratio.
            image_grid_pinpoints (`List` *optional*, defaults to `self.image_grid_pinpoints`):
                A list of possible resolutions to use for processing high resolution images. The best resolution is
                selected based on the original size of the image.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_center_crop (`bool`, *optional*, defaults to `self.do_center_crop`):
                Whether to center crop the image.
            crop_size (`Dict[str, int]`, *optional*, defaults to `self.crop_size`):
                Size of the center crop. Only has an effect if `do_center_crop` is set to `True`.
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
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
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
        image_grid_pinpoints = image_grid_pinpoints if image_grid_pinpoints is not None else self.image_grid_pinpoints
        resample = resample if resample is not None else self.resample
        do_center_crop = do_center_crop if do_center_crop is not None else self.do_center_crop
        crop_size = crop_size if crop_size is not None else self.crop_size
        crop_size = get_size_dict(crop_size, param_name="crop_size", default_to_square=True)
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        images = make_list_of_images(images)

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
            do_center_crop=do_center_crop,
            crop_size=crop_size,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        if do_convert_rgb:
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

        new_images = []
        image_sizes = [get_image_size(image, channel_dim=input_data_format) for image in images]
        meta_datas = []
        for image in images:
            # convert image into a list of patches
            # we intentially use the same data format as the input data format
            image_patches, meta_data = self.get_image_patches(
                image,
                image_grid_pinpoints,
                size=(size["shortest_edge"], size["shortest_edge"]),
                patch_size=crop_size["height"],
                resample=resample,
                data_format=input_data_format,
                input_data_format=input_data_format,
            )

            # preprocess patches
            pixel_values = self._preprocess(
                image_patches,
                do_resize=False,   # no need
                size=size,
                resample=resample,
                do_center_crop=False,   # no need
                crop_size=crop_size,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                data_format=data_format,
                input_data_format=input_data_format,
            )
            pixel_values = np.array(pixel_values)
            new_images.append(pixel_values)
            meta_datas.append(meta_data)

        data = {"pixel_values": new_images, "image_sizes": image_sizes, "meta_datas": meta_datas}

        return BatchFeature(data=data, tensor_type=return_tensors)


    @staticmethod
    def get_grid_shape(image, patch_size, input_data_format):
        height, width = get_image_size(image, channel_dim=input_data_format)
        assert height % patch_size == 0
        assert width % patch_size == 0

        return dict(height=height // patch_size, width=width // patch_size)
