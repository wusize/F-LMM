from transformers.image_processing_utils import BatchFeature
import numpy as np
from PIL import Image
from mmengine.logging import print_log


class Pad2Square:
    def __init__(self, image_mean=(0.48145466, 0.4578275, 0.40821073)):
        if not isinstance(image_mean[0], int):
            image_mean = tuple(int(x * 255) for x in image_mean)
        print_log(f"image_mean: {image_mean}")
        self.image_mean = image_mean

    def preprocess(self, image, return_tensors=None):
        image = image.convert('RGB')

        width, height = image.size
        if width == height:
            result = image
            before_height = after_height = before_width = after_width = 0
        elif width > height:
            result = Image.new(image.mode, (width, width), self.image_mean)
            result.paste(image, (0, (width - height) // 2))
            before_height = (width - height) // 2
            after_height = (width - height) - before_height
            before_width = after_width = 0
        else:
            result = Image.new(image.mode, (height, height), self.image_mean)
            result.paste(image, ((height - width) // 2, 0))
            # return result
            before_width = (height - width) // 2
            after_width = (height - width) - before_width
            before_height = after_height = 0

        meta = dict(padding=dict(before_height=before_height, after_height=after_height,
                                 before_width=before_width, after_width=after_width),
                    image_shape=dict(height=height, width=width),
                    padded_shape=dict(height=max(height, width), width=max(height, width)))

        data = {"pixel_values": [result], "image_sizes": [(height, width)], "meta_datas": [meta]}

        return BatchFeature(data=data, tensor_type=return_tensors)
