from transformers.image_processing_utils import BatchFeature
from transformers.models.clip.image_processing_clip import CLIPImageProcessor


class CustomHPTImageProcessor(CLIPImageProcessor):
    def preprocess(self, images, **kwargs):
        return_tensors = kwargs.pop('return_tensors', None)
        if not isinstance(images, (list, tuple)):
            images = [images]
        image_sizes = [(image.height, image.width) for image in images]
        images = super().preprocess(images, return_tensors=None, **kwargs)['pixel_values']
        meta_datas = [dict(padding=dict(before_height=0, after_height=0, before_width=0, after_width=0),
                           image_shape=dict(height=image.shape[1], width=image.shape[2]),
                           padded_shape=dict(height=image.shape[1], width=image.shape[2]))
                      for image in images]
        data = {"pixel_values": images, "image_sizes": image_sizes, "meta_datas": meta_datas}

        return BatchFeature(data=data, tensor_type=return_tensors)
