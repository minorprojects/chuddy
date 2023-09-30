from Chuddy.utils.header import *
from einops import rearrange
import torch
from resize_rigth import resize
def process_caption(caption):
    caption = re.sub(
        r"([\"()*#:;~])",
        " ",
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        " ",
        caption,
    )
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")

    return caption
class _Rescale:
    """
    Transformation to scale images to the proper size
    """

    def __init__(self, side_length):
        self.side_length = side_length

    def __call__(self, sample, *args, **kwargs):
        if len(sample.shape) == 2:
            sample = rearrange(sample, 'h w -> 1 h w')
        elif not len(sample.shape) == 3:
            raise ValueError("Improperly shaped image for rescaling")

        sample = _resize_image_to_square(sample, self.side_length)

        # If there was an error in the resizing, return None
        if sample is None:
            return None

        # Rescaling max push images out of [0,1] range, so have to standardize:
        sample -= sample.min()
        sample /= sample.max()
        return sample
        
def _resize_image_to_square(image: torch.tensor,
                            target_image_size: int,
                            clamp_range: tuple = None,
                            pad_mode: Literal['constant', 'edge', 'reflect', 'symmetric'] = 'reflect'
                            ) -> torch.tensor:
    """
    Resizes image to desired size.

    :param image: Images to resize. Shape (b, c, s, s)
    :param target_image_size: Edge length to resize to.
    :param clamp_range: Range to clamp values to. Tuple of length 2.
    :param pad_mode: `constant`, `edge`, `reflect`, `symmetric`.
        See [TorchVision documentation](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.pad.html) for additional details
    :return: Resized image. Shape (b, c, target_image_size, target_image_size)
    """
    h_scale = image.shape[-2]
    w_scale = image.shape[-1]

    if h_scale == target_image_size and w_scale == target_image_size:
        return image

    scale_factors = (target_image_size / h_scale, target_image_size / w_scale)
    try:
        out = resize(image, scale_factors=scale_factors, pad_mode=pad_mode)
    except:
        return None

    if exists(clamp_range):
        out = out.clamp(*clamp_range)

    return out
