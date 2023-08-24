import torch

from torchvision.transforms import CenterCrop, ConvertImageDtype, Normalize, Resize, ToTensor, Compose
from torchvision.transforms.functional import InterpolationMode

class Transform(torch.nn.Module):
    def __init__(self, image_size, mean=None, std=None):
        super().__init__()
        self.transforms = Compose([
            Resize([image_size], interpolation=InterpolationMode.BICUBIC,antialias=None),
            CenterCrop(image_size),  # CenterCrop is required because Resize doesn't ensure same output size
            # ConvertImageDtype(torch.float),
            ToTensor(), 
        ])
        if mean is not None and std is not None:
            self.transforms.transforms.append(Normalize(mean=mean, std=std))

    def forward(self, x) -> torch.Tensor:
        """`x` should be an instance of `PIL.Image.Image`"""
        with torch.no_grad():
            x = self.transforms(x)
        return x

def normalize_fn(x, mean, std):
    return Normalize(mean=mean, std=std)(x)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "return_loss": True,
    }

