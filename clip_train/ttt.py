from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoTokenizer,
    AutoImageProcessor
)

cache_dir = "/share/test/songtianwei/cache"
model = VisionTextDualEncoderModel.from_vision_text_pretrained(
    "openai/clip-vit-base-patch32", "roberta-base",
    cache_dir=cache_dir
)

tokenizer = AutoTokenizer.from_pretrained("roberta-base",
                                          cache_dir=cache_dir)
tokenizer2 = AutoTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4",subfolder="tokenizer", revision=None,
                                          cache_dir=cache_dir)
image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32",
                                          cache_dir=cache_dir)
processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)

# save the model and processor
path = "/share/test/songtianwei/model_save"
model.save_pretrained(path)
processor.save_pretrained(path)
image_processor.save_pretrained(path)
tokenizer.save_pretrained(path)
tokenizer2.save_pretrained(path)