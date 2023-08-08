from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoTokenizer,
    AutoImageProcessor
)

model = VisionTextDualEncoderModel.from_vision_text_pretrained(
    "openai/clip-vit-base-patch32", "roberta-base",
    cache_dir="/share/test/songtianwei/model_save"
)

tokenizer = AutoTokenizer.from_pretrained("roberta-base",
                                          cache_dir="/share/test/songtianwei/model_save")
image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32",
                                          cache_dir="/share/test/songtianwei/model_save")
processor = VisionTextDualEncoderProcessor(image_processor, tokenizer)

# save the model and processor
model.save_pretrained("/share/test/songtianwei/model_save")
processor.save_pretrained("/share/test/songtianwei/model_save")