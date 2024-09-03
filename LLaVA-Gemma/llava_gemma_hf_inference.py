import argparse
import requests
import torch
from PIL import Image

if torch.cuda.is_available():
    device="cuda"
else:
    from habana_frameworks.torch import hpu
    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
    adapt_transformers_to_gaudi()
    device="hpu"
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
)

def main(args):

    checkpoint = args.model_path
    # Load model
    model = LlavaForConditionalGeneration.from_pretrained(checkpoint).to(device)
    processor = AutoProcessor.from_pretrained(checkpoint)
    # Prepare inputs
    # Use gemma chat template
    prompt = processor.tokenizer.apply_chat_template(
        [{'role': 'user', 'content': "<image>\nWhat's the content of the image?"}],
        tokenize=False,
        add_generation_prompt=True
    )
    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    # Generate
    generate_ids = model.generate(**inputs, temperature=0, max_length=50)
    output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Intel/llava-gemma-2b")
    args = parser.parse_args()
    main(args)
