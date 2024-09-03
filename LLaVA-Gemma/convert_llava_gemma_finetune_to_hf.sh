# Run inference based on finetune checkpoints
python -m llava.serve.cli --model-path /data0/visual-llama/checkpoints/llava-gemma-2b-it-finetune/checkpoint-5150/ --image-file llava/serve/examples/waterview.jpg

# Convert checkpoints to hf format, llava-gemma inference needs transformer 4.41.1
pip install transformers==4.41.1

# Convert finetune checkpoints to hf format
python -m llava.serve.convert2hf --model-path /data0/visual-llama/checkpoints/llava-gemma-2b-it-finetune/checkpoint-5150/ --device hpu --output-dir hf-llava-gemma-2b-it-finetune

# Run inference based on hf transformer
PT_HPU_LAZY_MODE=0 python llava_gemma_hf_inference.py --model-path hf-llava-gemma-2b-it-finetune

# Restore transformer
pip install transformers==4.40.2

