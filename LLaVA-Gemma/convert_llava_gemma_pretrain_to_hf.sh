# Run inference based on pretrained checkpoints
python -m llava.serve.cli --model-base google/gemma-2b-it --model-path /data0/visual-llama/checkpoints/llava-gemma-2b-it-pretrain/checkpoint-2180/ --image-file llava/serve/examples/waterview.jpg

# Convert checkpoints to hf format, llava-gemma inference needs transformer 4.41.1
pip install transformers==4.41.1
# Convert pretrained checkpoints to hf format
python -m llava.serve.convert2hf --model-base google/gemma-2b-it --model-path /data0/visual-llama/checkpoints/llava-gemma-2b-it-pretrain/checkpoint-2180/ --device hpu --output-dir hf-llava-gemma-2b-it-pretrain

# Run inference based on hf transformer
PT_HPU_LAZY_MODE=0 python llava_gemma_hf_inference.py --model-path hf-llava-gemma-2b-it-pretrain

# Restore transformer
pip install transformers==4.40.2

