"""
llava.py

Class definition for the LLaVa VLM, wrapping utilities for VQA, image captioning, and (WIP) conditional likelihood
estimation.

Reference: https://github.com/haotian-liu/LLaVA/tree/main
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
from datetime import timedelta

import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from accelerate import PartialState
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.model import LlavaLlamaForCausalLM
from PIL.Image import Image
from transformers import AutoTokenizer, CLIPImageProcessor, TextStreamer
from transformers.utils import is_accelerate_available, strtobool
from transformers.training_args import ParallelMode
from habana_frameworks.torch import hpu as hthpu
import habana_frameworks.torch.core as htcore
from optimum.habana.accelerate.state import GaudiAcceleratorState, GaudiPartialState
from optimum.habana.accelerate.utils import GaudiDistributedType
from optimum.habana.transformers import GaudiConfig
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

from vlm_eval.models.llava import LLaVa
from vlm_eval.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger` and `accelerate.PartialState`
overwatch = initialize_overwatch(__name__)

class LLaVaGemmaGaudi(LLaVa):
    def __init__(
        self,
        model_family: str,
        model_id: str,
        run_dir: Path,
        load_precision: str = "bf16",
        ocr: bool = False,
        max_length: int = 128,
        temperature: float = 0.2,
        ddp_backend: str = 'hccl',
        gaudi_config_name: Optional[str] = None,
        **_: str,
    ) -> None:
        self.model_family, self.model_id, self.hub_path = model_family, model_id, run_dir
        self.ddp_backend = ddp_backend
        self.dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[load_precision]
        self.ocr = ocr
        self.gaudi_config_name = gaudi_config_name

        disable_torch_init()
        self.model_name = get_model_name_from_path(str(self.hub_path))

        if 'gemma' in self.model_name.lower():
            self.conv_mode = "llava_gemma"
            self.model_name = "llava_gemma"
        elif "llama-2" in self.model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            self.conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            self.conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"

        self._setup_devices()
        if self.distributed_state.is_main_process:
            overwatch.info(f"model_family: {self.model_family}")
            overwatch.info(f"model_id: {self.model_id}")
            overwatch.info(f"model_name: {self.model_name}")
            overwatch.info(f"dtype: {self.dtype}")
            overwatch.info(f"ocr: {self.ocr}")
            overwatch.info(f"device: {self.device}")
            overwatch.info(f"num_processes: {self.num_processes}")


        # Load Model on GPU(s) --> download if necessary via HF Hub
        with self.distributed_state.main_process_first():
            self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(self.hub_path, None, self.model_name, False, False, device=self.device)

        #self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # LLaVa is a chat-based model --> Load Chat-Specific VQA Prompts following LLaVa SciQA
        #   Ref: https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa_science.py#L29
        self.conv = conv_templates[self.conv_mode].copy()
        if "mpt" in self.model_name.lower():
            self.conv.roles = ('user', 'assistant')

        # Set Default Generation Configuration --> again from the Github Repository!
        self.max_length = max_length
        self.temperature = temperature

        # For computing likelihoods --> get tokens corresponding to "True", "False" and "Yes", "No"
        self.string2idx = {}
        for trigger_string in ["True", "False", "Yes", "No"] + [chr(ord("A") + i) for i in range(26)]:
            token_idx_list = self.tokenizer.encode(trigger_string, add_special_tokens=False)
            assert len(token_idx_list) == 1, f'String "{trigger_string}" is tokenized as more than one token!'
            self.string2idx[trigger_string] = token_idx_list[0]

    @torch.inference_mode()
    def generate_answer(
        self, pixel_values: torch.Tensor, questions: List[str], return_string_probabilities: Optional[List[str]] = None
    ) -> Union[List[str], List[List[float]]]:
        # By default, LLaVa code only neatly handles processing a single example at a time, due to the way the <image>
        # tokens are interleaved with the text; this code just loops over inputs (naive padding doesn't work...)
        #with torch.cuda.amp.autocast(dtype=self.dtype):
        with torch.inference_mode():
            batch_input_ids = [
                tokenizer_image_token(q, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt", hpu_skip_inds=False).unsqueeze(0).to(pixel_values.device)
                for q in questions
            ]

            # Greedy Decoding
            gen_texts, gen_probabilities = [], []
            for idx, input_ids in enumerate(batch_input_ids):
                stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                if return_string_probabilities is None:
                    full_out_ids = self.model.generate(
                        input_ids,
                        images=pixel_values,
                        image_sizes=[pixel_values.size],
                        do_sample=False, #True if self.temperature > 0 else False,
                        temperature=self.temperature,
                        max_new_tokens=self.max_length,
                        #streamer=self.streamer,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria],
                    )

                    # Decode results and strip any <EOS> tokens
                    gen_texts.append(self.tokenizer.decode(full_out_ids[0], skip_special_tokens=True).strip())

                else:
                    full_out_dict = self.model.generate(
                        input_ids,
                        images=pixel_values,
                        image_sizes=[pixel_values.size],
                        do_sample=False, #True if self.temperature > 0 else False,
                        temperature=self.temperature,
                        max_new_tokens=self.max_length,
                        #streamer=self.streamer,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria],
                        output_scores=True,
                        return_dict_in_generate=True,
                    )

                    # Decode results and strip any <EOS> tokens
                    gen_texts.append(self.tokenizer.decode(full_out_dict.sequences[0], skip_special_tokens=True).strip())

                    # Get all token probabilities --> softmax over logits
                    token_probs = torch.softmax(full_out_dict.scores[0][0], dim=0)

                    # Get *normalized* probabilities for all values in `return_string_probabilities`
                    slice_idxs = torch.tensor([self.string2idx[s] for s in return_string_probabilities])
                    string_probs_unnormalized = token_probs[slice_idxs]
                    string_probs = string_probs_unnormalized / string_probs_unnormalized.sum()
                    gen_probabilities.append(string_probs.cpu().numpy().tolist())
        return gen_texts if return_string_probabilities is None else gen_probabilities

    def _setup_devices(self):
        if self.gaudi_config_name is not None:
            gaudi_config = GaudiConfig.from_pretrained(self.gaudi_config_name)
            if self.dtype == torch.bfloat16:
                gaudi_config.declare_autocast_bf16_fp32_ops()

        if not is_accelerate_available():
            raise ImportError(
                f"Accelerate must be installed"
            )

        GaudiAcceleratorState._reset_state()
        GaudiPartialState._reset_state()
        self._distributed_state = None

        # Some methods needs to be tweaked to optimally run on Gaudi
        # Calling this method here to be sure it is done before model instantiation
        # Otherwise this will fail when some __init__ methods are overridden (cf. GPT2Attention)
        adapt_transformers_to_gaudi()

        self._distributed_state = GaudiPartialState(
            backend=self.ddp_backend, timeout=timedelta(seconds=1800)
        )
        self._device = self.distributed_state.device

    @property
    def device(self) -> "torch.device":
        """
        The device used by this process.
        """
        if not hasattr(self, "_device"):
            self._setup_devices()
        return self._device

    @property
    def distributed_state(self):
        """
        Get the distributed state.
        """
        if not hasattr(self, "_distributed_state"):
            self._setup_devices()
        return self._distributed_state

    @property
    def num_processes(self):
        """
        The number of processes used in parallel.
        """
        if not hasattr(self, "_distributed_state"):
            self._setup_devices()
        if self.distributed_state is not None:
            return self.distributed_state.num_processes
        return 1

    @property
    def process_index(self):
        """
        The index of the current process used.
        """
        if not hasattr(self, "_distributed_state"):
            self._setup_devices()
        if self.distributed_state is not None:
            return self.distributed_state.process_index
        return 0

    @property
    def local_process_index(self):
        """
        The index of the local process used.
        """
        if not hasattr(self, "_distributed_state"):
            self._setup_devices()
        if self.distributed_state is not None:
            return self.distributed_state.local_process_index
        return 0
