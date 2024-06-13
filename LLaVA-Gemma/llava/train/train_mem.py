# ==============================================================================
# Copyright (c) [2024] [Intel Labs]
#
# Original Copyright:
# Copyright (c) [2023] [Haotian Liu]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Modifications:
# - 2024: Adapted for training Gemma LLM with Intel Gaudi 2 AI accelerators.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


# Description:
# train scripts entry point
import torch
from llava.train.train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2" if torch.cuda.is_available() else None)
