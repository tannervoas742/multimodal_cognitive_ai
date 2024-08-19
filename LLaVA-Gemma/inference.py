#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys
sys.path.append("/home/sdp/krutrim/ayush.tarun/LLaVA-Gaudi")


# In[3]:


import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path#, KeywordsStoppingCriteria
from transformers import StoppingCriteria
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
import torch
from llava.model import *

from PIL import Image
import os
import shutil

import requests
from PIL import Image
from io import BytesIO


# In[ ]:


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

        # print("Stopping Criteria keyowrds: ", self.keywords)

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
            # print("Stopping Criteria: In if")
            return False
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)
            flag = False
            
            # print("Stopping Criteria: In else")
            for output in outputs:
                for keyword in self.keywords:
                    # print(f"output: {output}, keyword: {keyword}, all keywords: {self.keywords}")
                    if keyword in output:
                        return True
            return flag


# In[ ]:


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


# In[ ]:


# from transformers import AutoModel, AutoTokenizer
# import os

# def download_huggingface_model(model_name, save_directory):
#     # Create the directory if it doesn't exist
#     if not os.path.exists(save_directory):
#         os.makedirs(save_directory)
    
#     # Download model and tokenizer
#     model = AutoModel.from_pretrained(model_name)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)

#     # Save model and tokenizer to the specified directory
#     model.save_pretrained(save_directory)
#     tokenizer.save_pretrained(save_directory)

# model_name = 'lmsys/vicuna-7b-v1.5'  # Replace with the desired model name
# save_directory = '/home/sdp/krutrim/ayush.tarun/vicuna_model'  # Replace with the desired save directory

# download_huggingface_model(model_name, save_directory)


# In[ ]:


# save_directory = "vicuna-7b-v1.5"
# model.save_pretrained(save_directory)
# tokenizer.save_pretrained(save_directory)


# In[ ]:


model_base = "../../responder_v2_mpt"
model_base = "/scratch1/responder_v2_mpt"
model_base = "/workspace/ola_mpt_training/responder_v2_mpt/"
#model_base = None
#model_base = "../../vicuna_model"
#model_path = "/scratch-1/shahrukh/llava_projector"
#model_path = "/scratch-1/shahrukh/llava-mpt-responder_v2_mpt-pretrain"
#model_path = "/scratch1/data/checkpoints/llava-mpt-responder_v2_mpt-pretrain"
model_path = "/data0/visual-llama/checkpoints/llava-mpt-responder_v2_mpt-pretrain_2k_new_llava_2"


# In[ ]:


tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base=model_base, model_name='llava_mpt')
#tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base=model_base, model_name='llava_vicuna')

# In[7]:


import habana_frameworks.torch.core as htcore
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
adapt_transformers_to_gaudi()


# In[8]:


device = 'hpu'
model = model.to(device)


# In[15]:
import llava.conversation as conversation_lib

def eval_model(tokenizer, model, image_processor, context_len, query, image_file, sep=','):

    disable_torch_init()
    
    qs = "### " + query
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    # conv_mode = "krutrim"
    source = [{"from": "human", "value": f"<image>\n{query}"}]
    header = f"{conversation_lib.default_conversation.system}\n\n"
    # conversation = _add_speaker_and_signal(header, source)
    prompt = f"<|SYSTEM|> {header}<|USER|> <user> <image>\n{query} </user>\n<assistant> <|RESPONSE|> "
    print("prompt ", {prompt})
    if image_file:
        image = load_image(image_file)
        #image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to('hpu')#.cuda()
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].bfloat16().to('hpu')#.cuda()
    else:
        image_tensor = None

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt', hpu_skip_inds=False).unsqueeze(0).to('hpu')#.cuda()

    stop_str = "###"#conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = ["###", "</s>", '</assistant>']
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            temperature=0.9,
            top_k=200,
            top_p=0.9,
            max_new_tokens=100,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    print(output_ids[:, input_token_len:])
    print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()

    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    print(outputs)


# In[10]:


from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


# In[11]:


# model = AutoModelForCausalLM.from_pretrained("liuhaotian/llava-v1.5-7b")
# tokenizer = AutoTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b")
# device = 'hpu'
# model = model.to(device)


# In[16]:


query = "What is this image about?"
#image_file = "./images/lake.jpeg"
#image_file = "./llava/serve/examples/extreme_ironing.jpg"
image_file = "./llava/serve/examples/waterview.jpg"

eval_model(tokenizer, model, image_processor, context_len, query, image_file)


# In[ ]:




