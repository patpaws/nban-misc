# In this blog, I’ll show you a quick tip to use PEFT adapter with vLLM. It accelerates your fine-tuned model in production! vLLM is an amazing, easy-to-use library for LLM inference and serving. As of December 2023, vLLM doesn’t support adapters directly. A quick solution is to merge the weights and push a new model. Here, I will show you how to do that:
#
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
)
import numpy as np
from transformers import LlamaTokenizer
from transformers import pipeline
from peft import PeftModel

model_name = "meta-llama/Llama-2-7b-hf"
peft_model="<username>/<my-peft-model>"
merged_peft_model_name="<username>/<my-peft-model>-merged-peft"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
max_seq_length = tokenizer.model_max_length

bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=token,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload")

model = PeftModel.from_pretrained(
    model,
    peft_model,
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload",
)

model = model.merge_and_unload()
model.save_pretrained(merged_peft_model_name)
tokenizer.save_pretrained(merged_peft_model_name)
model.push_to_hub(merged_peft_model_name)



# pushing new model to hugging face for further download on another machine

import torch
import numpy as np
import pandas as pd
import json
import os
from vllm import LLM, SamplingParams

os.environ['CUDA_VISIBLE_DEVICES']="0"
base_model_name='meta-llama/Llama-2-7b-hf'
merged_peft_model_name="<username>/<my-peft-model>-merged-peft"
llm = LLM(model=merged_peft_model_name, tokenizer=base_model_name)

output = llm.generate("Hello World")

