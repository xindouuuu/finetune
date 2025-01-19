# test qwen2.5-0.5
# pay attention to trl and peft version. 
# trl 0.13.0 peft 0.14.0
# qwen2.5-0.5B  https://huggingface.co/Qwen/Qwen2.5-0.5B
# DATASET  https://huggingface.co/datasets/trl-lib/Capybara

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
import json
import torch


# test model before finetuning
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel

device = "cuda" # or "cpu"
model_name = "Qwen/Qwen2.5-0.5B"
output_dir = "finetuned_model/Qwen2.5"
adapter_model_name = output_dir + "/checkpoint-6"


print("--- test qwen2.5 ...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Model loaded successfully.")

print("test model befor finetuning...")
# Generate text
prompt = "Error pattern: example_error_pattern_1, Type: example_type_1"
messages = [
    {"role": "system", "content": "You are helpful assistant to analysis cisco ip phone log. given the error pattern and type, you need to find the root cause."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=50
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(f"input_text: {prompt}")
print(f"output: {response}")

# Delete model and tokenizer to free up memory
del model
del tokenizer

# Import the garbage collector module
import gc
gc.collect()
