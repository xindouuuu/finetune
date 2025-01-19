# test lora finetune qwen2.5-0.5
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


device = "cpu" # or "cpu"
model_name = "Qwen/Qwen2.5-0.5B"
output_dir = "finetuned_model/Qwen2.5"
adapter_model_name = output_dir + "/checkpoint-144"


print("\n\n ----------------test model after finetuning ----------------")

# Free up GPU memory


model_finetuned = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model_finetuned = PeftModel.from_pretrained(model_finetuned, adapter_model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the JSON data from the file
with open('data/phone_log_fake.json', 'r') as file:
    data = json.load(file)

# Iterate over the first 10 user inputs
for i, entry in enumerate(data[:10]):
    user_input = entry['messages'][1]['content']  # Extract the user input message
    print(f"Testing input {i+1}: {user_input}")

    # Prepare the input for the model
    messages = [
        {"role": "system", "content": "You are helpful assistant to analysis cisco ip phone log. given the error pattern and type, you need to find the root cause."},
        {"role": "user", "content": user_input}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model_finetuned.device)
    generated_ids = model_finetuned.generate(
        **model_inputs,
        max_new_tokens=50
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Output: {response}\n")
    print("----------------------------------------")

print("\n\n ----------------test model after finetuning end ----------------")

