# lora finetune qwen2.5-0.5
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


print("Loading dataset...")

# dataset_capybara = load_dataset("trl-lib/Capybara", split="train")
# small_dataset_capybara = dataset_capybara.select(range(100))
# print(type(small_dataset_capybara))
# print(json.dumps(small_dataset_capybara[0], indent=2))

# Load dataset from JSON file
dataset = load_dataset('json', data_files='data/phone_log_fake.json', split='train')

print(type(dataset))
print(json.dumps(dataset[0], indent=2))
print(dataset[0].keys())
exit()

device = "cuda" # or "cpu"
model_name = "Qwen/Qwen2.5-0.5B"
output_dir = "finetuned_model/Qwen2.5-0.5B"
adapter_model_name = output_dir + "/checkpoint-6"


print("\n\n ----------------finetuning qwen2.5 start ----------------")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    modules_to_save=["lm_head", "embed_token"],
    task_type="CAUSAL_LM"
)

trainer = SFTTrainer(
    model_name,
    train_dataset=dataset,
    args=SFTConfig(output_dir=output_dir, num_train_epochs=12),
    peft_config=peft_config,
)

trainer.train()



print("\n\n ----------------finetuning qwen2.5 end ----------------")
