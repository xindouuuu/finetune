# lora finetune qwen2.5-0.5
# pay attention to trl and peft version. 
# trl 0.13.0 peft 0.14.0
# qwen2.5-0.5B  https://huggingface.co/Qwen/Qwen2.5-0.5B
# DATASET  https://huggingface.co/datasets/trl-lib/Capybara

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
import json


# test model before finetuning
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

print("Model loaded successfully.")

print("test model befor finetuning...")
# Generate text
input_text = "recommend a movie to watch"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))


print("\n\n ----------------finetuning qwen2.5-0.5B start ----------------")
print("Loading dataset...")

dataset = load_dataset("trl-lib/Capybara", split="train")
small_dataset = dataset.select(range(100))
print(json.dumps(small_dataset[0], indent=2))

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    modules_to_save=["lm_head", "embed_token"],
    task_type="CAUSAL_LM"
)

trainer = SFTTrainer(
    "Qwen/Qwen2.5-0.5B",
    train_dataset=small_dataset,
    args=SFTConfig(output_dir="finetune/Qwen2.5-0.5B-SFT"),
    peft_config=peft_config
)

trainer.train()

print("\n\n ----------------finetuning qwen2.5-0.5B end ----------------")


print("\n\n ----------------test model after finetuning ----------------")
# load model
model_finetuned = AutoModelForCausalLM.from_pretrained("finetuned-model/Qwen2.5-0.5B-SFT")
# Generate text
input_text = "recommend a movie to watch"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model_finetuned.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
