from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import tqdm
import json
import os
import time

# Reuse your existing ChatData class
class ChatData(Dataset):
    def __init__(self, path:str, tokenizer):
        self.data = json.load(open(path, "r"))
        
        self.X = []
        for i in self.data:
            for j in i['dialog']:
                self.X.append(j['text'])

        for idx, i in enumerate(self.X):
            try:
                self.X[idx] = "<startofstring> "+i+" <bot>: "+self.X[idx+1]+" <endofstring>"
            except:
                break
        
        print("Number of conversations in X:", len(self.X))
        self.X_encoded = tokenizer(self.X, max_length=100, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])

# Configuration
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = 12
BATCH_SIZE = 64
LEARNING_RATE = 1e-4  # Lower learning rate for LoRA
MAX_PRINTS = 5

# LoRA Configuration
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
TARGET_MODULES = ["c_attn", "c_proj"]  # Target attention layers in GPT-2

def setup_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({
        "pad_token": "<pad>",
        "bos_token": "<startofstring>",
        "eos_token": "<endofstring>"
    })
    tokenizer.add_tokens(["<bot>:"])
    return tokenizer

def setup_model_with_lora(tokenizer):
    # Load the base GPT-2 model from HuggingFace hub
    # This is the foundation model that we'll be fine-tuning
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Resize the token embeddings to account for any special tokens we added
    # This ensures our model can handle our custom tokens like <bot>:, <startofstring>, etc.
    model.resize_token_embeddings(len(tokenizer))
    
    # Configure LoRA (Low-Rank Adaptation) parameters
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,      # Specify this is for causal language modeling
        inference_mode=False,               # Set to False during training
        r=LORA_R,                          # Rank of the update matrices (smaller = more efficient)
        lora_alpha=LORA_ALPHA,             # Scaling factor for the LoRA updates
        lora_dropout=LORA_DROPOUT,         # Dropout probability for LoRA layers
        target_modules=TARGET_MODULES      # Which layers to apply LoRA to (attention layers in this case)
    )
    
    # Wrap the model with LoRA using PEFT
    # This adds trainable LoRA layers while freezing the base model parameters
    model = get_peft_model(model, peft_config)
    
    # Print info about trainable vs frozen parameters
    model.print_trainable_parameters()
    
    # Move model to the appropriate device (CPU/GPU/MPS)
    return model.to(device)

def save_progress(model, losses):
    os.makedirs('lora_model', exist_ok=True)
    os.makedirs('losses', exist_ok=True)
    
    # Save LoRA weights
    model.save_pretrained("lora_model")
    
    # Save losses
    np.savetxt(os.path.join('losses', 'lora_training_losses.txt'), losses)

def infer(inp, model, tokenizer):
    input_text = f"<startofstring> {inp} <bot>:"
    inp = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inp,
            min_length=5,
            max_length=50,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            top_k=50,
            no_repeat_ngram_size=2,
            num_return_sequences=1,
            repetition_penalty=1.5,
        )
    
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    if decoded.startswith(input_text):
        decoded = decoded[len(input_text):].strip()
    return ' '.join(decoded.split())

def train(model, tokenizer, dataLoader, optim):
    """
    Main training function that handles the entire training loop
    
    Args:
        model: The LoRA-configured model
        tokenizer: GPT2 tokenizer
        dataLoader: DataLoader containing the chat data
        optim: Optimizer for training
    
    Returns:
        list: Training losses per epoch
    """
    model.train()
    losses = []
    start_time = time.time()
    
    for epoch in tqdm.tqdm(range(EPOCHS)):
        epoch_losses = []
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        
        for X, attention_mask in dataLoader:
            # Move data to device
            X = X.to(device)
            attention_mask = attention_mask.to(device)
            
            # Training step
            optim.zero_grad()
            outputs = model(X, attention_mask=attention_mask, labels=X)
            loss = outputs.loss
            loss.backward()
            optim.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        epoch_time = time.time() - epoch_start
        print(f"Average loss: {avg_loss:.4f} | Epoch time: {epoch_time:.2f}s")
        
        # Save checkpoint and test inference
        save_progress(model, losses)
        model.eval()
        print("Test inference:", infer("how are you", model, tokenizer))
        model.train()
    
    total_time = time.time() - start_time
    print("\nTraining completed!")
    print(f"Total time: {total_time / 3600:.2f} hours ({total_time:.2f}s)")
    
    return losses

if __name__ == "__main__":
    print(f"Using device: {device}")
    
    # Setup
    tokenizer = setup_tokenizer()
    model = setup_model_with_lora(tokenizer)
    
    # Print parameter details explicitly
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params:,d} || all params: {all_param:,d} "
          f"|| trainable%: {100 * trainable_params / all_param:.2f}%")
    
    # Load data
    chatData = ChatData("./data/chat_data.json", tokenizer)
    dataLoader = DataLoader(chatData, batch_size=BATCH_SIZE)
    
    # Optimizer
    optim = Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    losses = train(model, tokenizer, dataLoader, optim)
    
    print("test finetuned model")
    # Interactive inference
    print("Enter your message (type 'q' to quit):")
    while True:
        inp = input("You: ")
        if inp.lower() == 'q':
            print("Goodbye!")
            break
        response = infer(inp, model, tokenizer)
        print("Raw ouput Bot:", response)
        print("-" * 50)
