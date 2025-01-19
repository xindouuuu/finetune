from transformers import GPT2ForSequenceClassification, GPT2Tokenizer
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import torch
import numpy as np
import tqdm
import os
import time
import json

# Configuration
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = 12
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
MAX_PRINTS = 5

# LoRA Configuration
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
TARGET_MODULES = ["c_attn", "c_proj"]

def download_and_save_dataset(split="train"):
    """Download dataset and save to local file"""
    dataset = load_dataset("dair-ai/emotion", split=split)
    
    # Convert to list of dictionaries
    data_to_save = [
        {"text": item["text"], "label": item["label"]} 
        for item in dataset
    ]
    
    # Save to JSON file
    filename = f"data/emotion_dataset_{split}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    
    print(f"Dataset saved to {filename}")
    return filename

class EmotionData(Dataset):
    def __init__(self, tokenizer, split="train"):
        filename = f"data/emotion_dataset_{split}.json"
        
        # Download and save dataset if it doesn't exist
        if not os.path.exists(filename):
            filename = download_and_save_dataset(split)
        
        # Load from local file
        with open(filename, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
        
        # Convert texts to input format
        self.X = [item["text"] for item in self.dataset]
        self.labels = [item["label"] for item in self.dataset]
        
        # Tokenize all texts
        self.X_encoded = tokenizer(
            self.X, 
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.labels[idx]
        )

def setup_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # Add padding token and set it as the default
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding
    return tokenizer

def setup_model_with_lora(tokenizer):
    # Load base model for sequence classification
    model = GPT2ForSequenceClassification.from_pretrained(
        "gpt2",
        num_labels=6,  # Number of emotion classes
        pad_token_id=tokenizer.pad_token_id  # Set the pad token ID
    )
    
    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES
    )
    
    model = get_peft_model(model, peft_config)
    model.config.pad_token_id = tokenizer.pad_token_id  # Ensure pad token is set in config
    model.print_trainable_parameters()
    return model.to(device)

def save_progress(model, losses):
    """Save model checkpoints and training losses"""
    # Create directories if they don't exist
    os.makedirs('finetuned_model', exist_ok=True)
    os.makedirs('losses', exist_ok=True)
    
    # Save LoRA weights and configuration
    model.save_pretrained("finetuned_model")
    
    # Save model state dictionary separately
    torch.save(model.state_dict(), os.path.join('finetuned_model', 'lora_model_emotion.bin'))
    
    # Save losses
    np.savetxt(os.path.join('losses', 'emotion_training_losses.txt'), losses)

def infer(text, model, tokenizer):
    """Inference function for emotion classification"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
    
    # Map predictions to emotion labels
    emotion_map = {
        0: "sadness",
        1: "joy", 
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    }
    return emotion_map[predictions[0].item()]

def train(model, tokenizer, dataLoader, optim):
    """Training loop for emotion classification"""
    model.train()
    losses = []
    start_time = time.time()
    
    for epoch in tqdm.tqdm(range(EPOCHS)):
        epoch_losses = []
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(dataLoader):
            # Print first few examples of the first batch in each epoch
            if batch_idx == 0 and epoch == 0:
                print("\nExample inputs from first batch:")
                for i in range(min(3, len(input_ids))):
                    decoded_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    print(f"Input {i}:")
                    print(f"Text: {decoded_text}")
                    print(f"Label: {labels[i].item()}")
                    print("-" * 50)
            
            # Move data to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            # Training step
            optim.zero_grad()
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optim.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        epoch_time = time.time() - epoch_start
        print(f"Average loss: {avg_loss:.4f} | Epoch time: {epoch_time:.2f}s")
        total_time = time.time() - start_time
        print(f"Total time: {total_time / 3600:.2f} hours ({total_time:.2f}s)")
        
        # Save checkpoint and test inference after each epoch
        save_progress(model, losses)
        model.eval()
        print("Test inference:", infer("I am feeling great today!", model, tokenizer))
        model.train()
    
    total_time = time.time() - start_time
    print("\nTraining completed!")
    print(f"Total time: {total_time / 3600:.2f} hours ({total_time:.2f}s)")
    
    return losses

def load_trained_model(tokenizer):
    """Load the trained LoRA model"""
    # Load base model for sequence classification
    model = GPT2ForSequenceClassification.from_pretrained(
        "gpt2",
        num_labels=6,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=True,  # Set to True for inference
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES
    )
    
    model = get_peft_model(model, peft_config)
    # Load trained LoRA weights
    model.load_state_dict(torch.load("finetuned_model/lora_model_emotion.bin"))
    return model.to(device)

if __name__ == "__main__":
    print(f"Using device: {device}")
    
    # Setup
    tokenizer = setup_tokenizer()
    
    # Check if we want to train or test
    mode = input("Enter 'train' for training or 'test' for testing: ").lower()
    
    if mode == "train":
        model = setup_model_with_lora(tokenizer)
        # Print parameter details
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"trainable params: {trainable_params:,d} || all params: {all_param:,d} "
              f"|| trainable%: {100 * trainable_params / all_param:.2f}%")
        
        # Load data
        train_data = EmotionData(tokenizer, split="train")
        dataLoader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        
        # Optimizer
        optim = Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Training loop
        losses = train(model, tokenizer, dataLoader, optim)
        
        print("\nTesting emotion classification model")
        print("Enter your text (type 'q' to quit):")
        while True:
            text = input("You: ")
            if text.lower() == 'q':
                print("Goodbye!")
                break
            emotion = infer(text, model, tokenizer)
            print(f"Predicted emotion: {emotion}")
            print("-" * 50)
    
    elif mode == "test":
        print("Loading trained model...")
        model = load_trained_model(tokenizer)
        model.eval()
        
        print("\nTesting emotion classification model")
        print("Enter your text (type 'q' to quit):")
        while True:
            text = input("You: ")
            if text.lower() == 'q':
                print("Goodbye!")
                break
            emotion = infer(text, model, tokenizer)
            print(f"Predicted emotion: {emotion}")
            print("-" * 50)
    
    else:
        print("Invalid mode selected. Please choose 'train' or 'test'.") 