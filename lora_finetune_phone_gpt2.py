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
from typing import List, Tuple

class PhoneLogData(Dataset):
    def __init__(self, path: str, tokenizer):
        self.data = json.load(open(path, "r"))
        
        self.X = []
        # Format each log entry as a conversation
        for entry in self.data:
            try:
                # Updated to match JSON format
                input_text = (f"Error Pattern: {entry['input']['error_pattern']}, "
                            f"Type: {entry['input']['type']}")
                output_text = (f"Root Cause: {entry['output']['root_cause']}, "
                             f"Action Plan: {', '.join(entry['output']['action_plan'])}")
                self.X.append((input_text, output_text))
            except KeyError as e:
                print(f"Warning: Skipping malformed entry: {e}")
                continue

        # Format conversations with special tokens
        self.formatted_data = []
        for input_text, output_text in self.X:
            formatted = f"<startofstring> <user_input>: {input_text} <analysis>: {output_text} <endofstring>"
            self.formatted_data.append(formatted)
        
        print("Number of phone log entries:", len(self.formatted_data))
        print(self.formatted_data[0])
        
        # Add debug print to check token lengths
        sample_lengths = tokenizer(
            self.formatted_data,
            truncation=False,
            return_length=True
        )['length']
        print(f"Max sequence length in dataset: {max(sample_lengths)}")
        
        # Consider batch processing for large datasets
        self.tokenizer = tokenizer  # Store tokenizer
        self.max_length = 512
        
        # Store raw text instead of encoded tensors
        self.formatted_data = [text for text in self.formatted_data]  # Keep the existing formatted data

    def __len__(self):
        return len(self.formatted_data)

    def __getitem__(self, idx):
        # Encode on-the-fly instead of storing all tensors
        encoded = self.tokenizer(
            self.formatted_data[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return (encoded['input_ids'].squeeze(), encoded['attention_mask'].squeeze())

# Configuration
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = 15  # Increased epochs due to smaller dataset
BATCH_SIZE = 4  # Smaller batch size for smaller dataset
LEARNING_RATE = 5e-5
MAX_PRINTS = 5

# LoRA Configuration
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
TARGET_MODULES = ["c_attn", "c_proj"]

def infer(inp, model, tokenizer):
    if not inp or not isinstance(inp, str):
        raise ValueError("Input must be a non-empty string")
        
    try:
        # Format input EXACTLY like training data format from PhoneLogData class
        input_text = f"<startofstring> <user_input>: {inp} <analysis>: Root Cause: "
        
        # Encode the input text
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                min_length=50,
                max_length=512,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                num_return_sequences=1,
                repetition_penalty=1.2,
            )
        
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        return ' '.join(decoded.split())
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return "Error: Could not generate response"

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
        print("Test inference:", infer("Error Pattern: bluetooth_pairing_failed, Type: connectivity", model, tokenizer))
        model.train()
    
    total_time = time.time() - start_time
    print("\nTraining completed!")
    print(f"Total time: {total_time / 3600:.2f} hours ({total_time:.2f}s)")
    
    return losses




def test_data_format(data_path: str) -> Tuple[bool, List[str]]:
    """
    Test if the data file meets the expected format requirements.
    Returns (is_valid, error_messages)
    """
    errors = []
    try:
        data = json.load(open(data_path, "r"))
        
        if not isinstance(data, list):
            errors.append("Root data structure must be a list")
            return False, errors
            
        for idx, entry in enumerate(data):
            # Updated key checking to match JSON format
            if not all(key in entry for key in ['input', 'output']):
                errors.append(f"Entry {idx}: Missing required top-level keys (input/output)")
                continue
                
            # Check input format
            if not all(key in entry['input'] for key in ['error_pattern', 'type']):
                errors.append(f"Entry {idx}: Missing required input keys (error_pattern/type)")
                
            # Check output format
            if not all(key in entry['output'] for key in ['root_cause', 'action_plan']):
                errors.append(f"Entry {idx}: Missing required output keys (root_cause/action_plan)")
            
            # Check action_plan is a list
            if not isinstance(entry['output']['action_plan'], list):
                errors.append(f"Entry {idx}: action_plan must be a list")
                
        return len(errors) == 0, errors
        
    except json.JSONDecodeError:
        errors.append("Invalid JSON format")
        return False, errors
    except Exception as e:
        errors.append(f"Unexpected error: {str(e)}")
        return False, errors

def test_model_output(model, tokenizer) -> Tuple[bool, List[str]]:
    """
    Test if the model can generate valid outputs.
    Returns (is_valid, error_messages)
    """
    errors = []
    test_inputs = [
        "Error Pattern: bluetooth_pairing_failed, Type: connectivity",
        "Error Pattern: network_timeout_error, Type: connectivity"
    ]
    
    try:
        model.eval()
        for test_input in test_inputs:
            output = infer(test_input, model, tokenizer)
            
            # Check if output contains expected sections
            if "root_cause:" not in output.lower():
                errors.append(f"Missing 'root_cause' in output for input: {test_input}")
            if "action_plan:" not in output.lower():
                errors.append(f"Missing 'action_plan' in output for input: {test_input}")
                
        return len(errors) == 0, errors
        
    except Exception as e:
        errors.append(f"Model inference error: {str(e)}")
        return False, errors

def setup_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({
        "pad_token": "<pad>",
        "bos_token": "<startofstring>",
        "eos_token": "<endofstring>"
    })
    tokenizer.add_tokens(["<user_input>:"]) 
    tokenizer.add_tokens(["<analysis>:"])
    return tokenizer

def setup_model_with_lora(tokenizer):
    # Load the base GPT-2 model
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Resize token embeddings for special tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Configure LoRA parameters
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES
    )
    
    # Wrap model with LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model.to(device)

def save_progress(model, losses):
    os.makedirs('lora_model', exist_ok=True)
    os.makedirs('losses', exist_ok=True)
    
    # Save LoRA weights
    model.save_pretrained("lora_model")
    
    # Save losses
    np.savetxt(os.path.join('losses', 'lora_training_losses.txt'), losses)

if __name__ == "__main__":
    try:
        print(f"Using device: {device}")
        
        # Test data format
        print("\nTesting data format...")
        is_valid, data_errors = test_data_format("./data/phone_log_fake.json")
        if not is_valid:
            print("❌ Data validation failed:")
            for error in data_errors:
                print(f"  - {error}")
            print("Please fix the data format before continuing.")
            exit(1)
        print("✅ Data format validation passed")
        
        # Setup
        print("\nInitializing tokenizer and model...")
        tokenizer = setup_tokenizer()
        model = setup_model_with_lora(tokenizer)
        
        # Load and train
        print("\nLoading training data...")
        phone_data = PhoneLogData("./data/phone_log_fake.json", tokenizer)
        dataLoader = DataLoader(phone_data, batch_size=BATCH_SIZE, shuffle=True)
        
        # Optimizer
        optim = Adam(model.parameters(), lr=LEARNING_RATE)
        
        # First training sequence
        losses = train(model, tokenizer, dataLoader, optim)
        
        # Test model output
        print("\nTesting model output format...")
        is_valid, model_errors = test_model_output(model, tokenizer)
        if not is_valid:
            print("❌ Model validation failed:")
            for error in model_errors:
                print(f"  - {error}")
            print("Please check the model configuration.")
            exit(1)
        print("✅ Model output validation passed")
        
        # Redundant data loading and training
        print("\nLoading training data...")  # This is redundant
        phone_data = PhoneLogData("./data/phone_log_fake.json", tokenizer)  # Redundant
        dataLoader = DataLoader(phone_data, batch_size=BATCH_SIZE, shuffle=True)  # Redundant
        optim = Adam(model.parameters(), lr=LEARNING_RATE)  # Redundant
        losses = train(model, tokenizer, dataLoader, optim)  # Redundant training
        
        print("Test finetuned model")
        print("Enter error details (type 'q' to quit):")
        while True:
            inp = input("Error Description: ")
            if inp.lower() == 'q':
                print("Goodbye!")
                break
            response = infer(inp, model, tokenizer)
            print("Diagnosis Bot:", response)
            print("-" * 50) 
        
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 