from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import torch
import numpy as np
import argparse
from torch.utils.data import Dataset
import json
import os
import time


class ChatData(Dataset):
    '''
    This class is used to load the chat data and prepare it for training.
    '''
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

        # self.X = self.X[:10]
        print("Number of conversations in X:", len(self.X))
        print("\nFirst 5 conversation examples:")
        for i in range(min(5, len(self.X))):
            print(f"\nExample {i+1}:")
            print(self.X[i])
            print("-" * 50)
        print("\n")
        
        self.X_encoded = tokenizer(self.X,max_length=100, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])
    


# 1. First define all configuration/setup variables
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = 12
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_PRINTS = 5

def setup_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({
        "pad_token": "<pad>",
        "bos_token": "<startofstring>",
        "eos_token": "<endofstring>"
    })
    tokenizer.add_tokens(["<bot>:"])
    return tokenizer

def setup_model(tokenizer):
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    return model.to(device)

def infer(inp, model, tokenizer):
    # Format input with only one <bot>: token
    input_text = f"<startofstring> {inp} <bot>:"
    inp = tokenizer(input_text, return_tensors="pt").to(device)
    
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
        # Removed early_stopping to fix the warning
    )
    
    # Clean up the output by removing special tokens
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Remove the input prompt from the response
    if decoded.startswith(input_text):
        decoded = decoded[len(input_text):].strip()
    
    # Clean up extra spaces
    decoded = ' '.join(decoded.split())
    
    return decoded

def train(dataLoader, model, optim, tokenizer):
    losses = []
    print_count = 0
    print_shape = True
    start_time = time.time()

    for epoch in tqdm.tqdm(range(EPOCHS)):
        epoch_start_time = time.time()
        epoch_losses = []
        print(f"\n\nEpoch {epoch} started")
        
        for X, attention_mask in dataLoader:
            loss = train_step(X, attention_mask, model, optim)
            epoch_losses.append(loss)
            
            if print_shape:
                print(f"X shape: {X.shape}, attention_mask shape: {attention_mask.shape}")
                print_shape = False

            if print_count < MAX_PRINTS:
                print_debug_info(X, attention_mask, tokenizer)
                print_count += 1
        
        # Post-epoch processing
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        save_progress(model, losses)
        
        print(f"Average loss for epoch {epoch}: {avg_loss:.4f}")
        print(f"Epoch duration: {epoch_duration:.2f} seconds")
        
        # Validation inference
        model.eval()
        print(infer("how are you", model, tokenizer))
        model.train()
        print("\n")

    total_time = time.time() - start_time
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60
    print(f"\nTotal training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

def train_step(X, attention_mask, model, optim):
    X = X.to(device)
    attention_mask = attention_mask.to(device)
    
    optim.zero_grad()
    loss = model(X, attention_mask=attention_mask, labels=X).loss
    loss.backward()
    optim.step()
    
    return loss.item()

def print_debug_info(X, attention_mask, tokenizer):
    full_text = tokenizer.decode(X[0], skip_special_tokens=False)
    print("Full text:", full_text)
    
    active_positions = attention_mask[0].nonzero().squeeze()
    active_tokens = X[0][active_positions]
    decoded_active = tokenizer.decode(active_tokens, skip_special_tokens=False)
    print("Active tokens:", decoded_active)
    print("-" * 50)

def save_progress(model, losses):
    # Create folders if they don't exist
    os.makedirs('losses', exist_ok=True)
    os.makedirs('finetuned_model', exist_ok=True)
    
    # Save losses
    losses_path = os.path.join('losses', 'SFT_training_losses.txt')
    np.savetxt(losses_path, losses)
    
    # Save model state
    model_path = os.path.join('finetuned_model', 'SFT_model_state.pt')
    torch.save(model.state_dict(), model_path)

def load_and_test_model(model_path="finetuned_model/SFT_model_state.pt"):
    """
    Loads a saved model and runs an interactive test session
    """
    print(f"Loading model from {model_path}")
    
    # Setup model and tokenizer
    tokenizer = setup_tokenizer()
    model = setup_model(tokenizer)
    
    # Load the saved state
    try:
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Interactive testing
    print("\nModel loaded and ready for testing!")
    print("Enter your message (type 'q' to quit):")
    while True:
        inp = input("You: ")
        if inp.lower() == 'q':
            print("Goodbye!")
            break
        response = infer(inp, model, tokenizer)
        print("Bot:", response)
        print("-" * 50)


def extract_bot_response(text):
    """
    Extracts the bot response from the formatted string.
    Format: <startofstring> user_input <bot>: bot_response <endofstring>
    """
    try:
        # Find the response after '<bot>:'
        bot_marker = '<bot>:'
        bot_response = text.split(bot_marker)[1]
        # Remove <endofstring> and trim
        bot_response = bot_response.replace('<endofstring>', '').strip()
        return bot_response
    except:
        return "Error: Could not extract bot response"

# Main execution
if __name__ == "__main__":
    print(f"Using device: {device}")
    
    # Add argument parsing
    mode = "train"
    
    if mode == 'train':
        # Existing training code
        tokenizer = setup_tokenizer()
        model = setup_model(tokenizer)
        
        # Initial test
        print("Test inference from base GPT-2:")
        input_tokens = tokenizer("how are you", return_tensors="pt").to(device)
        print(f"Raw output: {tokenizer.decode(model.generate(**input_tokens)[0])}")
        print("\n")
        
        # Training
        chatData = ChatData("./data/chat_data.json", tokenizer)
        dataLoader = DataLoader(chatData, batch_size=BATCH_SIZE)
        model.train()
        optim = Adam(model.parameters(), lr=LEARNING_RATE)
        
        print("Training...")
        train(dataLoader, model, optim, tokenizer)
        
        # Interactive inference
        print("Enter your message (type 'q' to quit):")
        while True:
            inp = input("You: ")
            if inp.lower() == 'q':
                print("Goodbye!")
                break
            response = infer(inp, model, tokenizer)
            print("Raw ouput Bot:", response)
            extracted_response = extract_bot_response(response)
            print("Extracted Bot:", extracted_response)
            print("-" * 50)
    else:
        # Test mode
        # load_and_test_model()
        print("test")