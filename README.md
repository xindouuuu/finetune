python 3.10
# Using venv
python -m venv llama_env

# Activate the environment
# On Windows:
llama_env\Scripts\activate
# On Unix or MacOS:
source llama_env/bin/activate



#install dependencies
pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7

#run the script
python lora_finetune_phone_llama.py