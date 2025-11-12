import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import GenerationConfig
from dataset.dataset import Batch_D
from model.model_loader import setup_tokenizer, load_model


# Setup tokenizer
tokenizer = setup_tokenizer()

# Load model and weights
model = load_model()
model.load_state_dict(torch.load("model.pt", map_location="cpu"))

# Setup config
config = Batch_D()
config.device = "cpu"
model.to(config.device)

# Generation config
generation_config = GenerationConfig(
    max_new_tokens=100, 
    do_sample=True, 
    eos_token_id=tokenizer.eos_token_id, 
    pad_token_id=tokenizer.pad_token_id
)


def demo(query):
    messages = [
        {"role": "user", "content": query}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt", return_attention_mask=False).to(config.device)
    outputs = model.generate(**inputs, generation_config=generation_config)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return answer.split("<|assistant|>\n")[1]


# Test query
query = "I want to change my shipping address."
response = demo(query)
print(f"Query: {query}")
print(f"Response: {response}")
