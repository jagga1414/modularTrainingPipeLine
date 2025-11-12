import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_tokenizer():
    torch.random.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
    
    tokenizer.bos_token = "<s>"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(["<|user|>", "<end>", "<|end|>", "<|assistant|>"])
    
    tokenizer.chat_template = """{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|endoftext|>'}}{% endif %}{% endfor %}"""
    
    return tokenizer


def load_model():
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
    return model
