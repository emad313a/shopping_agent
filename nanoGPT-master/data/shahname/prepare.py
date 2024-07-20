import os
import pickle
import requests
import numpy as np
from transformers import AutoTokenizer

# Set file paths
input_file_path = os.path.join(os.path.dirname(__file__), 'C:/Mozaffar/project/pythonProject/.venv/demo/GPTnano/nanoGPT-master/shahname/outputdadma.txt')
train_file_path = os.path.join(os.path.dirname(__file__), 'train.bin')
val_file_path = os.path.join(os.path.dirname(__file__), 'val.bin')
meta_file_path = os.path.join(os.path.dirname(__file__), 'meta.pkl')

# Read the input file
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# Create train and validation splits
n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "CohereForAI/aya-23-35B",
    use_auth_token="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")

# Encode data
train_ids = tokenizer.encode(train_data, add_special_tokens=False)
val_ids = tokenizer.encode(val_data, add_special_tokens=False)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Save encoded data to binary files
train_ids = np.array(train_ids, dtype=np.uint32)
val_ids = np.array(val_ids, dtype=np.uint32)
train_ids.tofile(train_file_path)
val_ids.tofile(val_file_path)

# Save metadata
meta = {
    'vocab_size': tokenizer.vocab_size,
    'tokenizer': tokenizer,
}
with open(meta_file_path, 'wb') as f:
    pickle.dump(meta, f)

# Optionally, you can uncomment and use this code to print token information
"""
word_to_tokens = {}
words = train_data.split()

for word in words:
    tokens = tokenizer.encode(word, add_special_tokens=False)
    word_to_tokens[word] = tokens

flag = 0
for word, tokens in word_to_tokens.items():
    token_str = ', '.join(map(str, tokens))
    decoded_tokens = [tokenizer.decode([token]) for token in tokens]
    decoded_str = ', '.join(decoded_tokens)
    print(f"Word: '{word}' -> Tokens: [{token_str}] -> Decoded Tokens: [{decoded_str}]")
    flag += 1
    if flag == 10:
        break
"""