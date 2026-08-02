import numpy as np
import tiktoken

with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()


encoding = tiktoken.get_encoding("gpt2")

ids = np.array(encoding.encode(text), dtype=np.int32)

print(f"Total tokens: {len(ids)}")
print(f"First 20 tokens: {ids[:20]}")

print(f"Decoded first 200 tokens: {encoding.decode(ids[:20])}")