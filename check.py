import tiktoken

enc = tiktoken.get_encoding("gpt2")

with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokens = enc.encode(text)

print(len(tokens))
print(tokens[:20])

print(enc.decode(tokens[:200]))