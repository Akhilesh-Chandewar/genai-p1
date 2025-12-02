import tiktoken

text = "Hello world"

enc = tiktoken.encoding_for_model("gpt-4o")

tokens = enc.encode(text)

print("Token", tokens)

decoded_text = enc.decode(tokens)
print("Decoded" , decoded_text)
