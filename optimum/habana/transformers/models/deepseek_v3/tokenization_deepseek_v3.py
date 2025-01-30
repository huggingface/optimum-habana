"""
Copied from https://cdn.deepseek.com/api-docs/deepseek_v3_tokenizer.zip
"""

# pip3 install transformers
# python3 deepseek_tokenizer.py
import transformers

chat_tokenizer_dir = "./"

tokenizer = transformers.AutoTokenizer.from_pretrained( 
        chat_tokenizer_dir, trust_remote_code=True
        )

result = tokenizer.encode("Hello!")
print(result)