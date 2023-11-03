import tiktoken
from transformers import GPT2Tokenizer

enc = tiktoken.get_encoding("cl100k_base")
res=enc.encode("https://www.wikidata.org/wiki/Q7691083421094823094710")
#res=enc.encode("https://dbpedia.org/resource/Edoardo_Affini")
singletokens=[enc.decode_single_token_bytes(token) for token in res]
print(singletokens)
print(enc)
tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
e=tokenizer.tokenize("https://www.wikidata.org/wiki/Q7691083421094823094710")
out=tokenizer.decode(e)

print(res)

