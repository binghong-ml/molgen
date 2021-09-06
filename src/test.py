import os
from data.smilesstate import SmilesState
from tokenizers import Tokenizer

state = SmilesState("<bos> C - C <eos>".split(" "))
processed_dir = "../resource/data/zinc/yessinglebond"
tokenizer_path = os.path.join(processed_dir, "tokenizer.json")
tokenizer = Tokenizer.from_file(tokenizer_path)
print(state.featurize(tokenizer))