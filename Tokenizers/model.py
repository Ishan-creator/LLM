import random
import torch
import numpy as np
import pandas as pd
import json
from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders, trainers
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers import SentencePieceBPETokenizer

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Character-level tokenizer
class CharacterTokenizer:
    def __init__(self):
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.normalizer = normalizers.NFKC()
        self.tokenizer.pre_tokenizer = pre_tokenizers.CharDelimiterSplit(delimiter=' ')
        self.tokenizer.decoder = decoders.BPEDecoder()
    
    def train(self, files):
        trainer = trainers.BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        self.tokenizer.train(files, trainer)
    
    def tokenize(self, text):
        return self.tokenizer.encode(text).tokens

# Subword tokenizers
class SubwordTokenizer:
    def __init__(self, model_type='bpe'):
        self.model_type = model_type
        if model_type == 'bpe':
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            self.trainer = trainers.BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        elif model_type == 'wordpiece':
            self.tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
            self.trainer = trainers.WordPieceTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        elif model_type == 'unigram':
            self.tokenizer = Tokenizer(Unigram())
            self.trainer = trainers.UnigramTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        elif model_type == 'sentencepiece':
            self.tokenizer = SentencePieceBPETokenizer()
            self.trainer = None  # SentencePieceBPETokenizer uses a different training method
        else:
            raise ValueError("Unsupported model type")
        
        self.tokenizer.normalizer = normalizers.NFKC()
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.tokenizer.decoder = decoders.WordPiece()
    
    def train(self, files):
        if isinstance(self.tokenizer, SentencePieceBPETokenizer):
            self.tokenizer.train(files, vocab_size=30000, min_frequency=2, special_tokens=["<pad>", "<unk>", "<s>", "</s>"])
        else:
            self.tokenizer.train(files, self.trainer)
    
    def tokenize(self, text):
        return self.tokenizer.encode(text).tokens

def load_data(file_path):
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
    elif file_path.endswith('.csv'):
        data = pd.read_csv(file_path).iloc[:, 0].tolist()
    else:
        raise ValueError("Unsupported file format")
    return data

def save_to_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

class TokenizerPipeline:
    def __init__(self, tokenizer, tokenizer_name):
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name

    def run(self, train_files, input_file):
        data = load_data(input_file)
        
        self.tokenizer.train(train_files)
        
        tokenized_data = {self.tokenizer_name: [self.tokenizer.tokenize(text) for text in data]}
        
        return tokenized_data

if __name__ == "__main__":
    file_path = "/home/ishan-pc/Desktop/Ishan-Github/LLM/Tokenizers/data/introduction.txt"
    output_path = "output/tokenized_data_english.json"
    
    all_tokenized_data = {}
    
    char_tokenizer = CharacterTokenizer()
    char_pipeline = TokenizerPipeline(char_tokenizer, "CharacterTokenizer")
    all_tokenized_data.update(char_pipeline.run([file_path], file_path))
    
    for model_type in ['bpe', 'wordpiece', 'unigram', 'sentencepiece']:
        subword_tokenizer = SubwordTokenizer(model_type=model_type)
        subword_pipeline = TokenizerPipeline(subword_tokenizer, model_type.capitalize() + "Tokenizer")
        all_tokenized_data.update(subword_pipeline.run([file_path], file_path))
    
    save_to_json(all_tokenized_data, output_path)
