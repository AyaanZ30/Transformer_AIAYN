from pathlib import Path
from fsspec.core import conf
import torch
import torch.nn as nn 
from torch.utils.data import random_split, DataLoader, Dataset 
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from dataset import BilingualDataset

def get_all_sentences(dataset, language):
    for item in dataset:
        sentences = item['translation']
        yield sentences[language]

def build_tokenizer(config, dataset, language):
    tokenizer_path = Path(config['tokenizer_file'].format(language))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(vocab=None, unk_token = '[UNK]'))   
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer  

def get_dataset(config):
    raw_dataset = load_dataset('Helsinki-NLP/opus_books', f'{config['lang_src']}-{config['lang_target']}', split = 'train')

    # build tokenizers for both src and target lang vocab in the dataset
    tokenizer_src_lang = build_tokenizer(config, raw_dataset, config['lang_src'])  
    tokenizer_target_lang = build_tokenizer(config, raw_dataset, config['lang_target'])  

    # 90 percent data for training and 10 percent for validation
    train_data_size = int(0.9 * len(raw_dataset))
    val_data_size = len(raw_dataset) - train_data_size

    train_data_raw, val_data_raw = random_split(raw_dataset, [train_data_size, val_data_size])
    
    # building actual training data consisting of both src and target language text for supervised training
    bilingual_train_dataset = BilingualDataset(
        train_data_raw, tokenizer_src_lang, tokenizer_target_lang,
        config['lang_src'], config['lang_target'], config['seq_len'] 
    )
    bilingual_val_dataset = BilingualDataset(
        val_data_raw, tokenizer_src_lang, tokenizer_target_lang,
        config['lang_src'], config['lang_target'], config['seq_len'] 
    )

    max_seq_len_src = 0
    max_seq_len_target = 0

    for item in raw_dataset:
        src_ids = tokenizer_src_lang.encode(item['translation'][config['lang_src']]).ids
        target_ids = tokenizer_target_lang.encode(item['translation'][config['lang_target']]).ids

        max_seq_len_src = max(len(src_ids), max_seq_len_src)
        max_seq_len_target = max(len(target_ids), max_seq_len_target)
    
    print(f'Max length for source language sentence : {max_seq_len_src} ({config['lang_src']})')
    print(f'Max length for target language sentence : {max_seq_len_src} ({config['lang_target']})')

    train
        

    



