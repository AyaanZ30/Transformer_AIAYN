import torch
import torch.nn as nn 
from torch.utils import data
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, 
    dataset, 
    tokenizer_src_lang, 
    tokenizer_target_lang, 
    src_lang,
    target_lang,
    seq_len,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.tokenizer_src_lang = tokenizer_src_lang
        self.tokenizer_target_lang = tokenizer_target_lang
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.seq_len = seq_len

        self.sos_token = torch.Tensor([tokenizer_src_lang.token_to_id(['[SOS]'])], torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src_lang.token_to_id(['[EOS]'])], torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src_lang.token_to_id(['[PAD]'])], torch.int64)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        src_target_pair = self.dataset[index]
        src_text = src_target_pair['translation'][self.src_lang]
        target_text = src_target_pair['translation'][self.target_lang]

        # finally general code for obtaining numbers corresponding to each token(word) in a particular sentence (src as well sa target lang) in the dataset
        encoder_input_tokens = self.tokenizer_src_lang.encode(src_text).ids  
        decoder_input_tokens = self.tokenizer_target_lang.encode(target_text).ids

        # pad sentences to equal fixed seq lengths (models pre-requisite)
        encoder_num_padding_tokens = self.seq_len - len(encoder_input_tokens) - 2   # -2 being SOS and EOS tokens included  
        decoder_num_padding_tokens = self.seq_len - len(decoder_input_tokens) - 1   # -1 being only SOS token (as we are decoding from start till we reach the EOS token)   

        if encoder_num_padding_tokens < 0 or decoder_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')
        
        # Hence encoder input : <SOS>, Hello, world, <EOS>, <PAD>, <PAD>, <PAD>,...,<PAD> (till seq_len is reached) 
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(encoder_input_tokens, torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * encoder_num_padding_tokens, torch.int64)   
            ]
        )

        # Similarly decoder input : <SOS>, Si, Wolla, <PAD>, ...,<PAD> (till seq_len reached)
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(decoder_input_tokens, torch.int64),
                torch.tensor([self.pad_token] * decoder_num_padding_tokens, torch.int64)   
            ]
        )
    
        # only add EOS to the label (what we expect as output from the decoder)
        label = torch.cat(
            [
                torch.tensor(decoder_input_tokens, torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * decoder_num_padding_tokens, torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input" : encoder_input,    # size : seq_len
            "decoder_input" : decoder_input,    # size : seq_len
            # we dont want the extra padding tokens used to participate in the self attention
            "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),    # to add sequence dimension as well as batch dimension (1, 1, seq_len)
            # casual mask to ensure that only recent and previous tokens take part in self attention (to avoid peeking into future tokens)
            "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label" : label,
            "src_text" : src_text,
            "target_text" : target_text,
        }

def causal_mask(size):
    # the below line will retunr all values above diagonal of the self attnetion computed matrix (the future tokens for corresponding present token) and set them to 0
    mask = torch.triu(input = torch.ones(1, size, size), diagonal = 1).type(torch.int)
    return mask == 0

