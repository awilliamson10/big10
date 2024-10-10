import torch.nn as nn
from transformers import GPT2LMHeadModel

class GPT2MatchupEncoder(nn.Module):
    def __init__(self, config, delay_loading=False, **kwargs):
        super().__init__()
        self.config = config
        self.model = None
        if not delay_loading:
            self.load_model()

    def is_loaded(self):
        return self.model is not None
        
    def load_model(self):
        self.model = GPT2LMHeadModel(self.config)

    def forward(self, input_ids, attention_mask=None):
        if not self.is_loaded():
            self.load_model()
        return self.model(input_ids, attention_mask=attention_mask)
    
    def save(self, path):
        self.model.save_pretrained(path)
        
        
