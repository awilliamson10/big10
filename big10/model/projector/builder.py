import torch.nn as nn

def build_matchup_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'matchup_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    
    raise ValueError(f'Unknown projector type: {projector_type}')
