from big10.model.encoder import GPT2MatchupEncoder

def build_matchup_tower(config, **kwargs):
    return GPT2MatchupEncoder(config.encoder_config, **kwargs)