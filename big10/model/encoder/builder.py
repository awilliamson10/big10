from big10.model.encoder import MatchupEncoder

def build_matchup_tower(config, **kwargs):
    matchup_tower = getattr(config, 'matchup_tower')

    if matchup_tower == 'encoder':
        return MatchupEncoder(config, **kwargs)
    
    raise ValueError(f'Unknown matchup tower type: {matchup_tower}')