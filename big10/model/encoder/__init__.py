import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel

class MatchupEncoder(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        # Embedding layers
        self.season_type_embed = nn.Embedding(3, 16)
        self.boolean_embed = nn.Embedding(2, 8)
        self.venue_embed = nn.Embedding(config.num_venues, 32)
        self.team_embed = nn.Embedding(config.num_teams, 64)
        self.weather_condition_embed = nn.Embedding(config.num_weather_conditions, 16)
        self.conference_embed = nn.Embedding(config.num_conferences, 32)

        # Linear layer for numerical data
        self.numerical_encoder = nn.Linear(config.num_numerical_features, 128)

        # Calculate total embedding dimension
        total_embedding_dim = (16 + 8 * 3 + 32 + 64 * 2 + 16 + 32 * 2 + 128)
        
        # Final projection to model dimension
        self.final_projection = nn.Linear(total_embedding_dim, config.hidden_size)

    def load_model(self):
        pass

    def is_loaded(self):
        return True

    def forward(self, game_input):
        # Unpack the game input
        (week, season_type, neutral_site, conference_game, venue_id, home_id, away_id,
         home_pregame_elo, away_pregame_elo, game_indoors, temperature, humidity,
         wind_speed, weather_condition, home_win_prob, home_spread,
         home_conference, *numerical_features) = game_input

        # Process categorical data
        season_type_emb = self.season_type_embed(season_type)
        neutral_site_emb = self.boolean_embed(neutral_site)
        conference_game_emb = self.boolean_embed(conference_game)
        venue_emb = self.venue_embed(venue_id)
        home_team_emb = self.team_embed(home_id)
        away_team_emb = self.team_embed(away_id)
        game_indoors_emb = self.boolean_embed(game_indoors)
        weather_condition_emb = self.weather_condition_embed(weather_condition)
        home_conference_emb = self.conference_embed(home_conference)
        away_conference_emb = self.conference_embed(numerical_features[0])

        # Process numerical data
        numerical_data = torch.tensor([week, home_pregame_elo, away_pregame_elo, temperature,
                                       humidity, wind_speed, home_win_prob, home_spread] + numerical_features[1:],
                                      dtype=torch.float32)
        numerical_encoded = self.numerical_encoder(numerical_data)

        # Combine all features
        combined = torch.cat([
            season_type_emb, neutral_site_emb, conference_game_emb, venue_emb,
            home_team_emb, away_team_emb, game_indoors_emb, weather_condition_emb,
            home_conference_emb, away_conference_emb, numerical_encoded
        ], dim=-1)

        # Project to model dimension
        output = self.final_projection(combined)

        return output