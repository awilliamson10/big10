from transformers import PreTrainedModel, PretrainedConfig
import torch
import torch.nn as nn

class MatchupEncoderConfig(PretrainedConfig):
    model_type = "matchup_encoder"

    def __init__(self, num_venues=100, num_teams=32, num_weather_conditions=10, num_conferences=10, num_numerical_features=10, hidden_size=512, **kwargs):
        super().__init__(**kwargs)
        self.num_venues = num_venues
        self.num_teams = num_teams
        self.num_weather_conditions = num_weather_conditions
        self.num_conferences = num_conferences
        self.num_numerical_features = num_numerical_features
        self.hidden_size = hidden_size

class MatchupEncoder(PreTrainedModel):
    config_class = MatchupEncoderConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Embedding layers for categorical data
        self.season_type_embed = nn.Embedding(3, 16)  # 'regular', 'postseason', 'spring'
        self.boolean_embed = nn.Embedding(2, 8)  # For neutralSite, conferenceGame, gameIndoors
        self.venue_embed = nn.Embedding(config.num_venues, 32)
        self.team_embed = nn.Embedding(config.num_teams, 64)
        self.weather_condition_embed = nn.Embedding(config.num_weather_conditions, 16)
        self.conference_embed = nn.Embedding(config.num_conferences, 32)

        # Linear layers for numerical data
        self.numerical_encoder = nn.Linear(config.num_numerical_features, 128)

        # Combine all features
        total_embedding_dim = (16 + 8 * 3 + 32 + 64 * 2 + 16 + 32 * 2 + 128)
        
        # Final layers
        self.layer_norm = nn.LayerNorm(total_embedding_dim)
        self.fc1 = nn.Linear(total_embedding_dim, 256)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(256, config.hidden_size)

        self.post_init()

    def forward(self, game_input):
        # Unpack the input
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
        away_conference_emb = self.conference_embed(numerical_features[0])  # Assuming away_conference is the first item in numerical_features

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

        # Final processing
        normalized = self.layer_norm(combined)
        hidden = self.fc1(normalized)
        activated = self.activation(hidden)
        output = self.fc2(activated)

        return output