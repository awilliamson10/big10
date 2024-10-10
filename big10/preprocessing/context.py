import polars as pl
from typing import List, Tuple
from big10.preprocessing.load import load_and_preprocess_data

stats, games, drives, weather, winprobs = load_and_preprocess_data()

def get_game_information(game_id):
    game_info = games.filter(pl.col("id") == game_id)
    if game_info.is_empty():
        return None
    return game_info.to_dicts()[0]

def get_game_weather(game_id):
    weather_info = weather.filter(pl.col("id") == game_id)
    if weather_info.is_empty():
        return None
    return weather_info.to_dicts()[0]

def get_win_probabilities(game_id):
    win_probabilities = winprobs.filter(pl.col("game_id") == game_id)
    if win_probabilities.is_empty():
        return None
    win_prob = win_probabilities.to_dicts()[0]
    home_prob = win_prob['home_win_prob']
    home_spread = win_prob['spread']
    return {
        "home_win_prob": home_prob,
        "home_spread": home_spread,
    }

def get_game_team_stats(game_information):
    teams = {}
    home_team = game_information['homeTeam']
    away_team = game_information['awayTeam']
    season = game_information['season']
    week = game_information['week'] - 1
    home_team_stats = stats.filter(pl.col("team") == home_team).filter(pl.col("season") == season).filter(pl.col("week") == week)
    if home_team_stats.is_empty():
        raise ValueError(f"No stats found for {home_team} in season {season} week {week}")
    teams["home_team"] = home_team_stats.to_dicts()
    away_team_stats = stats.filter(pl.col("team") == away_team).filter(pl.col("season") == season).filter(pl.col("week") == week)
    if away_team_stats.is_empty():
        raise ValueError(f"No stats found for {away_team} in season {season} week {week}")
    teams["away_team"] = away_team_stats.to_dicts()

    return teams

def format_drive(drive):
    return f"{drive['offense']}, Start: {drive['startYardsToGoal']}, Plays: {drive['plays']}, Yards: {drive['yards']}, Result: {drive['driveResult']}"

def get_game_context(game_id):
    game_info = get_game_information(game_id)
    if game_info is None:
        print(f"No game information found for game_id {game_id}")
        return None
    weather_info = get_game_weather(game_id)
    winprobs = get_win_probabilities(game_id)
    team_stats = get_game_team_stats(game_info)
    game_drives = drives.filter(pl.col("gameId") == game_id).sort("driveNumber")

    drive_texts = []
    for drive in game_drives.to_dicts():
        drive_text = format_drive(drive)
        drive_texts.append(drive_text)
    text = "".join([f"<drive>{drive}</drive>" for drive in drive_texts])
    return {
        "game_info": game_info,
        "weather_info": weather_info,
        "winprobs": winprobs,
        "team_stats": team_stats,
        "drives": "<matchup> " + text
    }

def gen_game_input(game_context):
    game_info = game_context["game_info"]

    week = game_info["week"]
    seasonType = game_info["seasonType"]
    neutralSite = game_info["neutralSite"]
    conferenceGame = game_info["conferenceGame"]
    venueId = game_info["venueId"]
    homeId = game_info["homeId"]
    awayId = game_info["awayId"]
    homePregameElo = game_info["homePregameElo"]
    awayPregameElo = game_info["awayPregameElo"]

    weather_info = game_context["weather_info"]
    gameIndoors = weather_info["gameIndoors"] if weather_info is not None else True
    temperature = weather_info["temperature"] if weather_info is not None else 70
    humidity = weather_info["humidity"] if weather_info is not None else 50
    windSpeed = weather_info["windSpeed"] if weather_info is not None else 0
    weatherCondition = weather_info["weatherCondition"] if weather_info is not None else "Clear"

    winprobs = game_context["winprobs"]
    home_win_prob = winprobs["home_win_prob"] if winprobs is not None else -1
    home_spread = winprobs["home_spread"] if winprobs is not None else -1

    home_team_stats = game_context["team_stats"]["home_team"][0]
    away_team_stats = game_context["team_stats"]["away_team"][0]

    home_team_stats = {k: v for k, v in home_team_stats.items() if k not in ["team", "season", "week"]}
    away_team_stats = {k: v for k, v in away_team_stats.items() if k not in ["team", "season", "week"]}

    game_input = [
        week, seasonType, neutralSite, conferenceGame, venueId, homeId, awayId, homePregameElo, awayPregameElo,
        gameIndoors, temperature, humidity, windSpeed, weatherCondition, home_win_prob, home_spread
    ]
    game_input += list(home_team_stats.values())
    game_input += list(away_team_stats.values())

    return game_input, game_context["drives"]

def get_processed_game_data(game_id: str) -> Tuple[List[float], str]:
    game_context = get_game_context(game_id)
    if game_context is None:
        # print(f"Warning: No game context found for game_id {game_id}")
        return None
    game_input, drives_text = gen_game_input(game_context)
    if game_input is None or drives_text is None:
        return None
    return game_input, drives_text