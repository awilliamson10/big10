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
    teams["home_team"] = home_team_stats.to_dicts()[0]
    away_team_stats = stats.filter(pl.col("team") == away_team).filter(pl.col("season") == season).filter(pl.col("week") == week)
    if away_team_stats.is_empty():
        raise ValueError(f"No stats found for {away_team} in season {season} week {week}")
    teams["away_team"] = away_team_stats.to_dicts()[0]

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

def camelCase(st):
    # convert from snake_case to camelCase
    output = ''.join(x for x in st.title() if x.isalnum())
    return output[0].upper() + output[1:]

def extract_game_context(game):
    info = game.get("game_info", {})

    week = info.get("week", None)
    seasonType = info.get("seasonType", None)
    neutralSite = info.get("neutralSite", None)
    conferenceGame = info.get("conferenceGame", None)
    venueId = info.get("venueId", None)

    homeId = info.get("homeId", None)
    homeConference = info.get("homeConference", None)
    awayId = info.get("awayId", None)
    awayConference = info.get("awayConference", None)

    weather = game.get("weather_info", {})
    startTime = weather.get("startTime", None)
    if startTime is not None:
        startTime = startTime.strftime("%H:%M")
    gameIndoors = weather.get("gameIndoors", None)
    temperature = weather.get("temperature", None)
    if temperature is not None:
        temperature = int(temperature)
    humidity = weather.get("humidity", None)
    if humidity is not None:
        humidity = int(humidity)
    precipitation = weather.get("precipitation", None)
    if precipitation is not None:
        precipitation = int(precipitation * 100)
    windSpeed = weather.get("windSpeed", None)
    if windSpeed is not None:
        windSpeed = int(windSpeed)
    weatherCondition = weather.get("weatherCondition", None)

    winprobs = game.get("winprobs", {})
    homeWinProb = winprobs.get("home_win_prob", None)
    if homeWinProb is not None:
        homeWinProb = int(homeWinProb * 100)
    homeSpread = winprobs.get("home_spread", None)
    if homeSpread is not None:
        homeSpread = int(homeSpread)

    homeTeam = game.get("team_stats", {}).get("home_team", {})
    awayTeam = game.get("team_stats", {}).get("away_team", {})

    home_team_stats = {f"home{camelCase(k)}": int(v) for k, v in homeTeam.items() if k not in ["team", "conference", "season", "week"]}
    away_team_stats = {f"away{camelCase(k)}": int(v) for k, v in awayTeam.items() if k not in ["team", "conference", "season", "week"]}

    drives = game.get("drives", None)

    return {
        "week": week,
        "seasonType": seasonType,
        "neutralSite": neutralSite,
        "conferenceGame": conferenceGame,
        "venueId": venueId,
        "homeId": homeId,
        "homeConference": homeConference,
        "awayId": awayId,
        "awayConference": awayConference,
        "startTime": startTime,
        "gameIndoors": gameIndoors,
        "temperature": temperature,
        "humidity": humidity,
        "precipitation": precipitation,
        "windSpeed": windSpeed,
        "weatherCondition": weatherCondition,
        "homeWinProb": homeWinProb,
        "homeSpread": homeSpread,
        **home_team_stats,
        **away_team_stats,
        "drives": drives
    }
