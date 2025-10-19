import nfl_data_py as nfl
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

# -----------------------------
# CONFIGURATION
# -----------------------------
TARGET_SEASON = 2025
TARGET_WEEK = 7
ROLLING_GAMES = 5  # last N games for momentum features

# -----------------------------
# STEP 1: Load schedules & scores
# -----------------------------
def load_games(season):
    games = nfl.import_schedules([season])[['season','week','home_team','away_team','home_score','away_score']]
    games['home_win'] = (games['home_score'] > games['away_score']).astype(int)
    return games

games_prev = load_games(TARGET_SEASON - 1)
games_curr = load_games(TARGET_SEASON)

# Keep only completed games
games_prev = games_prev.dropna(subset=['home_score','away_score'])
games_curr_played = games_curr.dropna(subset=['home_score','away_score'])

# -----------------------------
# STEP 2: Season stats
# -----------------------------
def season_team_stats(games):
    teams = pd.unique(games[['home_team','away_team']].values.ravel())
    stats = {}
    for team in teams:
        home = games[games['home_team']==team]
        away = games[games['away_team']==team]
        scored = home['home_score'].tolist() + away['away_score'].tolist()
        allowed = home['away_score'].tolist() + away['home_score'].tolist()
        wins = home['home_win'].tolist() + (1 - away['home_win']).tolist()
        stats[team] = {
            'avg_scored': sum(scored)/len(scored) if scored else 0,
            'avg_allowed': sum(allowed)/len(allowed) if allowed else 0,
            'point_diff': (sum(scored)/len(scored) - sum(allowed)/len(allowed)) if scored else 0,
            'win_pct': sum(wins)/len(wins) if wins else 0
        }
    return pd.DataFrame(stats).T.reset_index().rename(columns={'index':'team'})

stats_prev = season_team_stats(games_prev)
stats_curr = season_team_stats(games_curr_played[games_curr_played['week'] < TARGET_WEEK])

# -----------------------------
# STEP 3: Strength of schedule
# -----------------------------
def strength_of_schedule(games, stats):
    sos = {}
    teams = pd.unique(games[['home_team','away_team']].values.ravel())
    for team in teams:
        team_games = games[(games['home_team']==team) | (games['away_team']==team)]
        opponents = [r['away_team'] if r['home_team']==team else r['home_team'] for _,r in team_games.iterrows()]
        opp_win_pcts = [stats.loc[stats['team']==opp,'win_pct'].values[0] for opp in opponents if opp in stats['team'].values]
        sos[team] = sum(opp_win_pcts)/len(opp_win_pcts) if opp_win_pcts else 0
    return pd.DataFrame(list(sos.items()), columns=['team','sos'])

sos_prev = strength_of_schedule(games_prev, stats_prev)
sos_curr = strength_of_schedule(games_curr_played[games_curr_played['week'] < TARGET_WEEK], stats_curr)

# -----------------------------
# STEP 4: Upcoming matchups
# -----------------------------
week_games = games_curr[games_curr['week']==TARGET_WEEK][['home_team','away_team']]

# -----------------------------
# STEP 5: Merge features
# -----------------------------
def merge_features(df, stats_prev, stats_curr, sos_prev, sos_curr):
    df = df.merge(stats_prev.add_prefix('home_last_'), left_on='home_team', right_on='home_last_team', how='left')
    df = df.merge(stats_prev.add_prefix('away_last_'), left_on='away_team', right_on='away_last_team', how='left')
    df = df.merge(sos_prev.add_prefix('home_last_'), left_on='home_team', right_on='home_last_team', how='left')
    df = df.merge(sos_prev.add_prefix('away_last_'), left_on='away_team', right_on='away_last_team', how='left')
    df = df.merge(stats_curr.add_prefix('home_curr_'), left_on='home_team', right_on='home_curr_team', how='left')
    df = df.merge(stats_curr.add_prefix('away_curr_'), left_on='away_team', right_on='away_curr_team', how='left')
    df = df.merge(sos_curr.add_prefix('home_curr_'), left_on='home_team', right_on='home_curr_team', how='left')
    df = df.merge(sos_curr.add_prefix('away_curr_'), left_on='away_team', right_on='away_curr_team', how='left')
    return df

train = merge_features(games_prev, stats_prev, stats_prev, sos_prev, sos_prev)
week = merge_features(week_games, stats_prev, stats_curr, sos_prev, sos_curr)

# Fill missing current-year values with last year's
for col in ['home_curr_avg_scored','home_curr_avg_allowed','home_curr_point_diff','home_curr_win_pct','home_curr_sos']:
    train[col] = train[col].fillna(train[col.replace('curr','last')])
    week[col] = week[col].fillna(week[col.replace('curr','last')])
for col in ['away_curr_avg_scored','away_curr_avg_allowed','away_curr_point_diff','away_curr_win_pct','away_curr_sos']:
    train[col] = train[col].fillna(train[col.replace('curr','last')])
    week[col] = week[col].fillna(week[col.replace('curr','last')])

# -----------------------------
# STEP 6: Matchup features
# -----------------------------
def add_matchup_features(df):
    df['score_diff_matchup'] = df['home_curr_avg_scored'] - df['away_curr_avg_allowed']
    df['defense_diff_matchup'] = df['home_curr_avg_allowed'] - df['away_curr_avg_scored']
    df['win_pct_diff'] = df['home_curr_win_pct'] - df['away_curr_win_pct']
    df['sos_diff'] = df['home_curr_sos'] - df['away_curr_sos']
    df['offense_vs_defense'] = df['home_curr_avg_scored'] * df['away_curr_avg_allowed']
    df['defense_vs_offense'] = df['home_curr_avg_allowed'] * df['away_curr_avg_scored']
    return df

train = add_matchup_features(train)
week = add_matchup_features(week)

# -----------------------------
# STEP 7: Rolling N-game stats
# -----------------------------
def add_rolling_features(df, games, n=ROLLING_GAMES):
    for team_type in ['home_team','away_team']:
        df[f'{team_type}_roll_avg_scored'] = 0
        df[f'{team_type}_roll_avg_allowed'] = 0
        df[f'{team_type}_roll_win_pct'] = 0
        for i, row in df.iterrows():
            team_games = games[(games['home_team']==row[team_type]) | (games['away_team']==row[team_type])]
            team_games = team_games.sort_values(['season','week'], ascending=False).head(n)
            scored = team_games.apply(lambda r: r['home_score'] if r['home_team']==row[team_type] else r['away_score'], axis=1)
            allowed = team_games.apply(lambda r: r['away_score'] if r['home_team']==row[team_type] else r['home_score'], axis=1)
            wins = team_games.apply(lambda r: r['home_win'] if r['home_team']==row[team_type] else 1 - r['home_win'], axis=1)
            df.at[i,f'{team_type}_roll_avg_scored'] = scored.mean() if not scored.empty else 0
            df.at[i,f'{team_type}_roll_avg_allowed'] = allowed.mean() if not allowed.empty else 0
            df.at[i,f'{team_type}_roll_win_pct'] = wins.mean() if not wins.empty else 0
    return df

train = add_rolling_features(train, games_prev)
week = add_rolling_features(week, games_curr_played)

# -----------------------------
# STEP 8: Feature columns
# -----------------------------
feature_cols = [
    'home_last_avg_scored','home_last_avg_allowed','home_last_point_diff','home_last_win_pct',
    'away_last_avg_scored','away_last_avg_allowed','away_last_point_diff','away_last_win_pct',
    'home_curr_avg_scored','home_curr_avg_allowed','home_curr_point_diff','home_curr_win_pct',
    'away_curr_avg_scored','away_curr_avg_allowed','away_curr_point_diff','away_curr_win_pct',
    'home_curr_sos','away_curr_sos',
    'score_diff_matchup','defense_diff_matchup','win_pct_diff','sos_diff',
    'offense_vs_defense','defense_vs_offense',
    'home_team_roll_avg_scored','home_team_roll_avg_allowed','home_team_roll_win_pct',
    'away_team_roll_avg_scored','away_team_roll_avg_allowed','away_team_roll_win_pct'
]

X_train = train[feature_cols].fillna(0)
y_train = train['home_win']
X_week = week[feature_cols].fillna(0)

# -----------------------------
# STEP 9: Train HistGB classifier
# -----------------------------
hgb = HistGradientBoostingClassifier(
    max_iter=500,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
hgb.fit(X_train, y_train)
week['home_win_prob'] = hgb.predict_proba(X_week)[:,1]

# -----------------------------
# STEP 10: Favorite / underdog
# -----------------------------
week['favorite'] = week.apply(lambda r: r['home_team'] if r['home_win_prob']>=0.5 else r['away_team'], axis=1)
week['underdog'] = week.apply(lambda r: r['away_team'] if r['home_win_prob']>=0.5 else r['home_team'], axis=1)

# -----------------------------
# STEP 11: Confidence tier
# -----------------------------
def confidence_tier(prob):
    if prob >= 0.75:
        return "High"
    elif prob >= 0.55:
        return "Medium"
    else:
        return "Low"

week['confidence'] = week['home_win_prob'].apply(confidence_tier)

# -----------------------------
# STEP 12: Display completed games
# -----------------------------
print(f"\n=== Completed games through Week {TARGET_WEEK-1}, {TARGET_SEASON} ===")
completed_games = games_curr_played[games_curr_played['week'] < TARGET_WEEK][
    ['week','home_team','away_team','home_score','away_score','home_win']
].sort_values(['week','home_team'])
print(completed_games.to_string(index=False))

# -----------------------------
# STEP 13: Display predictions
# -----------------------------
print(f"\n=== Predictions for Week {TARGET_WEEK}, {TARGET_SEASON} ===")
pred_table = week[['away_team','home_team','favorite','home_win_prob','confidence']].copy()
pred_table = pred_table.sort_values('home_win_prob', ascending=False)
print(pred_table.to_string(index=False))
