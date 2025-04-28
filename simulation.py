import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
teams_df = pd.read_csv('data/RST_Final_Sheet1.csv')
matchups_df = pd.read_csv('data/RST_Final_Sheet2.csv')

# Clean up strings
matchups_df['Higher Seed'] = matchups_df['Higher Seed'].str.strip().str.replace(')', '', regex=False)
matchups_df['Lower Seed'] = matchups_df['Lower Seed'].str.strip().str.replace(')', '', regex=False)

# Boosted score calculation
def boosted_score(team):
    return (2.0 * (team['Offensive Efficiency'] - team['Defensive Efficiency']) +
            1.5 * team['Strength of Schedule'] +
            1.0 * team['Tempo'])

# Logistic win probability
def win_probability(score_diff, k=0.1):
    return 1 / (1 + np.exp(-k * score_diff))

# Simulate one full tournament tracking round-by-round
def simulate_tournament_full(teams_df, matchups_df):
    def predict_matchup(team1_name, team2_name, teams_df):
        team1 = teams_df[teams_df['Team Name'] == team1_name].iloc[0]
        team2 = teams_df[teams_df['Team Name'] == team2_name].iloc[0]
        
        score1 = boosted_score(team1)
        score2 = boosted_score(team2)
        score_diff = score1 - score2
        
        prob_team1_wins = win_probability(score_diff)
        return team1_name if np.random.rand() < prob_team1_wins else team2_name

    def create_matchups(winners_list):
        return [(winners_list[i], winners_list[i+1]) for i in range(0, len(winners_list), 2)]

    def simulate_round(matchups, teams_df):
        winners = []
        for team1_name, team2_name in matchups:
            if team1_name in teams_df['Team Name'].values and team2_name in teams_df['Team Name'].values:
                winner = predict_matchup(team1_name, team2_name, teams_df)
                winners.append(winner)
            else:
                winners.append(None)
        return winners

    rounds_record = {
        "Round of 32": [],
        "Sweet 16": [],
        "Elite 8": [],
        "Final 4": [],
        "Championship Game": [],
        "Champion": []
    }

    # Round of 64 to 32
    first_round_winners = []
    for _, row in matchups_df.iterrows():
        team1 = row['Higher Seed']
        team2 = row['Lower Seed']

        if team1 in teams_df['Team Name'].values and team2 in teams_df['Team Name'].values:
            winner = predict_matchup(team1, team2, teams_df)
            first_round_winners.append(winner)
        else:
            first_round_winners.append(None)

    rounds_record["Round of 32"].extend(first_round_winners)

    # Next rounds
    current_winners = first_round_winners
    for round_name in ["Sweet 16", "Elite 8", "Final 4", "Championship Game", "Champion"]:
        matchups = create_matchups(current_winners)
        winners = simulate_round(matchups, teams_df)
        rounds_record[round_name].extend(winners)
        current_winners = winners

    return rounds_record

# Simulate tournaments
num_simulations = 1000
round_results = {
    "Round of 32": [],
    "Sweet 16": [],
    "Elite 8": [],
    "Final 4": [],
    "Championship Game": [],
    "Champion": []
}

for _ in range(num_simulations):
    tournament = simulate_tournament_full(teams_df, matchups_df)
    for round_name in round_results.keys():
        round_results[round_name].extend(tournament[round_name])

# Analyze results
round_summary = {}
for round_name, teams in round_results.items():
    team_counts = pd.Series(teams).value_counts(normalize=True) * 100
    round_summary[round_name] = team_counts

# Plot Champion Wins
top_champions = round_summary["Champion"].head(10)

plt.figure(figsize=(10,6))
plt.bar(top_champions.index, top_champions.values, edgecolor='black', color='lightgreen')
plt.title('Top 10 Championship Winners (1000 Simulations)')
plt.ylabel('Win Percentage (%)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
