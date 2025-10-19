NFL Game Prediction Model

This project predicts NFL game outcomes using Python, official NFL data, and a machine learning model built with scikit-learn.
The script automatically retrieves schedules, scores, and team statistics via the nfl_data_py API, which sources data directly from official NFL data feeds, and uses them to forecast weekly results.

Features

Automated Data Retrieval: Loads current and historical NFL schedules, scores, and statistics using nfl_data_py (no manual data files required).
Feature Engineering: Builds features such as:
Season averages (points scored, allowed, win percentage)
Strength of schedule
Rolling performance trends (last N games)
Matchup-based comparisons (offense vs. defense)

Machine Learning Model: Uses HistGradientBoostingClassifier to predict home team win probabilities.
Output Summary: Lists favorites, underdogs, and confidence tiers (High, Medium, Low) for the upcoming week.


Requirements
Install the required Python libraries:
pip install nfl_data_py pandas scikit-learn
Usage

Clone this repository:
git clone https://github.com/kaittysalazar/nfl-game-predictions.git
cd nfl-game-predictions
Run the script:
python nfl_predictions.py

The script will print:
Completed games through the previous week
Predicted outcomes for the upcoming week, including win probabilities and confidence tiers

Model Overview
Algorithm: HistGradientBoostingClassifier (scikit-learn)

Target Variable: Home team win

Feature Inputs: Season-level, rolling, and matchup-based statistics

Training: Model retrains weekly using updated team data

Future Improvements:

Incorporate player-level or injury-based features
Add a Streamlit or Tableau dashboard for interactive visualization
Track and compare model accuracy across weeks


Data Source
All data is retrieved from official NFL data feeds using the open-source nfl_data_py package.
This ensures that the model is built on verified and regularly updated league data.


Tech Stack
Python

pandas

scikit-learn

nfl_data_py
