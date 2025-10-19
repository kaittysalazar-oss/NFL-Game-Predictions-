#  NFL Game Predictions in Python

This project predicts **NFL game outcomes** using team performance data and machine learning.  
It demonstrates how to collect, transform, and model real-world sports data using **Python**, **pandas**, **scikit-learn**, and **nfl_data_py**.

---

## Project Overview

The analysis focuses on:
- Predicting game winners for a selected **season and week**.  
- Calculating team-level performance metrics such as average points scored, allowed, win percentage, and point differential.  
- Using **rolling averages** to capture recent team momentum.  
- Training a **HistGradientBoostingClassifier** to forecast outcomes based on historical performance.  
- Providing an adaptable framework for future weeks or seasons.

This project reflects applied **sports analytics and predictive modeling**, showing how data-driven insights can inform game outcome predictions.

---

## Skills Demonstrated

- **Python Programming** for data preparation and modeling.  
- **Data Wrangling & Feature Engineering** using `pandas`.  
- **Predictive Modeling** with `scikit-learn`’s `HistGradientBoostingClassifier`.  
- **API Integration** using `nfl_data_py` to import live schedule and score data.  
- **Analytical Thinking** — interpreting model features and performance trends.

---

## Tools & Libraries

- `nfl_data_py` — access to NFL schedules, scores, and stats.  
- `pandas` — data manipulation and aggregation.  
- `scikit-learn` — model training and evaluation.  

Install dependencies:
```bash
pip install nfl_data_py pandas scikit-learn
