import pandas as pd
import numpy as np
import predictor

# Prize structures and odds (approximate based on standard Sports Toto)
# 6/50 (Star Toto): 1 in 15,890,700
# 6/55 (Power Toto): 1 in 28,989,675
# 6/58 (Supreme Toto): 1 in 40,475,358

GAME_ODDS = {
    "6/50": 15890700,
    "6/55": 28989675,
    "6/58": 40475358
}

TICKET_COST = 2.0 # Standard cost per set

def evaluate_prediction(predicted, actual, bonus_predicted=None, actual_bonus=None):
    """
    Evaluates a prediction against actual results.
    Returns: (matches, has_bonus_match, prize_won)
    """
    predicted_set = set(predicted)
    actual_set = set(actual)
    matches = len(predicted_set.intersection(actual_set))

    has_bonus_match = False
    if bonus_predicted is not None and actual_bonus is not None:
        has_bonus_match = (bonus_predicted == actual_bonus)

    prize = 0
    # Approximate prize tiers for simulation
    if matches == 6:
        prize = 1000000 # Jackpot placeholder (usually millions, but let's use a base for ROI)
    elif matches == 5 and has_bonus_match:
        prize = 100000
    elif matches == 5:
        prize = 2000
    elif matches == 4:
        prize = 100
    elif matches == 3:
        prize = 20

    return matches, has_bonus_match, prize

def backtest_macro(df, game_range, model_type, lookback=100):
    """
    Runs backtests over the last 'lookback' draws.
    Returns a dictionary of results.
    """
    # Ensure df is descending
    df = df.sort_values('DrawDate', ascending=False).reset_index(drop=True)

    results = {
        "matches_3": 0,
        "matches_4": 0,
        "matches_5": 0,
        "matches_6": 0,
        "total_prize": 0,
        "total_cost": lookback * TICKET_COST
    }

    main_cols = ['DrawnNo1', 'DrawnNo2', 'DrawnNo3', 'DrawnNo4', 'DrawnNo5', 'DrawnNo6']

    # We need at least some data to predict
    if len(df) <= lookback + 10:
        return results

    for i in range(lookback):
        # Data available prior to draw at index i
        historical_data = df.iloc[i+1:]
        actual_row = df.iloc[i]
        actual_draw = actual_row[main_cols].values
        actual_bonus = actual_row['BonusNo'] if 'BonusNo' in actual_row else None

        # Predict
        if model_type == "Monte Carlo Simulation":
            pred = predictor.monte_carlo_simulation(historical_data, game_range, iterations=1000)[0]
        elif model_type == "Mean Reversion (Due)":
            pred = predictor.mean_reversion_due(historical_data, game_range)
        elif model_type == "Markov Chain Analysis":
            pred = predictor.markov_chain_analysis(historical_data, game_range)
        else: # Hybrid
            pred = predictor.hybrid_ensemble(historical_data, game_range)

        bonus_pred = None
        if actual_bonus is not None:
            bonus_pred = predictor.predict_bonus_number(historical_data, game_range, model_type)

        m, hb, p = evaluate_prediction(pred, actual_draw, bonus_pred, actual_bonus)

        results["total_prize"] += p
        if m == 3: results["matches_3"] += 1
        elif m == 4: results["matches_4"] += 1
        elif m == 5: results["matches_5"] += 1
        elif m == 6: results["matches_6"] += 1

    return results

def calculate_ev(game_selection, jackpot_size):
    """
    Calculates expected value of a ticket.
    EV = Sum(Probability * Prize)
    """
    odds = GAME_ODDS.get(game_selection, 15000000)
    # Simple EV focusing on Jackpot as main driver
    ev = (1.0 / odds) * jackpot_size
    # Add small factor for lower tiers (simplified)
    ev += 0.20 # Contribution from smaller prizes
    return ev
