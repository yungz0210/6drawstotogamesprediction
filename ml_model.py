import numpy as np
import pandas as pd
from collections import Counter

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59}

try:
    from sklearn.ensemble import RandomForestClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    RandomForestClassifier = None


def extract_ticket_features(ticket, game_range, freq_map=None, drought_map=None):
    """
    Extracts advanced statistical feature vectors for ML classification.
    """
    ticket = sorted([int(n) for n in ticket])
    sum_tot = sum(ticket)
    odd_c = sum(1 for n in ticket if n % 2 != 0)
    even_c = 6 - odd_c
    bday_c = sum(1 for n in ticket if n <= 31)
    high_c = sum(1 for n in ticket if n > (game_range // 2))
    prime_c = sum(1 for n in ticket if n in PRIMES)
    
    deltas = [ticket[i+1] - ticket[i] for i in range(len(ticket)-1)]
    delta_mean = float(np.mean(deltas)) if deltas else 0.0
    delta_std = float(np.std(deltas)) if deltas else 0.0
    delta_max = float(max(deltas)) if deltas else 0.0
    consec_c = sum(1 for g in deltas if g == 1)
    
    avg_freq = 0.0
    if freq_map:
        avg_freq = float(np.mean([freq_map.get(n, 0) for n in ticket]))
        
    avg_drought = 0.0
    if drought_map:
        avg_drought = float(np.mean([drought_map.get(n, 0) for n in ticket]))
        
    return [
        sum_tot, odd_c, even_c, bday_c, high_c, prime_c, 
        delta_mean, delta_std, delta_max, consec_c, 
        avg_freq, avg_drought
    ]

def train_ml_evaluator(df, game_range):
    """
    Trains a Random Forest classifier comparing actual historical winning sets (Positive class = 1)
    against random invalid combinations (Negative class = 0).
    """
    main_cols = ['DrawnNo1', 'DrawnNo2', 'DrawnNo3', 'DrawnNo4', 'DrawnNo5', 'DrawnNo6']
    
    # Calculate historical frequency and drought maps
    all_nums = df[main_cols].values.flatten()
    freq_map = dict(Counter(all_nums))
    
    drought_map = {}
    for i in range(1, game_range + 1):
        drought_map[i] = 999
    for idx, row in df.iterrows():
        for n in row[main_cols].values:
            if drought_map[int(n)] == 999:
                drought_map[int(n)] = idx

    if not HAS_SKLEARN:
        return None, freq_map, drought_map

    X = []
    y = []
    
    # Positive samples (actual draws)
    for _, row in df.iterrows():
        t = row[main_cols].values
        feats = extract_ticket_features(t, game_range, freq_map, drought_map)
        X.append(feats)
        y.append(1)
        
    # Synthetic Negative samples (random permutations outside natural distributions)
    num_samples = len(X)
    for _ in range(num_samples):
        rand_ticket = sorted(np.random.choice(range(1, game_range + 1), size=6, replace=False))
        feats = extract_ticket_features(rand_ticket, game_range, freq_map, drought_map)
        X.append(feats)
        y.append(0)
        
    clf = RandomForestClassifier(n_estimators=120, random_state=42, max_depth=10)
    clf.fit(X, y)
    
    return clf, freq_map, drought_map

def predict_ticket_ml_score(model_tuple, ticket, game_range):
    """
    Evaluates a candidate ticket and returns ML Confidence score (0 to 100%).
    """
    clf, freq_map, drought_map = model_tuple
    if clf is None or not HAS_SKLEARN:
        # Fallback heuristic calculation if scikit-learn is missing
        feats = extract_ticket_features(ticket, game_range, freq_map, drought_map)
        sum_tot = feats[0]
        sum_score = 40.0 if (110 <= sum_tot <= 230) else 20.0
        gap_score = min(30.0, feats[6] * 3.0)
        freq_score = min(30.0, feats[10] * 0.1)
        return round(sum_score + gap_score + freq_score, 1)
        
    feats = extract_ticket_features(ticket, game_range, freq_map, drought_map)
    prob = clf.predict_proba([feats])[0][1]
    return round(float(prob) * 100.0, 1)


