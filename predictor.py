import numpy as np
import pandas as pd
from collections import Counter
import random

def monte_carlo_simulation(df, game_range, iterations=10000):
    # Calculate historical probability weights
    main_cols = ['DrawnNo1', 'DrawnNo2', 'DrawnNo3', 'DrawnNo4', 'DrawnNo5', 'DrawnNo6']
    all_nums = df[main_cols].values.flatten()
    freq = Counter(all_nums)
    
    population = list(range(1, game_range + 1))
    weights = [freq.get(i, 0) + 1 for i in population] # Add 1 for laplace smoothing
    
    sets = []
    for _ in range(iterations):
        draw = sorted(random.choices(population, weights=weights, k=6))
        # Ensure unique numbers
        while len(set(draw)) < 6:
            draw = sorted(random.choices(population, weights=weights, k=6))
        sets.append(tuple(sorted([int(x) for x in draw])))
    
    most_common = Counter(sets).most_common(5)
    return [list(s[0]) for s in most_common]

def mean_reversion_due(df, game_range):
    # Identify numbers with lowest frequency relative to expected
    main_cols = ['DrawnNo1', 'DrawnNo2', 'DrawnNo3', 'DrawnNo4', 'DrawnNo5', 'DrawnNo6']
    all_nums = df[main_cols].values.flatten()
    freq = Counter(all_nums)
    
    # Drought (last seen)
    last_seen = {}
    for i in range(1, game_range + 1):
        last_seen[i] = 9999 # Large number
        
    for idx, row in df.iterrows():
        nums = row[main_cols].values
        for n in nums:
            if n not in last_seen or last_seen[n] == 9999:
                last_seen[n] = idx # df is sorted descending by date, so idx 0 is most recent
                
    # Sort by drought (largest index means longest ago)
    due_nums = sorted(last_seen.keys(), key=lambda x: last_seen[x], reverse=True)
    return sorted([int(x) for x in due_nums[:6]])

def markov_chain_analysis(df, game_range):
    # Transition matrix: what numbers appear after others
    # Since it's a set of 6, we can look at what appeared in T-1 and what appeared in T
    matrix = np.zeros((game_range + 1, game_range + 1))
    
    main_cols = ['DrawnNo1', 'DrawnNo2', 'DrawnNo3', 'DrawnNo4', 'DrawnNo5', 'DrawnNo6']
    df = df.sort_values('DrawDate', ascending=True) # Sort ascending for transition
    
    rows = df[main_cols].values
    for i in range(len(rows) - 1):
        prev_draw = rows[i]
        curr_draw = rows[i+1]
        for p in prev_draw:
            for c in curr_draw:
                matrix[int(p)][int(c)] += 1
                
    # Most recent draw
    last_draw = rows[-1]
    
    # Predict based on last draw
    probabilities = np.zeros(game_range + 1)
    for p in last_draw:
        probabilities += matrix[int(p)]
        
    # Get top 6
    predicted = np.argsort(probabilities)[-6:][::-1]
    return sorted([int(x) for x in predicted])

def hybrid_ensemble(df, game_range):
    # 2 Hot, 2 Due/Cold, 2 from common pairs
    main_cols = ['DrawnNo1', 'DrawnNo2', 'DrawnNo3', 'DrawnNo4', 'DrawnNo5', 'DrawnNo6']
    all_nums = df[main_cols].values.flatten()
    freq = Counter(all_nums)
    
    hot = sorted(freq.keys(), key=lambda x: freq[x], reverse=True)[:10]
    cold = sorted(freq.keys(), key=lambda x: freq[x])[:10]
    
    selected = set()
    # 1. Select 2 Hot numbers
    selected.update(random.sample(hot, 2))
    
    # 2. Select 2 Due/Cold numbers
    available_cold = [c for c in cold if c not in selected]
    selected.update(random.sample(available_cold, 2))
    
    # 3. Select 2 numbers based on common pair associations
    # Get pairs from recent history
    from analytics import get_pairs
    top_pairs = get_pairs(df, lookback=100) # Use last 100 draws for association
    
    # Try to find a pair where one of the selected numbers is present
    for (n1, n2), count in top_pairs:
        if len(selected) >= 6:
            break
        if n1 in selected and n2 not in selected:
            selected.add(n2)
        elif n2 in selected and n1 not in selected:
            selected.add(n1)
            
    # If still not enough, fill with common pairs regardless of association
    for (n1, n2), count in top_pairs:
        if len(selected) >= 6:
            break
        if n1 not in selected:
            selected.add(n1)
        if len(selected) < 6 and n2 not in selected:
            selected.add(n2)

    # Final fallback
    while len(selected) < 6:
        n = random.randint(1, game_range)
        if n not in selected:
            selected.add(n)
    
    return sorted([int(x) for x in selected])
