import pandas as pd
import numpy as np
from itertools import combinations, permutations
from collections import Counter
import random

def format_4d(num):
    """Ensures a 4D number is padded as a 4-digit string (e.g., '0123')."""
    s = str(num).strip()
    return s.zfill(4)[-4:]

def generate_4d_permutations(num_str):
    """
    Generates all unique Box Play / i-Perm permutations for a 4D string.
    Returns: (list_of_unique_perms, perm_type_label, total_cost_rm)
    """
    num_str = format_4d(num_str)
    raw_perms = set("".join(p) for p in permutations(num_str))
    sorted_perms = sorted(list(raw_perms))
    count = len(sorted_perms)
    
    label_map = {
        24: "24-Way Box (4 Unique Digits)",
        12: "12-Way Box (1 Pair)",
        6: "6-Way Box (2 Pairs)",
        4: "4-Way Box (Triple Digits)",
        1: "Direct 4D (All Same Digits)"
    }
    perm_label = label_map.get(count, f"{count}-Way Box")
    total_cost = float(count) # RM 1 per permutation
    
    return sorted_perms, perm_label, total_cost

def generate_system_4d_jackpot(number_list, ticket_cost=2.0):
    """
    Generates all pairs for Toto 4D Jackpot from a pool of candidate 4D numbers.
    For N 4D numbers, produces C(N, 2) jackpot pairs.
    """
    formatted_nums = sorted(list(set(format_4d(n) for n in number_list if str(n).strip())))
    if len(formatted_nums) < 2:
        return [], 0, 0.0
        
    pairs = list(combinations(formatted_nums, 2))
    num_pairs = len(pairs)
    total_cost = num_pairs * ticket_cost
    
    return pairs, num_pairs, total_cost

def generate_anti_popularity_4d(count=10):
    """
    Generates 4D numbers optimized for solo / unshared payouts.
    Avoids birth years (19XX, 20XX), repetitive digits (1111, 8888), and simple sequences.
    """
    candidates = []
    attempts = 0
    
    while len(candidates) < count and attempts < 1000:
        attempts += 1
        num = random.randint(0, 9999)
        num_str = format_4d(num)
        
        # Filter 1: Birth years (1930 - 2026)
        val = int(num_str)
        if 1930 <= val <= 2026:
            continue
            
        # Filter 2: All same digits (0000, 1111, ..., 9999)
        if len(set(num_str)) == 1:
            continue
            
        # Filter 3: Common sequences (1234, 2345, 6789, 4321, etc.)
        if num_str in ["0123", "1234", "2345", "3456", "4567", "5678", "6789", "9876", "8765", "4321"]:
            continue
            
        # Filter 4: Popular lucky/unlucky strings (8888, 1688, 9999, 0007, etc.)
        if num_str in ["0168", "1688", "8888", "9999", "0007", "1168", "0888", "8880"]:
            continue
            
        if num_str not in candidates:
            candidates.append(num_str)
            
    return candidates

def analyze_4d_digit_frequencies(df_4d=None):
    """
    Calculates position-wise digit frequency matrix for positions D1, D2, D3, D4.
    """
    freq_matrix = {pos: Counter() for pos in ['D1', 'D2', 'D3', 'D4']}
    
    if df_4d is not None and not df_4d.empty:
        cols = [c for c in df_4d.columns if any(k in c.lower() for k in ['1st', '2nd', '3rd', 'special', 'consolation', 'drawn'])]
        for col in cols:
            for val in df_4d[col].dropna():
                s = format_4d(val)
                if len(s) == 4 and s.isdigit():
                    freq_matrix['D1'][s[0]] += 1
                    freq_matrix['D2'][s[1]] += 1
                    freq_matrix['D3'][s[2]] += 1
                    freq_matrix['D4'][s[3]] += 1
    else:
        digits = [str(i) for i in range(10)]
        for pos in ['D1', 'D2', 'D3', 'D4']:
            for d in digits:
                freq_matrix[pos][d] = random.randint(45, 65)
                
    return freq_matrix

def analyze_4d_patterns(df_4d=None):
    """
    Classifies historical 4D draws into digit structure patterns:
    - Single (4 unique digits e.g. 1234) -> 24-way
    - Double (1 pair e.g. 1123) -> 12-way
    - Double-Double (2 pairs e.g. 1122) -> 6-way
    - Triple (3 same digits e.g. 1112) -> 4-way
    - Quad (all same e.g. 1111) -> 1-way
    """
    patterns = Counter()
    if df_4d is not None and not df_4d.empty:
        cols = [c for c in df_4d.columns if any(k in c.lower() for k in ['1st', '2nd', '3rd'])]
        for col in cols:
            for val in df_4d[col].dropna():
                s = format_4d(val)
                if len(s) == 4 and s.isdigit():
                    counts = sorted(Counter(s).values(), reverse=True)
                    if counts == [1, 1, 1, 1]:
                        patterns["Single (24-Way)"] += 1
                    elif counts == [2, 1, 1]:
                        patterns["Double (12-Way)"] += 1
                    elif counts == [2, 2]:
                        patterns["Double-Double (6-Way)"] += 1
                    elif counts == [3, 1]:
                        patterns["Triple (4-Way)"] += 1
                    elif counts == [4]:
                        patterns["Quad (1-Way)"] += 1
    else:
        patterns = Counter({
            "Single (24-Way)": 60,
            "Double (12-Way)": 30,
            "Double-Double (6-Way)": 5,
            "Triple (4-Way)": 4,
            "Quad (1-Way)": 1
        })
    return patterns

def monte_carlo_4d(df_4d, count=5, iterations=5000):
    """
    Monte Carlo simulator for 4D digits using position-wise digit probabilities.
    """
    freq_matrix = analyze_4d_digit_frequencies(df_4d)
    digits = [str(i) for i in range(10)]
    
    weights_d1 = [freq_matrix['D1'].get(d, 0) + 1 for d in digits]
    weights_d2 = [freq_matrix['D2'].get(d, 0) + 1 for d in digits]
    weights_d3 = [freq_matrix['D3'].get(d, 0) + 1 for d in digits]
    weights_d4 = [freq_matrix['D4'].get(d, 0) + 1 for d in digits]
    
    simulations = []
    for _ in range(iterations):
        d1 = random.choices(digits, weights=weights_d1, k=1)[0]
        d2 = random.choices(digits, weights=weights_d2, k=1)[0]
        d3 = random.choices(digits, weights=weights_d3, k=1)[0]
        d4 = random.choices(digits, weights=weights_d4, k=1)[0]
        simulations.append(f"{d1}{d2}{d3}{d4}")
        
    top_candidates = [num for num, _ in Counter(simulations).most_common(count)]
    return top_candidates

def hot_due_4d(df_4d, count=5):
    """
    Generates 4D candidates by pairing hot positional digits with long-due positional digits.
    """
    freq_matrix = analyze_4d_digit_frequencies(df_4d)
    digits = [str(i) for i in range(10)]
    
    candidates = []
    for _ in range(count * 3):
        res = []
        for pos in ['D1', 'D2', 'D3', 'D4']:
            # Pick either hot or random top digit for position
            top_digits = [d for d, _ in freq_matrix[pos].most_common(4)]
            res.append(random.choice(top_digits) if top_digits else str(random.randint(0, 9)))
        cand = "".join(res)
        if cand not in candidates:
            candidates.append(cand)
        if len(candidates) >= count:
            break
            
    # Fallback if needed
    while len(candidates) < count:
        candidates.append(format_4d(random.randint(0, 9999)))
        
    return candidates[:count]
