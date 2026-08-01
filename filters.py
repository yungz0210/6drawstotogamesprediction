import numpy as np

SUM_RANGES = {
    "6/50": (90, 215),
    "6/55": (105, 230),
    "6/58": (115, 240)
}

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59}

def analyze_ticket_entropy(ticket, game_range=58, last_draw=None):
    """
    Analyzes ticket for birthday bias, sum distribution, prime count, last draw repeats, and delta gap entropy.
    Returns a dictionary of metrics and an Anti-Popularity (Solo Jackpot) Score.
    """
    ticket = sorted([int(n) for n in ticket])
    bday_count = sum(1 for n in ticket if n <= 31)
    ticket_sum = sum(ticket)
    prime_count = sum(1 for n in ticket if n in PRIMES)
    
    # Delta gaps between consecutive numbers
    deltas = [ticket[i+1] - ticket[i] for i in range(len(ticket)-1)]
    unique_deltas = len(set(deltas))
    min_gap = min(deltas) if deltas else 99
    consec_count = sum(1 for g in deltas if g == 1)
    
    # Last draw repeat count
    repeat_count = 0
    if last_draw is not None:
        last_set = set(int(n) for n in last_draw)
        repeat_count = len(set(ticket).intersection(last_set))
    
    # Anti-Popularity (Solo Jackpot) Score calculation
    # Base score 100
    score = 100
    # Penalty for birthday saturation (numbers <= 31)
    score -= bday_count * 12
    # Bonus for numbers > 31
    score += (6 - bday_count) * 10
    # Bonus for unique gap diversity (avoiding arithmetic sequences like 5,10,15...)
    score += unique_deltas * 6
    # Penalty for consecutive numbers (e.g. 12, 13, 14)
    if consec_count > 0:
        score -= consec_count * 10
        
    # Penalty for too many repeats from last draw
    if repeat_count > 2:
        score -= (repeat_count - 2) * 15
        
    # Sum range check
    sum_min, sum_max = SUM_RANGES.get(f"6/{game_range}", (100, 230))
    if sum_min <= ticket_sum <= sum_max:
        score += 15
    else:
        score -= 20
        
    return {
        "ticket": ticket,
        "bday_count": bday_count,
        "prime_count": prime_count,
        "sum": ticket_sum,
        "deltas": deltas,
        "unique_deltas": unique_deltas,
        "consec_count": consec_count,
        "repeat_count": repeat_count,
        "solo_jackpot_score": max(0, score)
    }

def filter_tickets(tickets, game_range=58, max_bday=4, sum_filter=True, remove_consecutive=False, max_consecutive=2, max_repeat=2, last_draw=None):
    """
    Filters a list of candidate tickets based on anti-popularity and statistical criteria.
    """
    filtered = []
    sum_min, sum_max = SUM_RANGES.get(f"6/{game_range}", (100, 230))
    
    for t in tickets:
        analysis = analyze_ticket_entropy(t, game_range, last_draw=last_draw)
        
        if analysis["bday_count"] > max_bday:
            continue
        if sum_filter and not (sum_min <= analysis["sum"] <= sum_max):
            continue
        if remove_consecutive and analysis["consec_count"] > 0:
            continue
        if analysis["consec_count"] > max_consecutive:
            continue
        if last_draw is not None and analysis["repeat_count"] > max_repeat:
            continue
            
        filtered.append(analysis)
        
    # Sort by solo jackpot score descending
    filtered = sorted(filtered, key=lambda x: x["solo_jackpot_score"], reverse=True)
    return [item["ticket"] for item in filtered]

