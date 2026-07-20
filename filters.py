import numpy as np

SUM_RANGES = {
    "6/50": (90, 215),
    "6/55": (105, 230),
    "6/58": (115, 240)
}

def analyze_ticket_entropy(ticket, game_range=58):
    """
    Analyzes ticket for birthday bias, sum distribution, and delta gap entropy.
    Returns a dictionary of metrics and an Anti-Popularity (Solo Jackpot) Score.
    """
    ticket = sorted([int(n) for n in ticket])
    bday_count = sum(1 for n in ticket if n <= 31)
    ticket_sum = sum(ticket)
    
    # Delta gaps between consecutive numbers
    deltas = [ticket[i+1] - ticket[i] for i in range(len(ticket)-1)]
    unique_deltas = len(set(deltas))
    min_gap = min(deltas)
    
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
    if min_gap == 1:
        consec_count = sum(1 for g in deltas if g == 1)
        score -= consec_count * 8
        
    # Sum range check
    sum_min, sum_max = SUM_RANGES.get(f"6/{game_range}", (100, 230))
    if sum_min <= ticket_sum <= sum_max:
        score += 15
    else:
        score -= 20
        
    return {
        "ticket": ticket,
        "bday_count": bday_count,
        "sum": ticket_sum,
        "deltas": deltas,
        "unique_deltas": unique_deltas,
        "solo_jackpot_score": max(0, score)
    }

def filter_tickets(tickets, game_range=58, max_bday=4, sum_filter=True, remove_consecutive=False):
    """
    Filters a list of candidate tickets based on anti-popularity criteria.
    """
    filtered = []
    sum_min, sum_max = SUM_RANGES.get(f"6/{game_range}", (100, 230))
    
    for t in tickets:
        analysis = analyze_ticket_entropy(t, game_range)
        if analysis["bday_count"] > max_bday:
            continue
        if sum_filter and not (sum_min <= analysis["sum"] <= sum_max):
            continue
        if remove_consecutive and 1 in analysis["deltas"]:
            continue
        filtered.append(analysis)
        
    # Sort by solo jackpot score descending
    filtered = sorted(filtered, key=lambda x: x["solo_jackpot_score"], reverse=True)
    return [item["ticket"] for item in filtered]
