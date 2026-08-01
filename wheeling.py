from itertools import combinations
import pandas as pd

def generate_full_wheel(numbers):
    """
    Generates all possible 6-number ticket combinations from a given pool of numbers.
    """
    numbers = sorted(list(set(int(n) for n in numbers)))
    if len(numbers) < 6:
        raise ValueError("Number pool must contain at least 6 numbers.")
    return [list(t) for t in combinations(numbers, 6)]

def generate_abbreviated_wheel(numbers, target_match=4, pool_match=4):
    """
    Generates an optimized abbreviated wheel using greedy covering.
    Guarantees at least 1 ticket hits 'target_match' matching numbers if 'pool_match' numbers
    from the winning draw fall inside your selected number pool.
    """
    numbers = sorted(list(set(int(n) for n in numbers)))
    if len(numbers) < 6:
        raise ValueError("Number pool must contain at least 6 numbers.")
    
    all_tickets = list(combinations(numbers, 6))
    if len(numbers) <= 6:
        return [list(t) for t in all_tickets]
    
    # Target subset size must be <= 6 and <= pool_match
    target_match = min(target_match, 6)
    pool_match = min(pool_match, len(numbers))
    
    # Subsets to cover from the pool
    subsets_to_cover = set(combinations(numbers, pool_match))
    selected_tickets = []
    
    ticket_coverage = {t: set(combinations(t, target_match)) for t in all_tickets}
    uncovered = set(subsets_to_cover)
    
    while uncovered and len(selected_tickets) < len(all_tickets):
        # Select ticket covering the maximum number of remaining uncovered combinations
        best_ticket = max(all_tickets, key=lambda t: len(ticket_coverage[t].intersection(uncovered)))
        new_covers = ticket_coverage[best_ticket].intersection(uncovered)
        if not new_covers:
            # Fallback: add remaining tickets if needed
            break
        selected_tickets.append(list(best_ticket))
        uncovered -= new_covers
        
    return selected_tickets if selected_tickets else [list(t) for t in all_tickets[:1]]

def generate_key_number_wheel(numbers, key_numbers, target_match=4, pool_match=4):
    """
    Generates a Banker / Key Number wheel.
    Every generated ticket is guaranteed to contain ALL key_numbers.
    The remaining slots (6 - len(key_numbers)) are filled from non-key numbers in the pool.
    """
    numbers = sorted(list(set(int(n) for n in numbers)))
    key_numbers = sorted(list(set(int(k) for k in key_numbers if int(k) in numbers)))
    
    if len(key_numbers) == 0:
        return generate_abbreviated_wheel(numbers, target_match, pool_match)
    if len(key_numbers) >= 6:
        return [key_numbers[:6]]
        
    non_keys = [n for n in numbers if n not in key_numbers]
    needed_slots = 6 - len(key_numbers)
    
    if len(non_keys) < needed_slots:
        raise ValueError(f"Need at least {needed_slots} non-key numbers in pool.")
        
    # Generate combinations for remaining slots from non_keys
    if target_match >= 6 or len(non_keys) <= needed_slots:
        slot_combs = list(combinations(non_keys, needed_slots))
    else:
        # Abbreviate non-key slots
        sub_target = max(1, target_match - len(key_numbers))
        sub_pool = max(sub_target, pool_match - len(key_numbers))
        all_slot_combs = list(combinations(non_keys, needed_slots))
        subsets_to_cover = set(combinations(non_keys, min(sub_pool, len(non_keys))))
        coverage = {sc: set(combinations(sc, min(sub_target, needed_slots))) for sc in all_slot_combs}
        uncovered = set(subsets_to_cover)
        slot_combs = []
        while uncovered and len(slot_combs) < len(all_slot_combs):
            best = max(all_slot_combs, key=lambda sc: len(coverage[sc].intersection(uncovered)))
            new_c = coverage[best].intersection(uncovered)
            if not new_c:
                break
            slot_combs.append(best)
            uncovered -= new_c
        if not slot_combs:
            slot_combs = all_slot_combs[:1]

    # Combine key_numbers with non-key slot combinations
    wheeled_tickets = []
    for sc in slot_combs:
        t = sorted(key_numbers + list(sc))
        wheeled_tickets.append(t)
        
    return wheeled_tickets

def get_wheel_summary(numbers, tickets, ticket_cost=2.0, key_numbers=None):
    """
    Calculates cost and coverage summary for a generated wheel.
    """
    num_tickets = len(tickets)
    total_cost = num_tickets * ticket_cost
    full_count = len(list(combinations(numbers, 6)))
    savings_pct = (1.0 - (num_tickets / full_count)) * 100 if full_count > 0 else 0
    
    res = {
        "pool_size": len(numbers),
        "total_tickets": num_tickets,
        "full_wheel_tickets": full_count,
        "total_cost_rm": total_cost,
        "savings_percentage": savings_pct
    }
    if key_numbers:
        res["key_numbers_count"] = len(key_numbers)
    return res

