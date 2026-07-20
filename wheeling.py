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
    if len(numbers) <= 7:
        return [list(t) for t in all_tickets]
    
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
            break
        selected_tickets.append(list(best_ticket))
        uncovered -= new_covers
        
    return selected_tickets

def get_wheel_summary(numbers, tickets, ticket_cost=2.0):
    """
    Calculates cost and coverage summary for a generated wheel.
    """
    num_tickets = len(tickets)
    total_cost = num_tickets * ticket_cost
    full_count = len(list(combinations(numbers, 6)))
    savings_pct = (1.0 - (num_tickets / full_count)) * 100 if full_count > 0 else 0
    
    return {
        "pool_size": len(numbers),
        "total_tickets": num_tickets,
        "full_wheel_tickets": full_count,
        "total_cost_rm": total_cost,
        "savings_percentage": savings_pct
    }
