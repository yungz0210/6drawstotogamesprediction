import os
import json
import uuid
from datetime import datetime
import pandas as pd

TRACKER_FILE = os.path.join("data", "tracker_store.json")

def _ensure_dir():
    os.makedirs("data", exist_ok=True)

def load_tracker_data():
    """Loads saved tickets from JSON storage file."""
    _ensure_dir()
    if not os.path.exists(TRACKER_FILE):
        return []
    try:
        with open(TRACKER_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        print(f"Error reading tracker file: {e}")
        return []

def save_tracker_data(tickets):
    """Saves tickets list to JSON storage file."""
    _ensure_dir()
    try:
        with open(TRACKER_FILE, "w", encoding="utf-8") as f:
            json.dump(tickets, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving tracker file: {e}")
        return False

def add_ticket(game_type, numbers, bonus=None, strategy="Manual", notes="", target_draw_date=None, play_type="Lotto"):
    """
    Adds a new ticket to the persistent tracker.
    numbers: list of ints for Lotto (e.g. [3, 12, 19, 28, 35, 42]) or string/list of 4D numbers (e.g. ['1234'] or ['1234', '5678']).
    """
    tickets = load_tracker_data()
    
    # Standardize numbers
    if play_type == "Lotto":
        clean_nums = sorted([int(n) for n in numbers])
        cost = 2.0
    elif play_type == "4D":
        if isinstance(numbers, str):
            clean_nums = [numbers.zfill(4)[-4:]]
        else:
            clean_nums = [str(n).zfill(4)[-4:] for n in numbers]
        cost = float(len(clean_nums)) # RM 1 per 4D permutation/direct
    elif play_type == "4D Jackpot":
        # Expect pair of 2 numbers
        clean_nums = [str(n).zfill(4)[-4:] for n in numbers]
        cost = 2.0 # RM 2 per pair
    else:
        clean_nums = list(numbers)
        cost = 2.0

    ticket_id = str(uuid.uuid4())[:8]
    new_ticket = {
        "id": ticket_id,
        "game_type": game_type, # "6/50", "6/55", "6/58", "4D", "4D Jackpot"
        "play_type": play_type,
        "numbers": clean_nums,
        "bonus": int(bonus) if bonus is not None else None,
        "strategy": strategy,
        "notes": notes,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_draw_date": target_draw_date if target_draw_date else datetime.now().strftime("%Y-%m-%d"),
        "cost": cost,
        "status": "Pending"
    }
    
    tickets.insert(0, new_ticket) # Most recent first
    save_tracker_data(tickets)
    return new_ticket

def delete_ticket(ticket_id):
    """Deletes a ticket by ID."""
    tickets = load_tracker_data()
    filtered = [t for t in tickets if t.get("id") != ticket_id]
    save_tracker_data(filtered)
    return len(filtered) < len(tickets)

def clear_all_tickets():
    """Clears all stored tickets."""
    return save_tracker_data([])

def parse_manual_input(input_str, play_type="Lotto"):
    """
    Parses raw user typed string into structured numbers.
    Supports space, comma, newline, dash delimiters.
    """
    if not input_str or not input_str.strip():
        return None, "Input string is empty."
        
    raw = input_str.replace(",", " ").replace("-", " ").replace("\n", " ")
    tokens = [t.strip() for t in raw.split() if t.strip()]
    
    if play_type == "Lotto":
        nums = []
        for t in tokens:
            if t.isdigit():
                val = int(t)
                nums.append(val)
        nums = sorted(list(set(nums)))
        if len(nums) < 6:
            return None, f"Expected at least 6 unique numbers between 1 and 58. Got {len(nums)} valid numbers."
        return nums[:6], None
        
    elif play_type == "4D":
        nums = []
        for t in tokens:
            s = t.zfill(4)[-4:]
            if len(s) == 4 and s.isdigit():
                nums.append(s)
        if not nums:
            return None, "Please enter a valid 4-digit number (e.g. 1234)."
        return nums, None

    elif play_type == "4D Jackpot":
        nums = []
        for t in tokens:
            s = t.zfill(4)[-4:]
            if len(s) == 4 and s.isdigit():
                nums.append(s)
        if len(nums) < 2:
            return None, "Toto 4D Jackpot requires at least two 4D numbers (e.g. 1234, 5678)."
        return nums[:2], None
        
    return tokens, None

def evaluate_single_ticket(ticket, df_lotto=None, df_4d=None):
    """
    Evaluates a single tracked ticket against available historical/latest draws.
    Returns evaluated dict with match count, matched numbers, prize won, and match label.
    """
    res = dict(ticket)
    g_type = ticket.get("game_type", "6/50")
    p_type = ticket.get("play_type", "Lotto")
    numbers = ticket.get("numbers", [])
    bonus = ticket.get("bonus")
    
    res["matches"] = 0
    res["matched_numbers"] = []
    res["bonus_match"] = False
    res["prize"] = 0.0
    res["badge_label"] = "No Match"
    res["badge_color"] = "gray"
    
    if p_type == "Lotto" and df_lotto is not None and not df_lotto.empty:
        # Match against latest or target draw date
        target_date_str = ticket.get("target_draw_date")
        matching_rows = df_lotto
        if target_date_str:
            dt_target = pd.to_datetime(target_date_str, errors='coerce')
            if dt_target is not None and not pd.isna(dt_target):
                matching_rows = df_lotto[df_lotto['DrawDate'].dt.date <= dt_target.date()]
        
        if matching_rows.empty:
            matching_rows = df_lotto
            
        latest_row = matching_rows.iloc[0]
        actual_draw_date = latest_row['DrawDate'].strftime('%Y-%m-%d')
        res["evaluated_draw_date"] = actual_draw_date
        
        main_cols = ['DrawnNo1', 'DrawnNo2', 'DrawnNo3', 'DrawnNo4', 'DrawnNo5', 'DrawnNo6']
        actual_set = set(latest_row[main_cols].values)
        user_set = set(numbers)
        matched = sorted(list(user_set.intersection(actual_set)))
        
        res["matches"] = len(matched)
        res["matched_numbers"] = matched
        
        # Bonus check
        if 'BonusNo' in latest_row and pd.notna(latest_row['BonusNo']):
            actual_bonus = int(latest_row['BonusNo'])
            if bonus is not None and bonus == actual_bonus:
                res["bonus_match"] = True
                
        # Prize Calculation
        m = res["matches"]
        bm = res["bonus_match"]
        if m == 6:
            res["prize"] = 3000000.0 # Estimated jackpot
            res["badge_label"] = "🏆 JACKPOT 1 WINNER!"
            res["badge_color"] = "#ffd700"
        elif m == 5 and bm:
            res["prize"] = 100000.0
            res["badge_label"] = "🔥 5 + Bonus Match!"
            res["badge_color"] = "#ff5722"
        elif m == 5:
            res["prize"] = 2000.0
            res["badge_label"] = "⭐ 5 Main Match"
            res["badge_color"] = "#ff9800"
        elif m == 4:
            res["prize"] = 100.0
            res["badge_label"] = "🟡 4 Main Match"
            res["badge_color"] = "#ffeb3b"
        elif m == 3:
            res["prize"] = 20.0
            res["badge_label"] = "🟢 3 Main Match"
            res["badge_color"] = "#4caf50"
        else:
            res["badge_label"] = "⚪ 0-2 Matches"
            res["badge_color"] = "#9e9e9e"

    elif (p_type == "4D" or p_type == "4D Jackpot") and df_4d is not None and not df_4d.empty:
        latest_row = df_4d.iloc[0]
        actual_draw_date = pd.to_datetime(latest_row['DrawDate']).strftime('%Y-%m-%d')
        res["evaluated_draw_date"] = actual_draw_date
        
        # Get winning 4D numbers
        p1 = str(latest_row.get('1stPrize', '')).zfill(4)[-4:]
        p2 = str(latest_row.get('2ndPrize', '')).zfill(4)[-4:]
        p3 = str(latest_row.get('3rdPrize', '')).zfill(4)[-4:]
        
        top3_set = {p1, p2, p3}
        
        specials = set(str(latest_row.get(f'Special{i}', '')).zfill(4)[-4:] for i in range(1, 11))
        consolations = set(str(latest_row.get(f'Consolation{i}', '')).zfill(4)[-4:] for i in range(1, 11))
        
        if p_type == "4D":
            user_num = numbers[0] if numbers else ""
            if user_num == p1:
                res["matches"] = 4
                res["prize"] = 2500.0
                res["badge_label"] = "🥇 1st Prize Winner!"
                res["badge_color"] = "#ffd700"
            elif user_num == p2:
                res["matches"] = 4
                res["prize"] = 1000.0
                res["badge_label"] = "🥈 2nd Prize Winner!"
                res["badge_color"] = "#c0c0c0"
            elif user_num == p3:
                res["matches"] = 4
                res["prize"] = 500.0
                res["badge_label"] = "🥉 3rd Prize Winner!"
                res["badge_color"] = "#cd7f32"
            elif user_num in specials:
                res["matches"] = 4
                res["prize"] = 180.0
                res["badge_label"] = "⭐ Special Prize!"
                res["badge_color"] = "#00bcd4"
            elif user_num in consolations:
                res["matches"] = 4
                res["prize"] = 60.0
                res["badge_label"] = "🔹 Consolation Prize!"
                res["badge_color"] = "#2196f3"
            else:
                res["badge_label"] = "⚪ No 4D Hit"
                res["badge_color"] = "#9e9e9e"

        elif p_type == "4D Jackpot":
            if len(numbers) >= 2:
                num1, num2 = numbers[0], numbers[1]
                # Jackpot 1: matches 2 of Top 3
                if {num1, num2}.issubset(top3_set):
                    res["matches"] = 2
                    res["prize"] = 2000000.0
                    res["badge_label"] = "👑 4D JACKPOT 1 WINNER!"
                    res["badge_color"] = "#ffd700"
                # Jackpot 2: matches 1 of Top 3 + 1 Special/Consolation
                elif (num1 in top3_set and (num2 in specials or num2 in consolations)) or \
                     (num2 in top3_set and (num1 in specials or num1 in consolations)):
                    res["matches"] = 2
                    res["prize"] = 100000.0
                    res["badge_label"] = "🔥 4D JACKPOT 2 WINNER!"
                    res["badge_color"] = "#ff5722"
                else:
                    res["badge_label"] = "⚪ No Jackpot Pair Hit"
                    res["badge_color"] = "#9e9e9e"

    return res

def get_tracker_summary(evaluated_tickets):
    """Computes summary stats across all evaluated tracked tickets."""
    total_tickets = len(evaluated_tickets)
    if total_tickets == 0:
        return {
            "total_tickets": 0,
            "total_cost": 0.0,
            "total_prizes": 0.0,
            "net_pl": 0.0,
            "roi_pct": 0.0,
            "winning_tickets": 0,
            "hit_rate_pct": 0.0
        }
        
    total_cost = sum(t.get("cost", 2.0) for t in evaluated_tickets)
    total_prizes = sum(t.get("prize", 0.0) for t in evaluated_tickets)
    net_pl = total_prizes - total_cost
    roi_pct = (net_pl / total_cost * 100.0) if total_cost > 0 else 0.0
    
    winning_tickets = sum(1 for t in evaluated_tickets if t.get("prize", 0.0) > 0)
    hit_rate = (winning_tickets / total_tickets * 100.0) if total_tickets > 0 else 0.0
    
    return {
        "total_tickets": total_tickets,
        "total_cost": total_cost,
        "total_prizes": total_prizes,
        "net_pl": net_pl,
        "roi_pct": roi_pct,
        "winning_tickets": winning_tickets,
        "hit_rate_pct": hit_rate
    }
