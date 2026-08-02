import data_manager
import analytics
import predictor
import probability_lab
import ticket_tracker
import toto4d_studio
import pandas as pd

def test_data_loading():
    print("Testing data loading...")
    for game in ["6/50", "6/55", "6/58"]:
        df = data_manager.load_data(game)
        print(f"{game} Loaded: {len(df)} rows")
        assert len(df) > 0
        if game == "6/50":
            assert "BonusNo" in df.columns
            print("BonusNo present for 6/50")
    print("Data loading test passed.")

def test_analytics():
    print("Testing analytics...")
    df = data_manager.load_data("6/50")
    
    freq = analytics.get_frequency(df, lookback=10)
    assert len(freq) > 0
    print("Frequency test passed.")
    
    pairs = analytics.get_pairs(df, lookback=10)
    assert len(pairs) > 0
    print("Pairs test passed.")
    
    oe = analytics.get_odd_even_ratio(df, lookback=10)
    assert len(oe) > 0
    print("Odd/Even ratio test passed.")

    df_3m = analytics.filter_by_timeframe(df, "3 Months")
    df_1y = analytics.filter_by_timeframe(df, "1 Year")
    df_5y = analytics.filter_by_timeframe(df, "5 Years")
    df_all = analytics.filter_by_timeframe(df, "All Time")

    assert len(df_3m) <= len(df_1y) <= len(df_5y) <= len(df_all)
    assert len(df_all) == len(df)
    print("Timeframe preset filtering test passed.")

def test_predictor():
    print("Testing predictor...")
    df = data_manager.load_data("6/50")
    
    mc = predictor.monte_carlo_simulation(df, 50, iterations=100)
    assert len(mc) == 5
    print("Monte Carlo test passed.")
    
    due = predictor.mean_reversion_due(df, 50)
    assert len(due) == 6
    print("Mean Reversion test passed.")
    
    markov = predictor.markov_chain_analysis(df, 50)
    assert len(markov) == 6
    print("Markov Chain test passed.")
    
    hybrid = predictor.hybrid_ensemble(df, 50)
    assert len(hybrid) == 6
    print("Hybrid test passed.")

    for model in ["Monte Carlo Simulation", "Mean Reversion (Due)", "Markov Chain Analysis", "Hybrid/Ensemble Model"]:
        bonus = predictor.predict_bonus_number(df, 50, model)
        assert 1 <= bonus <= 50
        print(f"Bonus prediction ({model}) test passed: {bonus}")

def test_4d_studio():
    print("Testing 4D studio...")
    df_4d = data_manager.load_4d_data()
    assert not df_4d.empty
    
    patterns = toto4d_studio.analyze_4d_patterns(df_4d)
    assert len(patterns) > 0
    print("4D pattern analysis passed.")
    
    mc_4d = toto4d_studio.monte_carlo_4d(df_4d, count=3)
    assert len(mc_4d) == 3
    print("4D Monte Carlo passed.")
    
    hd_4d = toto4d_studio.hot_due_4d(df_4d, count=3)
    assert len(hd_4d) == 3
    print("4D Hot/Due passed.")

def test_ticket_tracker():
    print("Testing ticket tracker...")
    # Add ticket
    t = ticket_tracker.add_ticket("6/50", [1, 5, 12, 18, 25, 33], bonus=8, strategy="Unit Test", play_type="Lotto")
    assert t["numbers"] == [1, 5, 12, 18, 25, 33]
    
    df_lotto = data_manager.load_data("6/50")
    df_4d = data_manager.load_4d_data()
    
    eval_t = ticket_tracker.evaluate_single_ticket(t, df_lotto=df_lotto, df_4d=df_4d)
    assert "matches" in eval_t
    assert "prize" in eval_t
    
    # Clean up test ticket
    ticket_tracker.delete_ticket(t["id"])
    print("Ticket tracker test passed.")

if __name__ == "__main__":
    try:
        test_data_loading()
        test_analytics()
        test_predictor()
        test_4d_studio()
        test_ticket_tracker()
        print("=== ALL LOGIC TESTS PASSED SUCCESSFULLY! ===")
    except Exception as e:
        print(f"Tests failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
