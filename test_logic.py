import data_manager
import analytics
import predictor
import probability_lab
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

    # Bonus prediction test
    for model in ["Monte Carlo Simulation", "Mean Reversion (Due)", "Markov Chain Analysis", "Hybrid/Ensemble Model"]:
        bonus = predictor.predict_bonus_number(df, 50, model)
        assert 1 <= bonus <= 50
        print(f"Bonus prediction ({model}) test passed: {bonus}")

def test_probability_lab():
    print("Testing probability lab...")
    df = data_manager.load_data("6/50")

    # Test evaluation
    m, hb, prize = probability_lab.evaluate_prediction([1, 2, 3, 10, 11, 12], [1, 2, 3, 4, 5, 6], 50, 50)
    assert m == 3
    assert hb == True
    assert prize == 20
    print("Evaluation test passed.")

    # Test EV
    ev = probability_lab.calculate_ev("6/50", 10000000)
    assert ev > 0
    print("EV calculation test passed.")

    # Test Macro Backtest
    res = probability_lab.backtest_macro(df, 50, "Mean Reversion (Due)", lookback=5)
    assert "total_prize" in res
    assert res["total_cost"] == 10.0
    print("Macro backtest test passed.")

if __name__ == "__main__":
    try:
        test_data_loading()
        test_analytics()
        test_predictor()
        test_probability_lab()
        print("All tests passed!")
    except Exception as e:
        print(f"Tests failed: {e}")
        exit(1)
