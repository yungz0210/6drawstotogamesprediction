import wheeling
import filters
import ml_model
import toto4d_studio
import data_manager

def test_banker_wheeling():
    pool = [1, 5, 12, 18, 23, 30, 35, 42, 49, 50]
    bankers = [5, 12]
    tickets = wheeling.generate_key_number_wheel(pool, bankers, target_match=4, pool_match=4)
    assert len(tickets) > 0, "No tickets generated"
    for t in tickets:
        assert 5 in t and 12 in t, f"Banker numbers missing in ticket {t}"
    print(f"[OK] Banker wheeling test passed! Pool: {len(pool)}, Bankers: {bankers}, Generated tickets: {len(tickets)}")

def test_4d_permutations():
    perms, label, cost = toto4d_studio.generate_4d_permutations("1234")
    assert len(perms) == 24, f"Expected 24 perms, got {len(perms)}"
    
    perms_12, label_12, cost_12 = toto4d_studio.generate_4d_permutations("1123")
    assert len(perms_12) == 12, f"Expected 12 perms, got {len(perms_12)}"
    
    print(f"[OK] 4D permutation test passed! '1234' -> {label} ({len(perms)} perms), '1123' -> {label_12} ({len(perms_12)} perms)")

def test_4d_jackpot_pairs():
    pool_4d = ["1234", "5678", "8888", "0168"]
    pairs, num_pairs, cost = toto4d_studio.generate_system_4d_jackpot(pool_4d)
    assert num_pairs == 6, f"Expected C(4,2)=6 pairs, got {num_pairs}"
    print(f"[OK] 4D Jackpot pair test passed! Pool size: {len(pool_4d)}, Pairs generated: {num_pairs}, Cost: RM {cost}")

def test_filters():
    tickets = [
        [1, 2, 3, 4, 5, 6],     # Consec = 5
        [5, 12, 18, 23, 35, 42], # Good set
        [10, 11, 12, 13, 14, 15] # Consec = 5
    ]
    last_draw = [5, 12, 99, 98, 97, 96]
    filtered = filters.filter_tickets(tickets, game_range=50, max_consecutive=2, max_repeat=2, last_draw=last_draw)
    assert [5, 12, 18, 23, 35, 42] in filtered, "Valid set was filtered out"
    assert [1, 2, 3, 4, 5, 6] not in filtered, "High consecutive set was not filtered out"
    print(f"[OK] Filter test passed! Input tickets: {len(tickets)}, Filtered count: {len(filtered)}")

if __name__ == "__main__":
    test_banker_wheeling()
    test_4d_permutations()
    test_4d_jackpot_pairs()
    test_filters()
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")

