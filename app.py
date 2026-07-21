import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import data_manager
import analytics
import predictor
import probability_lab
import wheeling
import filters
import ml_model
from datetime import datetime

st.set_page_config(page_title="Sports Toto Analytics & Prediction Studio", layout="wide", page_icon="🎰")

# Automatic data update on startup
@st.cache_resource
def startup_update():
    data_manager.download_and_extract()
    return datetime.now()

startup_update()

# Sidebar
st.sidebar.title("🎰 Global Controls")
game_selection = st.sidebar.selectbox("Select Game", ["6/50", "6/55", "6/58"])
game_ranges = {"6/50": 50, "6/55": 55, "6/58": 58}
game_range = game_ranges[game_selection]

# Data Loading
@st.cache_data(show_spinner="Loading data...")
def get_data(game):
    return data_manager.load_data(game)

@st.cache_data(show_spinner="Fetching zip data...")
def get_zip_data(url):
    return data_manager.fetch_zip_bytes(url)

@st.cache_data(ttl=3600, show_spinner="Scraping live jackpots...")
def load_live_jackpots():
    return data_manager.get_live_jackpots()

if st.sidebar.button("🔄 Update Data Daily"):
    data_manager.download_and_extract()
    st.cache_data.clear()
    st.sidebar.success("Data updated!")

df_all = get_data(game_selection)

# Display Latest Draw Date
latest_date = df_all['DrawDate'].max()
st.sidebar.info(f"📅 Latest Draw Date: {latest_date.strftime('%Y-%m-%d')}")

# Live Jackpot & EV Alert Widget
st.sidebar.subheader("💰 Live Estimated Jackpots")
live_jackpots = load_live_jackpots()

jackpot_value = 2000000.0
if live_jackpots and game_selection in live_jackpots:
    g_info = live_jackpots[game_selection]
    if game_selection == "6/50":
        st.sidebar.markdown(f"**Jackpot 1:** `{g_info.get('raw_j1', 'N/A')}`")
        st.sidebar.markdown(f"**Jackpot 2:** `{g_info.get('raw_j2', 'N/A')}`")
        jackpot_value = g_info.get('jackpot1', 2000000.0)
    else:
        st.sidebar.markdown(f"**Jackpot:** `{g_info.get('raw', 'N/A')}`")
        jackpot_value = g_info.get('jackpot', 2000000.0)
else:
    jackpot_value = st.sidebar.number_input("Estimated Jackpot (RM)", min_value=1000000, value=15000000, step=1000000)

ev_val = probability_lab.calculate_ev(game_selection, jackpot_value)
st.sidebar.markdown(f"**Calculated Ticket EV:** `RM {ev_val:.4f}`")
if ev_val > 2.0:
    st.sidebar.success("🟢 **BUY SIGNAL!** Positive Expected Value opportunity!")
else:
    st.sidebar.caption("🔴 Negative EV (Jackpot building...)")

# Download buttons
st.sidebar.subheader("📥 Download Source Data")
for game, url in data_manager.URLS.items():
    zip_content = get_zip_data(url)
    if zip_content:
        st.sidebar.download_button(
            label=f"Download {game} Zip",
            data=zip_content,
            file_name=f"Toto{game.replace('/', '')}.zip",
            mime="application/zip"
        )

# Filters
st.sidebar.subheader("🔍 Date & Timeframe Filters")

timeframe_presets = ["All Time", "10 Years", "5 Years", "3 Years", "1 Year", "6 Months", "3 Months"]
selected_timeframe = st.sidebar.selectbox("⚡ Quick Timeframe Preset", timeframe_presets, index=0, key=f"timeframe_{game_selection}")

# Apply timeframe preset first
df = analytics.filter_by_timeframe(df_all, selected_timeframe)

# Additional date fine-tuning
with st.sidebar.expander("⚙️ Fine-Tune Date Filters"):
    years = sorted(df_all['DrawDate'].dt.year.unique(), reverse=True)
    selected_year = st.selectbox("Year", ["All"] + list(years), key=f"year_{game_selection}")
    months = list(range(1, 13))
    selected_month = st.selectbox("Month", ["All"] + months, key=f"month_{game_selection}")
    dows = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    selected_dow = st.selectbox("Day of Week", ["All"] + dows, key=f"dow_{game_selection}")

if selected_year != "All":
    df = analytics.filter_by_date(df, year=selected_year)
if selected_month != "All":
    df = analytics.filter_by_date(df, month=selected_month)
if selected_dow != "All":
    df = analytics.filter_by_date(df, dow=dows.index(selected_dow))

max_lookback = len(df) if len(df) >= 10 else 10
lookback = st.sidebar.number_input("Lookback Period (Draws)", min_value=10, max_value=max_lookback, value=max_lookback, key=f"lookback_{game_selection}")

if not df.empty:
    min_date_str = df['DrawDate'].min().strftime('%Y-%m-%d')
    max_date_str = df['DrawDate'].max().strftime('%Y-%m-%d')
    st.sidebar.caption(f"📊 **Active Data:** `{len(df)} draws`\n📅 `{min_date_str}` to `{max_date_str}`")


# Train ML Evaluator once per session/game
@st.cache_resource
def get_ml_model(game):
    g_df = get_data(game)
    return ml_model.train_ml_evaluator(g_df, game_ranges[game])

trained_ml = get_ml_model(game_selection)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Insights & Stats", 
    "🔮 Predictor Studio", 
    "🎡 Combinatorial Wheeling", 
    "🏆 Master Summary", 
    "🧪 Backtesting & Probability Lab"
])

# ----------------- TAB 1: Insights & Stats -----------------
with tab1:
    st.header(f"Insights & Statistics - {game_selection}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hot & Cold Number Frequencies")
        freq = analytics.get_frequency(df, lookback)
        freq_df = pd.DataFrame.from_dict(freq, orient='index', columns=['Count']).reset_index()
        freq_df.columns = ['Number', 'Count']
        freq_df = freq_df.sort_values('Count', ascending=False)
        
        fig_freq = px.bar(freq_df, x='Number', y='Count', title="Number Frequency Distribution", color='Count', color_continuous_scale='Viridis')
        st.plotly_chart(fig_freq, use_container_width=True)
        
        if game_selection == "6/50":
            st.subheader("Bonus Number Frequency")
            b_freq = analytics.get_bonus_frequency(df, lookback)
            if b_freq:
                b_freq_df = pd.DataFrame.from_dict(b_freq, orient='index', columns=['Count']).reset_index()
                b_freq_df.columns = ['Bonus Number', 'Count']
                b_freq_df = b_freq_df.sort_values('Count', ascending=False)
                fig_b_freq = px.bar(b_freq_df, x='Bonus Number', y='Count', title="Bonus Number Frequency", color_discrete_sequence=['orange'])
                st.plotly_chart(fig_b_freq, use_container_width=True)

    with col2:
        st.subheader("Odd vs Even Ratio Distribution")
        oe_ratio = analytics.get_odd_even_ratio(df, lookback)
        fig_oe = px.pie(values=oe_ratio.values, names=oe_ratio.index, title="Odd:Even Balance", hole=0.4)
        st.plotly_chart(fig_oe, use_container_width=True)
        
        st.subheader("Sum Total Bell Curve Analysis")
        sums = analytics.get_sum_analysis(df, lookback)
        fig_sum = px.histogram(sums, nbins=25, title="Distribution of Draw Sum Totals", color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig_sum, use_container_width=True)

    st.subheader("Common Co-occurring Pairs & Triplets")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        pairs = analytics.get_pairs(df, lookback)
        formatted_pairs = [ (", ".join(map(str, p)), f) for p, f in pairs ]
        pairs_df = pd.DataFrame(formatted_pairs, columns=['Pair', 'Frequency'])
        st.write("Top 20 Pairs")
        st.dataframe(pairs_df, use_container_width=True)
    with p_col2:
        triplets = analytics.get_triplets(df, lookback)
        formatted_triplets = [ (", ".join(map(str, t)), f) for t, f in triplets ]
        triplets_df = pd.DataFrame(formatted_triplets, columns=['Triplet', 'Frequency'])
        st.write("Top 20 Triplets")
        st.dataframe(triplets_df, use_container_width=True)

# ----------------- TAB 2: Predictor Studio -----------------
with tab2:
    st.header("🔮 Predictor Studio")
    st.caption("Generate optimized ticket combinations using statistical, ensemble, and anti-popularity strategies.")
    
    col_m1, col_m2 = st.columns([2, 1])
    with col_m1:
        model_choice = st.selectbox(
            "Select Prediction Strategy", 
            [
                "Anti-Popularity / Solo Jackpot Strategy",
                "Hybrid / Ensemble Model",
                "Monte Carlo Simulation", 
                "Mean Reversion (Due)", 
                "Markov Chain Analysis"
            ]
        )
    with col_m2:
        max_bday_filter = st.slider("Max Birthday Numbers (<=31)", min_value=1, max_value=6, value=3)

    if st.button("🚀 Generate Candidate Ticket Sets", type="primary"):
        sets_to_display = []
        
        if model_choice == "Anti-Popularity / Solo Jackpot Strategy":
            sets_to_display = predictor.anti_popularity_model(df, game_range, count=5)
        elif model_choice == "Monte Carlo Simulation":
            raw_sets = predictor.monte_carlo_simulation(df, game_range)
            sets_to_display = filters.filter_tickets(raw_sets, game_range=game_range, max_bday=max_bday_filter)
            if not sets_to_display: sets_to_display = raw_sets
        elif model_choice == "Mean Reversion (Due)":
            raw = predictor.mean_reversion_due(df, game_range)
            sets_to_display = [raw]
        elif model_choice == "Markov Chain Analysis":
            raw = predictor.markov_chain_analysis(df, game_range)
            sets_to_display = [raw]
        else: # Hybrid
            sets_to_display = [predictor.hybrid_ensemble(df, game_range) for _ in range(5)]

        st.subheader("🎯 Recommended Candidate Ticket Sets")
        
        export_rows = []
        for i, ticket in enumerate(sets_to_display):
            analysis = filters.analyze_ticket_entropy(ticket, game_range)
            ml_score = ml_model.predict_ticket_ml_score(trained_ml, ticket, game_range)
            
            bonus_str = ""
            if game_selection == "6/50":
                b_num = predictor.predict_bonus_number(df, game_range, model_choice)
                bonus_str = f" | Bonus: **{b_num}**"

            formatted_set = ", ".join(f"{n:02d}" for n in ticket)
            
            col_t1, col_t2, col_t3 = st.columns([3, 1, 1])
            with col_t1:
                st.markdown(f"**Ticket Set {i+1}:** `[ {formatted_set} ]`{bonus_str}")
            with col_t2:
                st.metric("Solo Jackpot Score", f"{analysis['solo_jackpot_score']}/100")
            with col_t3:
                st.metric("ML Pattern Score", f"{ml_score}%")
            
            export_rows.append({
                "Set": i+1,
                "Numbers": formatted_set,
                "Solo_Jackpot_Score": analysis['solo_jackpot_score'],
                "ML_Pattern_Score": f"{ml_score}%",
                "Sum": analysis['sum'],
                "Birthday_Nums_Count": analysis['bday_count']
            })
            st.divider()

        # Ticket Slip Exporter
        exp_df = pd.DataFrame(export_rows)
        csv_data = exp_df.to_csv(index=False)
        st.download_button(
            label="📥 Export Ticket Slips to CSV",
            data=csv_data,
            file_name=f"Toto_{game_selection.replace('/', '')}_Tickets.csv",
            mime="text/csv"
        )

# ----------------- TAB 3: Combinatorial Wheeling Studio -----------------
with tab3:
    st.header("🎡 Combinatorial Wheeling Studio")
    st.markdown("""
    **Combinatorial Wheeling** allows you to pick a pool of candidate numbers (e.g. 8 to 14 numbers) and automatically 
    generates an abbreviated wheel that guarantees prize coverage while cutting total ticket costs by **up to 75%**!
    """)
    
    col_w1, col_w2 = st.columns([2, 1])
    with col_w1:
        # Default pool selection using hot numbers
        freq_dict = analytics.get_frequency(df, lookback=100)
        top_hot = [n for n, c in freq_dict.most_common(12)]
        
        pool_selection = st.multiselect(
            "Select Your Pool of Candidate Numbers (8 to 14 numbers recommended):",
            options=list(range(1, game_range + 1)),
            default=sorted(top_hot[:10])
        )
    
    with col_w2:
        wheel_type = st.radio("Wheel Strategy", ["Abbreviated Wheel (4-if-4 Guarantee)", "Full Wheel (100% Coverage)"])

    if len(pool_selection) < 6:
        st.warning("Please select at least 6 numbers for your pool.")
    else:
        if wheel_type == "Abbreviated Wheel (4-if-4 Guarantee)":
            wheeled_tickets = wheeling.generate_abbreviated_wheel(pool_selection, target_match=4, pool_match=4)
        else:
            wheeled_tickets = wheeling.generate_full_wheel(pool_selection)
            
        summary = wheeling.get_wheel_summary(pool_selection, wheeled_tickets)
        
        st.subheader("💰 Wheeling Cost & Optimization Breakdown")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Pool Size", f"{summary['pool_size']} Numbers")
        m_col2.metric("Tickets Generated", f"{summary['total_tickets']} Tickets")
        m_col3.metric("Total Investment", f"RM {summary['total_cost_rm']:.2f}")
        m_col4.metric("Cost Savings vs Full Wheel", f"{summary['savings_percentage']:.1f}%")
        
        st.subheader("🎟️ Generated Ticket Slips")
        wheel_export = []
        for idx, t in enumerate(wheeled_tickets):
            formatted_t = ", ".join(f"{n:02d}" for n in t)
            st.code(f"Ticket #{idx+1:02d}:  [ {formatted_t} ]", language="text")
            wheel_export.append({"Ticket": idx+1, "Numbers": formatted_t})
            
        wheel_df = pd.DataFrame(wheel_export)
        st.download_button(
            label="📥 Download Printable Wheeled Ticket Slips (CSV)",
            data=wheel_df.to_csv(index=False),
            file_name=f"Wheeled_Tickets_{game_selection.replace('/', '')}.csv",
            mime="text/csv"
        )

# ----------------- TAB 4: Master Summary -----------------
with tab4:
    st.header("🏆 Master Summary")
    summary_model = st.selectbox("Select Strategy for Master Summary", ["Anti-Popularity / Solo Jackpot Strategy", "Markov Chain Analysis", "Monte Carlo Simulation", "Mean Reversion (Due)", "Hybrid/Ensemble Model"], index=0)
    st.write(f"Top predictions for all Sports Toto games based on **{summary_model}**")
    
    for g in ["6/50", "6/55", "6/58"]:
        g_df = get_data(g)
        g_range = game_ranges[g]
        g_ml = get_ml_model(g)

        if summary_model == "Anti-Popularity / Solo Jackpot Strategy":
            pred = predictor.anti_popularity_model(g_df, g_range, count=1)[0]
        elif summary_model == "Markov Chain Analysis":
            pred = predictor.markov_chain_analysis(g_df, g_range)
        elif summary_model == "Monte Carlo Simulation":
            pred = predictor.monte_carlo_simulation(g_df, g_range)[0]
        elif summary_model == "Mean Reversion (Due)":
            pred = predictor.mean_reversion_due(g_df, g_range)
        else:
            pred = predictor.hybrid_ensemble(g_df, g_range)

        bonus_str = ""
        if g == "6/50":
            bonus = predictor.predict_bonus_number(g_df, g_range, summary_model)
            bonus_str = f" | Bonus: **{bonus}**"

        analysis = filters.analyze_ticket_entropy(pred, g_range)
        ml_score = ml_model.predict_ticket_ml_score(g_ml, pred, g_range)
        formatted_pred = ", ".join(f"{n:02d}" for n in pred)

        st.subheader(f"🎮 Game {g}")
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            st.markdown(f"**Recommended Set:** `[ {formatted_pred} ]`{bonus_str}")
        with c2:
            st.metric("Solo Jackpot Score", f"{analysis['solo_jackpot_score']}/100")
        with c3:
            st.metric("ML Pattern Score", f"{ml_score}%")
        st.divider()

# ----------------- TAB 5: Backtesting & Probability Lab -----------------
with tab5:
    st.header("🧪 Backtesting & Probability Lab")

    # Time-Travel Backtester
    st.subheader("🕰️ Time-Travel Backtester")
    col_bt1, col_bt2 = st.columns([1, 2])

    with col_bt1:
        available_dates = df_all['DrawDate'].dt.date.unique()
        selected_date = st.selectbox("Pick a Historical Draw Date", available_dates)
        selected_model = st.selectbox("Model to Backtest", ["Anti-Popularity / Solo Jackpot Strategy", "Markov Chain Analysis", "Monte Carlo Simulation", "Mean Reversion (Due)", "Hybrid/Ensemble Model"])

        if st.button("Run Time-Travel Backtest"):
            actual_row = df_all[df_all['DrawDate'].dt.date == selected_date].iloc[0]
            actual_draw = [actual_row[f'DrawnNo{i}'] for i in range(1, 7)]
            actual_bonus = actual_row.get('BonusNo')

            historical_df = df_all[df_all['DrawDate'].dt.date < selected_date]

            if len(historical_df) < 20:
                st.warning("Not enough historical data before this date.")
            else:
                if selected_model == "Anti-Popularity / Solo Jackpot Strategy":
                    pred = predictor.anti_popularity_model(historical_df, game_range, count=1)[0]
                elif selected_model == "Markov Chain Analysis":
                    pred = predictor.markov_chain_analysis(historical_df, game_range)
                elif selected_model == "Monte Carlo Simulation":
                    pred = predictor.monte_carlo_simulation(historical_df, game_range, iterations=1000)[0]
                elif selected_model == "Mean Reversion (Due)":
                    pred = predictor.mean_reversion_due(historical_df, game_range)
                else:
                    pred = predictor.hybrid_ensemble(historical_df, game_range)

                bonus_pred = None
                if game_selection == "6/50":
                    bonus_pred = predictor.predict_bonus_number(historical_df, game_range, selected_model)

                m, hb, prize = probability_lab.evaluate_prediction(pred, actual_draw, bonus_pred, actual_bonus)

                st.info(f"Actual Result: {', '.join(map(str, sorted(actual_draw)))}" + (f" | Bonus: {actual_bonus}" if actual_bonus is not None else ""))
                st.success(f"Prediction: {', '.join(map(str, sorted(pred)))}" + (f" | Bonus: {bonus_pred}" if bonus_pred is not None else ""))

                st.metric("Matches", f"{m} of 6")
                if hb: st.write("✅ Bonus Match!")
                st.write(f"Hypothetical Prize: RM {prize}")

    # Model Leaderboard & P&L
    st.divider()
    st.subheader("🏆 Model Leaderboard (Last 100 Draws)")

    if st.button("Calculate Leaderboard & ROI"):
        leaderboard = []
        for model in ["Markov Chain Analysis", "Monte Carlo Simulation", "Mean Reversion (Due)", "Hybrid/Ensemble Model"]:
            res = probability_lab.backtest_macro(df_all, game_range, model, lookback=100)
            res['Model'] = model
            roi = ((res['total_prize'] - res['total_cost']) / res['total_cost']) * 100
            res['ROI (%)'] = f"{roi:.2f}%"
            leaderboard.append(res)

        lb_df = pd.DataFrame(leaderboard)
        st.table(lb_df[['Model', 'matches_3', 'matches_4', 'matches_5', 'matches_6', 'total_prize', 'ROI (%)']])

    # Expected Value Calculator
    st.divider()
    st.subheader("🎲 Interactive Expected Value (EV) Calculator")
    calc_jackpot = st.slider("Simulated Estimated Jackpot (RM)", min_value=1000000, max_value=60000000, value=int(jackpot_value), step=1000000)
    calc_ev = probability_lab.calculate_ev(game_selection, calc_jackpot)

    col_ev1, col_ev2 = st.columns(2)
    with col_ev1:
        st.metric("Calculated EV", f"RM {calc_ev:.4f}")
        st.write(f"Standard Ticket Cost: RM {probability_lab.TICKET_COST:.2f}")

    with col_ev2:
        if calc_ev > probability_lab.TICKET_COST:
            st.success("🟢 Positive EV! High jackpot makes statistical expectations favorable.")
        else:
            st.error("🔴 Negative EV. Jackpot prize is below the statistical breakeven threshold.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>🎰 Sports Toto Analytics Studio • For Entertainment & Analytical Purposes Only. Play Responsibly.</p>", unsafe_allow_html=True)
