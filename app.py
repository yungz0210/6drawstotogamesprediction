import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import data_manager
import analytics
import predictor
import probability_lab
import requests
from datetime import datetime

st.set_page_config(page_title="Sports Toto Analytics & Prediction", layout="wide")

# Sidebar
st.sidebar.title("Global Controls")
game_selection = st.sidebar.selectbox("Select Game", ["6/50", "6/55", "6/58"])
game_ranges = {"6/50": 50, "6/55": 55, "6/58": 58}
game_range = game_ranges[game_selection]

# Data Loading
@st.cache_data(show_spinner="Loading data...")
def get_data(game):
    return data_manager.load_data(game)

@st.cache_data(show_spinner="Fetching zip data...")
def get_zip_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    return None

if st.sidebar.button("Update Data Daily"):
    data_manager.download_and_extract()
    st.cache_data.clear()
    st.success("Data updated!")

df_all = get_data(game_selection)

# Display Latest Draw Date
latest_date = df_all['DrawDate'].max()
st.sidebar.info(f"Latest Draw Date: {latest_date.strftime('%Y-%m-%d')}")

# Download buttons
st.sidebar.subheader("Download Source Data")
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
st.sidebar.subheader("Filters")
years = sorted(df_all['DrawDate'].dt.year.unique(), reverse=True)
selected_year = st.sidebar.selectbox("Year", ["All"] + list(years))
months = list(range(1, 13))
selected_month = st.sidebar.selectbox("Month", ["All"] + months)
dows = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
selected_dow = st.sidebar.selectbox("Day of Week", ["All"] + dows)

lookback = st.sidebar.number_input("Lookback Period (Draws)", min_value=10, max_value=len(df_all), value=len(df_all), key=f"lookback_{game_selection}")

# Apply filters
df = df_all.copy()
if selected_year != "All":
    df = analytics.filter_by_date(df, year=selected_year)
if selected_month != "All":
    df = analytics.filter_by_date(df, month=selected_month)
if selected_dow != "All":
    df = analytics.filter_by_date(df, dow=dows.index(selected_dow))

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Insights & Stats", "Predictor Studio", "Master Summary", "Backtesting & Probability Lab"])

with tab1:
    st.header(f"Insights & Statistics - {game_selection}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Frequency Analysis (Hot/Cold)")
        freq = analytics.get_frequency(df, lookback)
        freq_df = pd.DataFrame.from_dict(freq, orient='index', columns=['Count']).reset_index()
        freq_df.columns = ['Number', 'Count']
        freq_df = freq_df.sort_values('Count', ascending=False)
        
        fig_freq = px.bar(freq_df, x='Number', y='Count', title="Number Frequency")
        st.plotly_chart(fig_freq, use_container_width=True)
        
        if game_selection == "6/50":
            st.subheader("Bonus Number Frequency")
            b_freq = analytics.get_bonus_frequency(df, lookback)
            b_freq_df = pd.DataFrame.from_dict(b_freq, orient='index', columns=['Count']).reset_index()
            b_freq_df.columns = ['Bonus Number', 'Count']
            b_freq_df = b_freq_df.sort_values('Count', ascending=False)
            fig_b_freq = px.bar(b_freq_df, x='Bonus Number', y='Count', title="Bonus Number Frequency", color_discrete_sequence=['orange'])
            st.plotly_chart(fig_b_freq, use_container_width=True)

    with col2:
        st.subheader("Odd vs Even Ratio")
        oe_ratio = analytics.get_odd_even_ratio(df, lookback)
        fig_oe = px.pie(values=oe_ratio.values, names=oe_ratio.index, title="Odd:Even Distribution")
        st.plotly_chart(fig_oe, use_container_width=True)
        
        st.subheader("Sum Analysis")
        sums = analytics.get_sum_analysis(df, lookback)
        fig_sum = px.histogram(sums, nbins=20, title="Distribution of Sums")
        st.plotly_chart(fig_sum, use_container_width=True)

    st.subheader("Common Pairs & Triplets")
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        pairs = analytics.get_pairs(df, lookback)
        # Format pairs for better display: (1, 2) -> "1, 2"
        formatted_pairs = [ (", ".join(map(str, p)), f) for p, f in pairs ]
        pairs_df = pd.DataFrame(formatted_pairs, columns=['Pair', 'Frequency'])
        st.write("Top 20 Pairs")
        st.table(pairs_df)
    with p_col2:
        triplets = analytics.get_triplets(df, lookback)
        # Format triplets for better display: (1, 2, 3) -> "1, 2, 3"
        formatted_triplets = [ (", ".join(map(str, t)), f) for t, f in triplets ]
        triplets_df = pd.DataFrame(formatted_triplets, columns=['Triplet', 'Frequency'])
        st.write("Top 20 Triplets")
        st.table(triplets_df)

with tab2:
    st.header("Predictor Studio")
    model_choice = st.selectbox("Select Prediction Model", ["Monte Carlo Simulation", "Mean Reversion (Due)", "Markov Chain Analysis", "Hybrid/Ensemble Model"])
    
    if st.button("Generate Prediction"):
        if model_choice == "Monte Carlo Simulation":
            results = predictor.monte_carlo_simulation(df, game_range)
            st.write("Top 5 simulated sets:")
            for i, r in enumerate(results):
                bonus_str = ""
                if game_selection == "6/50":
                    bonus = predictor.predict_bonus_number(df, game_range, model_choice)
                    bonus_str = f" | Bonus: {bonus}"
                st.write(f"Set {i+1}: {', '.join(map(str, r))}{bonus_str}")
        
        elif model_choice == "Mean Reversion (Due)":
            result = predictor.mean_reversion_due(df, game_range)
            bonus_str = ""
            if game_selection == "6/50":
                bonus = predictor.predict_bonus_number(df, game_range, model_choice)
                bonus_str = f" | Bonus: {bonus}"
            st.write(f"Predicted 'Due' Numbers: {', '.join(map(str, result))}{bonus_str}")
            
        elif model_choice == "Markov Chain Analysis":
            result = predictor.markov_chain_analysis(df, game_range)
            bonus_str = ""
            if game_selection == "6/50":
                bonus = predictor.predict_bonus_number(df, game_range, model_choice)
                bonus_str = f" | Bonus: {bonus}"
            st.write(f"Markov Chain Prediction: {', '.join(map(str, result))}{bonus_str}")
            
        elif model_choice == "Hybrid/Ensemble Model":
            result = predictor.hybrid_ensemble(df, game_range)
            bonus_str = ""
            if game_selection == "6/50":
                bonus = predictor.predict_bonus_number(df, game_range, model_choice)
                bonus_str = f" | Bonus: {bonus}"
            st.write(f"Hybrid Ensemble Prediction: {', '.join(map(str, result))}{bonus_str}")

with tab3:
    st.header("Master Summary")
    summary_model = st.selectbox("Select Prediction Model for Summary", ["Markov Chain Analysis", "Monte Carlo Simulation", "Mean Reversion (Due)", "Hybrid/Ensemble Model"], index=0)
    st.write(f"Top predictions for all games based on {summary_model}")
    
    for g in ["6/50", "6/55", "6/58"]:
        g_df = get_data(g)
        g_range = game_ranges[g]

        if summary_model == "Markov Chain Analysis":
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
            bonus_str = f" | Recommended Bonus: {bonus}"

        st.subheader(f"Game {g}")
        st.write(f"Recommended Set: {', '.join(map(str, pred))}{bonus_str}")
        st.divider()

with tab4:
    st.header("Backtesting & Probability Lab")

    # Time-Travel Backtester
    st.subheader("🕰️ Time-Travel Backtester")
    col_bt1, col_bt2 = st.columns([1, 2])

    with col_bt1:
        # User selects a historical draw by date
        available_dates = df_all['DrawDate'].dt.date.unique()
        selected_date = st.selectbox("Pick a Historical Draw Date", available_dates)
        selected_model = st.selectbox("Model to Backtest", ["Markov Chain Analysis", "Monte Carlo Simulation", "Mean Reversion (Due)", "Hybrid/Ensemble Model"])

        if st.button("Run Time-Travel Backtest"):
            # Get actual result
            actual_row = df_all[df_all['DrawDate'].dt.date == selected_date].iloc[0]
            actual_draw = [actual_row[f'DrawnNo{i}'] for i in range(1, 7)]
            actual_bonus = actual_row.get('BonusNo')

            # Use only data available PRIOR to this date
            historical_df = df_all[df_all['DrawDate'].dt.date > selected_date]

            if len(historical_df) < 20:
                st.warning("Not enough historical data before this date to generate a reliable prediction.")
            else:
                # Predict
                if selected_model == "Markov Chain Analysis":
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
            # ROI
            roi = ((res['total_prize'] - res['total_cost']) / res['total_cost']) * 100
            res['ROI (%)'] = f"{roi:.2f}%"
            leaderboard.append(res)

        lb_df = pd.DataFrame(leaderboard)
        st.table(lb_df[['Model', 'matches_3', 'matches_4', 'matches_5', 'matches_6', 'total_prize', 'ROI (%)']])

        st.subheader("💰 Simulated P&L Analysis")
        st.info("Based on purchasing 1 ticket per draw for the last 100 draws (Total Cost: RM 200)")

    # Expected Value Indicator
    st.divider()
    st.subheader("🎲 Expected Value (EV) Indicator")
    jackpot = st.slider("Current Estimated Jackpot (RM)", min_value=1000000, max_value=50000000, value=2000000, step=1000000)
    ev = probability_lab.calculate_ev(game_selection, jackpot)

    col_ev1, col_ev2 = st.columns(2)
    with col_ev1:
        st.metric("Calculated EV", f"RM {ev:.4f}")
        st.write(f"Ticket Cost: RM {probability_lab.TICKET_COST:.2f}")

    with col_ev2:
        if ev > probability_lab.TICKET_COST:
            st.success("Positive EV! The jackpot size makes the bet statistically favorable (excluding taxes and splits).")
        else:
            st.error("Negative EV. Statistically, you expect to lose money on this bet.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>For Entertainment & Analytical Purposes Only. Lottery games are games of chance.</p>", unsafe_allow_html=True)
