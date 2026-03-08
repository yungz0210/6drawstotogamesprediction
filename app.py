import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import data_manager
import analytics
import predictor
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

if st.sidebar.button("Update Data Daily"):
    data_manager.download_and_extract()
    st.cache_data.clear()
    st.success("Data updated!")

df_all = get_data(game_selection)

# Filters
st.sidebar.subheader("Filters")
years = sorted(df_all['DrawDate'].dt.year.unique(), reverse=True)
selected_year = st.sidebar.selectbox("Year", ["All"] + list(years))
months = list(range(1, 13))
selected_month = st.sidebar.selectbox("Month", ["All"] + months)
dows = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
selected_dow = st.sidebar.selectbox("Day of Week", ["All"] + dows)

lookback = st.sidebar.number_input("Lookback Period (Draws)", min_value=10, max_value=len(df_all), value=100)

# Apply filters
df = df_all.copy()
if selected_year != "All":
    df = analytics.filter_by_date(df, year=selected_year)
if selected_month != "All":
    df = analytics.filter_by_date(df, month=selected_month)
if selected_dow != "All":
    df = analytics.filter_by_date(df, dow=dows.index(selected_dow))

# Tabs
tab1, tab2, tab3 = st.tabs(["Insights & Stats", "Predictor Studio", "Master Summary"])

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
                st.write(f"Set {i+1}: {', '.join(map(str, r))}")
        
        elif model_choice == "Mean Reversion (Due)":
            result = predictor.mean_reversion_due(df, game_range)
            st.write(f"Predicted 'Due' Numbers: {', '.join(map(str, result))}")
            
        elif model_choice == "Markov Chain Analysis":
            result = predictor.markov_chain_analysis(df, game_range)
            st.write(f"Markov Chain Prediction: {', '.join(map(str, result))}")
            
        elif model_choice == "Hybrid/Ensemble Model":
            result = predictor.hybrid_ensemble(df, game_range)
            st.write(f"Hybrid Ensemble Prediction: {', '.join(map(str, result))}")

with tab3:
    st.header("Master Summary")
    st.write("Top predictions for all games based on Hybrid Model")
    
    for g in ["6/50", "6/55", "6/58"]:
        g_df = get_data(g)
        g_range = game_ranges[g]
        pred = predictor.hybrid_ensemble(g_df, g_range)
        st.subheader(f"Game {g}")
        st.write(f"Recommended Set: {', '.join(map(str, pred))}")
        st.divider()

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>For Entertainment & Analytical Purposes Only. Lottery games are games of chance.</p>", unsafe_allow_html=True)
