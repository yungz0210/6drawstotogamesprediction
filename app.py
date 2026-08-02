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
import toto4d_studio
import ticket_tracker
from datetime import datetime
import importlib

# Force module reload to ensure latest code changes are loaded on Streamlit Cloud
importlib.reload(toto4d_studio)
importlib.reload(ticket_tracker)

st.set_page_config(page_title="Sports Toto Analytics & Prediction Studio", layout="wide", page_icon="🎰")

# Inject Custom CSS for Modern Dark Glassmorphic Aesthetic & Lottery Balls
st.markdown("""
<style>
    /* Dark Theme Accent Polish */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* 3D Visual Lottery Balls */
    .lotto-ball {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 38px;
        height: 38px;
        border-radius: 50%;
        background: radial-gradient(circle at 30% 30%, #3a7bd5, #00d2ff);
        color: #ffffff;
        font-weight: 800;
        font-size: 15px;
        box-shadow: 0 4px 10px rgba(0,210,255,0.35), inset -2px -2px 6px rgba(0,0,0,0.5);
        text-shadow: 0 1px 2px rgba(0,0,0,0.6);
        margin-right: 4px;
    }
    .bonus-ball {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 38px;
        height: 38px;
        border-radius: 50%;
        background: radial-gradient(circle at 30% 30%, #ff9800, #ff5722);
        color: #ffffff;
        font-weight: 800;
        font-size: 15px;
        box-shadow: 0 4px 10px rgba(255,87,34,0.45), inset -2px -2px 6px rgba(0,0,0,0.5);
        text-shadow: 0 1px 2px rgba(0,0,0,0.6);
        margin-left: 4px;
    }
    .badge-4d {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 6px 14px;
        border-radius: 8px;
        background: linear-gradient(135deg, #8e2de2, #4a00e0);
        color: #ffffff;
        font-weight: 800;
        font-size: 18px;
        letter-spacing: 2px;
        box-shadow: 0 4px 12px rgba(142,45,226,0.35);
    }
    
    /* Glassmorphic Container Cards */
    .glass-card {
        background: rgba(22, 27, 34, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)

def render_lotto_balls_html(numbers, bonus=None):
    balls_html = '<div style="display: flex; gap: 6px; align-items: center; flex-wrap: wrap; margin-top: 4px; margin-bottom: 6px;">'
    for n in numbers:
        balls_html += f'<span class="lotto-ball">{n:02d}</span>'
    if bonus is not None:
        balls_html += '<span style="font-weight: bold; color: #ffca28; margin: 0 4px; font-size: 18px;">+</span>'
        balls_html += f'<span class="bonus-ball">{bonus:02d}</span>'
    balls_html += '</div>'
    return balls_html

# Automatic data update on startup
@st.cache_resource
def startup_update():
    data_manager.download_and_extract()
    return datetime.now()

startup_update()

# Sidebar - Modernized Visual Layout
st.sidebar.title("🎰 Global Controls")

game_selection = st.sidebar.selectbox("🎯 Select Active Game", ["6/50", "6/55", "6/58"])
game_ranges = {"6/50": 50, "6/55": 55, "6/58": 58}
game_range = game_ranges[game_selection]

game_icons = {"6/50": "⭐ Star Toto 6/50", "6/55": "⚡ Power Toto 6/55", "6/58": "👑 Supreme Toto 6/58"}
st.sidebar.caption(f"Active Mode: **{game_icons[game_selection]}**")

# Data Loading
@st.cache_data(show_spinner="Loading draw data...")
def get_data(game):
    return data_manager.load_data(game)

@st.cache_data(show_spinner="Loading 4D data...")
def get_4d_data():
    return data_manager.load_4d_data()

@st.cache_data(show_spinner="Fetching zip data...")
def get_zip_data(url):
    return data_manager.fetch_zip_bytes(url)

@st.cache_data(ttl=3600, show_spinner="Scraping live jackpots...")
def load_live_jackpots():
    return data_manager.get_live_jackpots()

if st.sidebar.button("🔄 Update Data Daily", use_container_width=True):
    data_manager.download_and_extract()
    st.cache_data.clear()
    st.sidebar.success("Data successfully updated!")

df_all = get_data(game_selection)
df_4d = get_4d_data()

# Display Latest Draw Date
latest_date = df_all['DrawDate'].max()
st.sidebar.info(f"📅 Latest Draw Date: **{latest_date.strftime('%Y-%m-%d')}**")

# Sidebar Saved Tracker Widget
saved_tickets_list = ticket_tracker.load_tracker_data()
st.sidebar.markdown(f"📜 **Saved Tracker:** `{len(saved_tickets_list)} Tickets Active`")

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

# Filters
st.sidebar.subheader("🔍 Date & Timeframe Filters")

timeframe_presets = ["All Time", "10 Years", "5 Years", "3 Years", "1 Year", "6 Months", "3 Months"]
selected_timeframe = st.sidebar.selectbox("⚡ Quick Timeframe Preset", timeframe_presets, index=0, key=f"timeframe_{game_selection}")

df = analytics.filter_by_timeframe(df_all, selected_timeframe)

# Streamlined Expanders in Sidebar
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

with st.sidebar.expander("📥 Download Source Data"):
    for game, url in data_manager.URLS.items():
        zip_content = get_zip_data(url)
        if zip_content:
            st.download_button(
                label=f"Download {game} Zip",
                data=zip_content,
                file_name=f"Toto{game.replace('/', '')}.zip",
                mime="application/zip",
                use_container_width=True
            )

# Train ML Evaluator once per session/game
@st.cache_resource
def get_ml_model(game):
    g_df = get_data(game)
    return ml_model.train_ml_evaluator(g_df, game_ranges[game])

trained_ml = get_ml_model(game_selection)

# Tabs Navigation
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Insights & Stats", 
    "🔮 Predictor Studio", 
    "🎡 Combinatorial Wheeling", 
    "🎯 4D & Toto 4D Studio",
    "📜 Tracker & History Compare",
    "🧪 Backtesting & Lab",
    "🏆 Master Summary"
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
    st.caption("Generate candidate ticket sets using statistical, ensemble, and anti-popularity strategies.")
    
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

    with st.expander("⚙️ Fine-Tune Advanced Statistical Filters"):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            max_repeat_filter = st.slider("Max Repeats from Last Draw", min_value=0, max_value=3, value=1)
        with f_col2:
            max_consec_filter = st.slider("Max Consecutive Numbers", min_value=0, max_value=3, value=2)
        with f_col3:
            enforce_sum_filter = st.checkbox("Enforce Bell-Curve Sum Range", value=True)

    last_draw_nums = None
    if not df.empty:
        last_row = df.iloc[0]
        main_cols = ['DrawnNo1', 'DrawnNo2', 'DrawnNo3', 'DrawnNo4', 'DrawnNo5', 'DrawnNo6']
        last_draw_nums = [last_row[c] for c in main_cols if c in last_row]

    if st.button("🚀 Generate Candidate Ticket Sets", type="primary", use_container_width=True):
        sets_to_display = []
        
        if model_choice == "Anti-Popularity / Solo Jackpot Strategy":
            sets_to_display = predictor.anti_popularity_model(df, game_range, count=5)
        elif model_choice == "Monte Carlo Simulation":
            raw_sets = predictor.monte_carlo_simulation(df, game_range)
            sets_to_display = filters.filter_tickets(
                raw_sets, game_range=game_range, max_bday=max_bday_filter,
                sum_filter=enforce_sum_filter, max_consecutive=max_consec_filter,
                max_repeat=max_repeat_filter, last_draw=last_draw_nums
            )
            if not sets_to_display: sets_to_display = raw_sets
        elif model_choice == "Mean Reversion (Due)":
            raw = predictor.mean_reversion_due(df, game_range)
            sets_to_display = [raw]
        elif model_choice == "Markov Chain Analysis":
            raw = predictor.markov_chain_analysis(df, game_range)
            sets_to_display = [raw]
        else: # Hybrid
            raw_sets = [predictor.hybrid_ensemble(df, game_range) for _ in range(10)]
            sets_to_display = filters.filter_tickets(
                raw_sets, game_range=game_range, max_bday=max_bday_filter,
                sum_filter=enforce_sum_filter, max_consecutive=max_consec_filter,
                max_repeat=max_repeat_filter, last_draw=last_draw_nums
            )
            if not sets_to_display: sets_to_display = raw_sets[:5]

        st.session_state['generated_lotto_sets'] = sets_to_display

    if 'generated_lotto_sets' in st.session_state:
        st.subheader("🎯 Recommended Candidate Ticket Sets")
        sets_to_display = st.session_state['generated_lotto_sets']
        export_rows = []
        
        for i, ticket in enumerate(sets_to_display[:5]):
            analysis = filters.analyze_ticket_entropy(ticket, game_range, last_draw=last_draw_nums)
            ml_score = ml_model.predict_ticket_ml_score(trained_ml, ticket, game_range)
            
            b_num = None
            if game_selection == "6/50":
                b_num = predictor.predict_bonus_number(df, game_range, model_choice)

            col_t1, col_t2, col_t3, col_t4 = st.columns([3, 1, 1, 1])
            with col_t1:
                st.markdown(f"**Ticket Set #{i+1}**")
                st.markdown(render_lotto_balls_html(ticket, b_num), unsafe_allow_html=True)
            with col_t2:
                st.metric("Solo Jackpot", f"{analysis['solo_jackpot_score']}/100")
            with col_t3:
                st.metric("ML Score", f"{ml_score}%")
            with col_t4:
                save_key = f"save_lotto_{game_selection}_{i}_{ticket[0]}"
                if st.button("💾 Save to Tracker", key=save_key):
                    ticket_tracker.add_ticket(
                        game_type=game_selection,
                        numbers=ticket,
                        bonus=b_num,
                        strategy=f"Predictor Studio ({model_choice})",
                        notes=f"Solo Score: {analysis['solo_jackpot_score']}, ML: {ml_score}%",
                        play_type="Lotto"
                    )
                    st.success("Saved!")
                    st.rerun()

            formatted_set = ", ".join(f"{n:02d}" for n in ticket)
            export_rows.append({
                "Set": i+1, "Numbers": formatted_set,
                "Solo_Jackpot_Score": analysis['solo_jackpot_score'],
                "ML_Pattern_Score": f"{ml_score}%", "Sum": analysis['sum']
            })
            st.divider()

        if export_rows:
            exp_df = pd.DataFrame(export_rows)
            st.download_button(
                label="📥 Export Ticket Slips to CSV",
                data=exp_df.to_csv(index=False),
                file_name=f"Toto_{game_selection.replace('/', '')}_Tickets.csv",
                mime="text/csv"
            )

# ----------------- TAB 3: Combinatorial Wheeling Studio -----------------
with tab3:
    st.header("🎡 Combinatorial Wheeling Studio")
    st.markdown("""
    **Combinatorial Wheeling** lets you select a candidate pool of numbers (e.g. 8 to 16 numbers) and automatically 
    generates an abbreviated wheel or banker wheel that guarantees prize coverage while cutting ticket costs by **up to 80%**!
    """)
    
    col_w1, col_w2 = st.columns([2, 1])
    with col_w1:
        freq_dict = analytics.get_frequency(df, lookback=100)
        top_hot = [n for n, c in freq_dict.most_common(14)]
        
        pool_selection = st.multiselect(
            "Select Candidate Number Pool (8 to 16 numbers recommended):",
            options=list(range(1, game_range + 1)),
            default=sorted(top_hot[:10])
        )
        
        banker_selection = st.multiselect(
            "📌 Select Banker / Key Numbers (Must appear in EVERY ticket):",
            options=pool_selection,
            default=[]
        )
    
    with col_w2:
        wheel_type = st.radio(
            "Wheel Guarantee Strategy", 
            [
                "Abbreviated Wheel (4-if-4 Guarantee)", 
                "Abbreviated Wheel (3-if-3 Guarantee)",
                "Abbreviated Wheel (5-if-5 Guarantee)",
                "Full Wheel (100% Coverage)"
            ]
        )

    if len(pool_selection) < 6:
        st.warning("Please select at least 6 numbers for your pool.")
    else:
        if banker_selection:
            if wheel_type == "Abbreviated Wheel (3-if-3 Guarantee)":
                wheeled_tickets = wheeling.generate_key_number_wheel(pool_selection, banker_selection, target_match=3, pool_match=3)
            elif wheel_type == "Abbreviated Wheel (5-if-5 Guarantee)":
                wheeled_tickets = wheeling.generate_key_number_wheel(pool_selection, banker_selection, target_match=5, pool_match=5)
            elif wheel_type == "Full Wheel (100% Coverage)":
                wheeled_tickets = wheeling.generate_key_number_wheel(pool_selection, banker_selection, target_match=6, pool_match=6)
            else:
                wheeled_tickets = wheeling.generate_key_number_wheel(pool_selection, banker_selection, target_match=4, pool_match=4)
        else:
            if wheel_type == "Abbreviated Wheel (3-if-3 Guarantee)":
                wheeled_tickets = wheeling.generate_abbreviated_wheel(pool_selection, target_match=3, pool_match=3)
            elif wheel_type == "Abbreviated Wheel (5-if-5 Guarantee)":
                wheeled_tickets = wheeling.generate_abbreviated_wheel(pool_selection, target_match=5, pool_match=5)
            elif wheel_type == "Full Wheel (100% Coverage)":
                wheeled_tickets = wheeling.generate_full_wheel(pool_selection)
            else:
                wheeled_tickets = wheeling.generate_abbreviated_wheel(pool_selection, target_match=4, pool_match=4)
            
        summary = wheeling.get_wheel_summary(pool_selection, wheeled_tickets, key_numbers=banker_selection)
        
        st.subheader("💰 Wheeling Cost & Optimization Breakdown")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Pool Size", f"{summary['pool_size']} Numbers")
        m_col2.metric("Tickets Generated", f"{summary['total_tickets']} Tickets")
        m_col3.metric("Total Investment", f"RM {summary['total_cost_rm']:.2f}")
        m_col4.metric("Savings vs Full", f"{summary['savings_percentage']:.1f}%")
        
        if banker_selection:
            st.info(f"📌 **Banker Numbers Locked:** `{', '.join(map(str, banker_selection))}` (Included in all {summary['total_tickets']} tickets)")

        st.subheader("🎟️ Generated Ticket Slips")
        wheel_export = []
        for idx, t in enumerate(wheeled_tickets):
            col_wh1, col_wh2 = st.columns([4, 1])
            with col_wh1:
                st.markdown(f"**Ticket #{idx+1:02d}**")
                st.markdown(render_lotto_balls_html(t), unsafe_allow_html=True)
            with col_wh2:
                if st.button("💾 Save Ticket", key=f"save_wheel_{idx}_{t[0]}"):
                    ticket_tracker.add_ticket(
                        game_type=game_selection,
                        numbers=t,
                        strategy=f"Wheeling Studio ({wheel_type})",
                        notes=f"Pool Size: {len(pool_selection)}",
                        play_type="Lotto"
                    )
                    st.success("Saved!")
                    st.rerun()

            formatted_t = ", ".join(f"{n:02d}" for n in t)
            wheel_export.append({"Ticket": idx+1, "Numbers": formatted_t})
            
        wheel_df = pd.DataFrame(wheel_export)
        st.download_button(
            label="📥 Download Printable Wheeled Ticket Slips (CSV)",
            data=wheel_df.to_csv(index=False),
            file_name=f"Wheeled_Tickets_{game_selection.replace('/', '')}.csv",
            mime="text/csv"
        )

# ----------------- TAB 4: 4D & Sports Toto 4D Studio -----------------
with tab4:
    st.header("🎯 4D & Sports Toto 4D Studio")
    st.markdown("""
    Analyze **4D position digit distributions**, digit patterns (Single, Double, Triple, Quad), generate **Box Play / i-Perm** permutations,
    wheel **System 4D Jackpot pairs**, and run **Dedicated 4D Predictors**!
    """)
    
    sec_a, sec_b, sec_c = st.tabs(["📊 4D Digit Analytics", "🔮 4D Predictors & Generators", "🎲 Box Play & Jackpot Pair Wheels"])
    
    with sec_a:
        st.subheader("📈 Position-Wise Digit Frequency (D1, D2, D3, D4)")
        freq_matrix = toto4d_studio.analyze_4d_digit_frequencies(df_4d)
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            df_d1_d2 = pd.DataFrame({
                'Digit': [str(i) for i in range(10)],
                'D1 (1st Digit)': [freq_matrix['D1'][str(i)] for i in range(10)],
                'D2 (2nd Digit)': [freq_matrix['D2'][str(i)] for i in range(10)]
            })
            fig_pos1 = px.bar(df_d1_d2, x='Digit', y=['D1 (1st Digit)', 'D2 (2nd Digit)'], barmode='group', title="D1 & D2 Digit Distribution")
            st.plotly_chart(fig_pos1, use_container_width=True)
            
        with col_d2:
            df_d3_d4 = pd.DataFrame({
                'Digit': [str(i) for i in range(10)],
                'D3 (3rd Digit)': [freq_matrix['D3'][str(i)] for i in range(10)],
                'D4 (4th Digit)': [freq_matrix['D4'][str(i)] for i in range(10)]
            })
            fig_pos2 = px.bar(df_d3_d4, x='Digit', y=['D3 (3rd Digit)', 'D4 (4th Digit)'], barmode='group', title="D3 & D4 Digit Distribution")
            st.plotly_chart(fig_pos2, use_container_width=True)

        st.subheader("🧩 Historical 4D Digit Structure Pattern Distribution")
        if hasattr(toto4d_studio, 'analyze_4d_patterns'):
            pattern_counts = toto4d_studio.analyze_4d_patterns(df_4d)
        else:
            pattern_counts = {"Single (24-Way)": 60, "Double (12-Way)": 30, "Double-Double (6-Way)": 5, "Triple (4-Way)": 4, "Quad (1-Way)": 1}
            
        fig_pat = px.pie(values=list(pattern_counts.values()), names=list(pattern_counts.keys()), title="4D Digit Pattern Breakdown", hole=0.4)
        st.plotly_chart(fig_pat, use_container_width=True)

    with sec_b:
        st.subheader("🔮 Dedicated 4D Prediction Algorithms")
        c_pred1, c_pred2 = st.columns(2)
        
        with c_pred1:
            st.markdown("#### 🎲 Positional Monte Carlo 4D Generator")
            if st.button("Generate Monte Carlo 4D Sets"):
                if hasattr(toto4d_studio, 'monte_carlo_4d'):
                    mc_4d = toto4d_studio.monte_carlo_4d(df_4d, count=5)
                else:
                    mc_4d = toto4d_studio.generate_anti_popularity_4d(count=5)
                st.session_state['mc_4d'] = mc_4d
                
            if 'mc_4d' in st.session_state:
                for idx, num in enumerate(st.session_state['mc_4d']):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**Candidate #{idx+1}:** <span class='badge-4d'>{num}</span>", unsafe_allow_html=True)
                    with c2:
                        if st.button("💾 Save", key=f"save_mc4d_{idx}_{num}"):
                            ticket_tracker.add_ticket("4D", numbers=[num], strategy="4D Monte Carlo", play_type="4D")
                            st.success("Saved!")
                            st.rerun()

        with c_pred2:
            st.markdown("#### ⚡ Hot & Due Positional Digit 4D Generator")
            if st.button("Generate Hot & Due 4D Sets"):
                if hasattr(toto4d_studio, 'hot_due_4d'):
                    hd_4d = toto4d_studio.hot_due_4d(df_4d, count=5)
                else:
                    hd_4d = toto4d_studio.generate_anti_popularity_4d(count=5)
                st.session_state['hd_4d'] = hd_4d
                
            if 'hd_4d' in st.session_state:
                for idx, num in enumerate(st.session_state['hd_4d']):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**Candidate #{idx+1}:** <span class='badge-4d'>{num}</span>", unsafe_allow_html=True)
                    with c2:
                        if st.button("💾 Save", key=f"save_hd4d_{idx}_{num}"):
                            ticket_tracker.add_ticket("4D", numbers=[num], strategy="4D Hot & Due", play_type="4D")
                            st.success("Saved!")
                            st.rerun()

        st.divider()
        st.subheader("🛡️ Anti-Popularity / Unshared 4D Candidate Generator")
        if st.button("🎲 Generate Unshared 4D Candidates"):
            anti_4d = toto4d_studio.generate_anti_popularity_4d(count=10)
            st.session_state['anti_4d'] = anti_4d

        if 'anti_4d' in st.session_state:
            cols_ap = st.columns(5)
            for idx, num in enumerate(st.session_state['anti_4d']):
                with cols_ap[idx % 5]:
                    st.markdown(f"<span class='badge-4d'>{num}</span>", unsafe_allow_html=True)
                    if st.button("💾 Save", key=f"save_anti4d_{idx}_{num}"):
                        ticket_tracker.add_ticket("4D", numbers=[num], strategy="Anti-Popularity 4D", play_type="4D")
                        st.success("Saved!")
                        st.rerun()

    with sec_c:
        sub_c1, sub_c2 = st.columns(2)
        
        with sub_c1:
            st.subheader("📦 Box Play / i-Perm Permutation Wheel")
            input_4d = st.text_input("Enter a 4-Digit Number (e.g. 1234 or 8812):", value="1234")
            
            if st.button("Generate 4D Permutations"):
                perms, label, cost = toto4d_studio.generate_4d_permutations(input_4d)
                st.success(f"**Permutation Type:** `{label}` | **Total Permutations:** `{len(perms)}`")
                st.metric("Total Investment (Standard RM 1/perm)", f"RM {cost:.2f}")
                
                st.write("Generated Permutation Slips:")
                st.code(", ".join(perms), language="text")
                
                if st.button("💾 Save All Permutations to Tracker"):
                    for p in perms:
                        ticket_tracker.add_ticket("4D", numbers=[p], strategy=f"Box Play ({label})", play_type="4D")
                    st.success("All permutations saved!")
                    st.rerun()
                
        with sub_c2:
            st.subheader("💰 System 4D Jackpot Pair Generator")
            st.caption("Select a pool of 4D numbers to generate all 2-pair combinations for Toto 4D Jackpot 1/2.")
            
            pool_input = st.text_area(
                "Enter Pool of 4D Numbers (separated by commas or newlines):",
                value="1234, 5678, 8888, 0168, 9999"
            )
            
            if st.button("🚀 Generate 4D Jackpot Pairs", type="primary"):
                nums = [n.strip() for n in pool_input.replace('\n', ',').split(',') if n.strip()]
                pairs, num_pairs, total_cost = toto4d_studio.generate_system_4d_jackpot(nums)
                st.session_state['jp_pairs'] = pairs
                st.session_state['jp_cost'] = total_cost
                
            if 'jp_pairs' in st.session_state:
                pairs = st.session_state['jp_pairs']
                total_cost = st.session_state['jp_cost']
                st.success(f"**Jackpot Pairs Generated:** {len(pairs)}")
                st.metric("Total Ticket Investment (RM 2/pair)", f"RM {total_cost:.2f}")
                
                for idx, (p1, p2) in enumerate(pairs):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"**Pair #{idx+1:02d}:** <span class='badge-4d'>{p1}</span> + <span class='badge-4d'>{p2}</span>", unsafe_allow_html=True)
                    with c2:
                        if st.button("💾 Save Pair", key=f"save_jppair_{idx}_{p1}_{p2}"):
                            ticket_tracker.add_ticket("4D Jackpot", numbers=[p1, p2], strategy="System 4D Jackpot", play_type="4D Jackpot")
                            st.success("Saved!")
                            st.rerun()

# ----------------- TAB 5: Tracker & History Compare -----------------
with tab5:
    st.header("📜 Ticket History, Prediction Tracker & Draw Compare")
    st.caption("View your saved predictions, type custom ticket entries, automatically compare against real draw results, and track P&L!")
    
    col_tr1, col_tr2 = st.columns([1, 2])
    
    with col_tr1:
        st.subheader("➕ Save / Type New Selection")
        input_play_type = st.radio("Play Type", ["Lotto", "4D", "4D Jackpot"], horizontal=True)
        
        target_game = "6/50"
        if input_play_type == "Lotto":
            target_game = st.selectbox("Game Type", ["6/50", "6/55", "6/58"])
        elif input_play_type == "4D" or input_play_type == "4D Jackpot":
            target_game = "4D"
            
        manual_notes = st.text_input("Strategy / Tag / Notes", value="My Selection")
        
        if input_play_type == "Lotto":
            raw_input = st.text_area("Enter 6 Lotto Numbers (e.g. 05, 12, 19, 28, 35, 42):", value="05, 12, 19, 28, 35, 42")
            bonus_input = st.number_input("Bonus Number (Optional for 6/50)", min_value=1, max_value=50, value=7) if target_game == "6/50" else None
        elif input_play_type == "4D":
            raw_input = st.text_area("Enter 4-Digit Number (e.g. 1234):", value="1234")
            bonus_input = None
        else: # 4D Jackpot
            raw_input = st.text_area("Enter 2x 4D Numbers for Jackpot Pair (e.g. 1234, 5678):", value="1234, 5678")
            bonus_input = None

        if st.button("➕ Add Ticket to Tracker", type="primary", use_container_width=True):
            parsed_nums, err = ticket_tracker.parse_manual_input(raw_input, play_type=input_play_type)
            if err:
                st.error(err)
            else:
                ticket_tracker.add_ticket(
                    game_type=target_game,
                    numbers=parsed_nums,
                    bonus=bonus_input,
                    strategy=manual_notes,
                    play_type=input_play_type
                )
                st.success("Ticket saved to persistent memory!")
                st.rerun()

    with col_tr2:
        st.subheader("📊 Tracker Summary & Performance")
        raw_tickets = ticket_tracker.load_tracker_data()
        
        # Load dictionary of dataframes for evaluation
        lotto_dfs = {
            "6/50": get_data("6/50"),
            "6/55": get_data("6/55"),
            "6/58": get_data("6/58")
        }
        
        evaluated_tickets = [
            ticket_tracker.evaluate_single_ticket(t, df_lotto=lotto_dfs.get(t.get("game_type", "6/50")), df_4d=df_4d)
            for t in raw_tickets
        ]
        
        summary = ticket_tracker.get_tracker_summary(evaluated_tickets)
        
        m_s1, m_s2, m_s3, m_s4 = st.columns(4)
        m_s1.metric("Total Saved Slips", f"{summary['total_tickets']}")
        m_s2.metric("Total Investment", f"RM {summary['total_cost']:.2f}")
        m_s3.metric("Total Prizes Claimed", f"RM {summary['total_prizes']:.2f}")
        
        pl_color = "normal" if summary['net_pl'] >= 0 else "inverse"
        m_s4.metric("Net P&L (ROI)", f"RM {summary['net_pl']:.2f}", delta=f"{summary['roi_pct']:.1f}%", delta_color=pl_color)

    st.divider()
    st.subheader("📋 Saved Tickets Log & Historical Match Verification")
    
    if not evaluated_tickets:
        st.info("No saved tickets found in tracker. Generate candidate sets in Predictor Studio or type custom numbers above!")
    else:
        filter_game = st.selectbox("Filter Log by Game", ["All Games", "6/50", "6/55", "6/58", "4D", "4D Jackpot"])
        filtered_tickets = evaluated_tickets
        if filter_game != "All Games":
            filtered_tickets = [t for t in evaluated_tickets if t.get("game_type") == filter_game or t.get("play_type") == filter_game]
            
        for t in filtered_tickets:
            with st.container():
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                c_lk1, c_lk2, c_lk3, c_lk4 = st.columns([3, 2, 2, 1])
                
                with c_lk1:
                    st.markdown(f"**ID:** `{t['id']}` | **Game:** `{t['game_type']}` | **Strategy:** *{t['strategy']}*")
                    if t['play_type'] == "Lotto":
                        st.markdown(render_lotto_balls_html(t['numbers'], t.get('bonus')), unsafe_allow_html=True)
                    else:
                        nums_html = " + ".join([f"<span class='badge-4d'>{n}</span>" for n in t['numbers']])
                        st.markdown(nums_html, unsafe_allow_html=True)
                        
                with c_lk2:
                    st.markdown(f"📅 **Added:** `{t['created_at'][:10]}`")
                    if "evaluated_draw_date" in t:
                        st.markdown(f"🎯 **Tested Draw Date:** `{t['evaluated_draw_date']}`")
                        
                with c_lk3:
                    st.markdown(f"<span style='background: {t['badge_color']}; color: black; font-weight: bold; padding: 4px 8px; border-radius: 6px;'>{t['badge_label']}</span>", unsafe_allow_html=True)
                    if t.get('prize', 0) > 0:
                        st.markdown(f"💰 **Prize Won:** `RM {t['prize']:.2f}`")
                        
                with c_lk4:
                    if st.button("🗑️ Delete", key=f"del_{t['id']}"):
                        ticket_tracker.delete_ticket(t['id'])
                        st.rerun()
                        
                st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            log_df = pd.DataFrame(evaluated_tickets)
            st.download_button(
                label="📥 Export Tracker History to CSV",
                data=log_df.to_csv(index=False),
                file_name="Sports_Toto_Saved_Tracker_History.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_act2:
            if st.button("⚠️ Clear All Tracker History", type="secondary", use_container_width=True):
                ticket_tracker.clear_all_tickets()
                st.success("Tracker cleared.")
                st.rerun()

# ----------------- TAB 6: Backtesting & Probability Lab -----------------
with tab6:
    st.header("🧪 Backtesting & Probability Lab")

    st.subheader("🕰️ Time-Travel Backtester")
    col_bt1, col_bt2 = st.columns([1, 2])

    with col_bt1:
        available_dates = df_all['DrawDate'].dt.date.unique()
        selected_date = st.selectbox("Pick Historical Draw Date", available_dates)
        selected_model = st.selectbox("Model to Backtest", ["Anti-Popularity / Solo Jackpot Strategy", "Markov Chain Analysis", "Monte Carlo Simulation", "Mean Reversion (Due)", "Hybrid/Ensemble Model"])

        if st.button("Run Time-Travel Backtest"):
            actual_row = df_all[df_all['DrawDate'].dt.date == selected_date].iloc[0]
            actual_draw = [actual_row[f'DrawnNo{i}'] for i in range(1, 7)]
            actual_bonus = actual_row.get('BonusNo')

            historical_df = df_all[df_all['DrawDate'].dt.date < selected_date]

            if len(historical_df) < 20:
                st.warning("Not enough historical data prior to selected date.")
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

# ----------------- TAB 7: Master Summary -----------------
with tab7:
    st.header("🏆 Master Multi-Game Dashboard")
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
        b_num = None
        if g == "6/50":
            b_num = predictor.predict_bonus_number(g_df, g_range, summary_model)
            bonus_str = f" | Bonus: **{b_num}**"

        analysis = filters.analyze_ticket_entropy(pred, g_range)
        ml_score = ml_model.predict_ticket_ml_score(g_ml, pred, g_range)

        st.subheader(f"🎮 Game {g}")
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        with c1:
            st.markdown(render_lotto_balls_html(pred, b_num), unsafe_allow_html=True)
        with c2:
            st.metric("Solo Jackpot Score", f"{analysis['solo_jackpot_score']}/100")
        with c3:
            st.metric("ML Pattern Score", f"{ml_score}%")
        with c4:
            if st.button(f"💾 Save {g} Ticket", key=f"save_master_{g}"):
                ticket_tracker.add_ticket(
                    game_type=g,
                    numbers=pred,
                    bonus=b_num,
                    strategy=f"Master Summary ({summary_model})",
                    play_type="Lotto"
                )
                st.success("Saved!")
                st.rerun()
        st.divider()

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>🎰 Sports Toto Analytics Studio • For Entertainment & Analytical Purposes Only. Play Responsibly.</p>", unsafe_allow_html=True)
