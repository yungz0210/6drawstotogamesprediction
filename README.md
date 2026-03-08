# Sports Toto Analytics & Prediction

This application provides analytics and predictions for Sports Toto games: Star Toto 6/50, Power Toto 6/55, and Supreme Toto 6/58.

## Features

- **Insights & Stats**: Frequency analysis (Hot/Cold numbers), Odd/Even ratio, Sum analysis, and common pairs/triplets.
- **Predictor Studio**: Multiple prediction models to generate potential winning sets.
- **Master Summary**: Recommended sets for all games based on a hybrid model.
- **Data Updates**: One-click update to fetch the latest draw results from Sports Toto.
- **Download Source Data**: Access to original zip files for further analysis.
- **Latest Draw Date**: Clear indication of the data's currency.

## Prediction Models Explained

### 1. Monte Carlo Simulation
The Monte Carlo simulation uses historical frequency as weights to simulate thousands of potential draws (10,000 by default). It accounts for the "Hot" numbers by giving them a higher probability of being picked, while still allowing for "Cold" numbers through Laplace smoothing. The top 5 most frequent sets from these simulations are returned.

### 2. Mean Reversion (Due)
This model is based on the theory that over time, every number should appear with roughly equal frequency. It identifies numbers that haven't appeared for the longest period (the "most due" numbers). It calculates the "drought" for each number and selects the top 6 that have been missing from the results the longest.

### 3. Markov Chain Analysis
This model treats the lottery draws as a sequence of states. It builds a transition matrix that tracks which numbers tend to appear together or in sequence. By analyzing the most recent draw and looking at the historical transitions from those numbers to others in subsequent draws, it predicts the next set of numbers with the highest transition probabilities.

### 4. Hybrid/Ensemble Model
The Hybrid model combines different strategies to create a balanced set:
- **Hot Numbers**: Selects 2 numbers from the top 10 most frequent numbers.
- **Due/Cold Numbers**: Selects 2 numbers from the top 10 least frequent/longest-absent numbers.
- **Common Associations**: Selects 2 numbers based on common pair associations with already selected numbers or the most common pairs overall.
This approach aims to cover different statistical trends simultaneously.

## Installation

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Disclaimer

This application is for **entertainment and analytical purposes only**. Lottery games are games of chance, and no prediction model can guarantee a win. Play responsibly.
