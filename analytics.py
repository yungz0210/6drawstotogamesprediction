import pandas as pd
import numpy as np
from itertools import combinations
from collections import Counter

def get_frequency(df, lookback=None):
    if lookback:
        df = df.head(lookback)
    
    main_cols = ['DrawnNo1', 'DrawnNo2', 'DrawnNo3', 'DrawnNo4', 'DrawnNo5', 'DrawnNo6']
    all_nums = df[main_cols].values.flatten()
    freq = Counter(all_nums)
    return freq

def get_bonus_frequency(df, lookback=None):
    if 'BonusNo' not in df.columns:
        return None
    if lookback:
        df = df.head(lookback)
    
    bonus_nums = df['BonusNo'].values.flatten()
    freq = Counter(bonus_nums)
    return freq

def get_pairs(df, lookback=None):
    if lookback:
        df = df.head(lookback)
    
    main_cols = ['DrawnNo1', 'DrawnNo2', 'DrawnNo3', 'DrawnNo4', 'DrawnNo5', 'DrawnNo6']
    pair_counts = Counter()
    
    for _, row in df[main_cols].iterrows():
        nums = tuple(sorted([int(x) for x in row.values]))
        pair_counts.update(combinations(nums, 2))
    
    return pair_counts.most_common(20)

def get_triplets(df, lookback=None):
    if lookback:
        df = df.head(lookback)
    
    main_cols = ['DrawnNo1', 'DrawnNo2', 'DrawnNo3', 'DrawnNo4', 'DrawnNo5', 'DrawnNo6']
    triplet_counts = Counter()
    
    for _, row in df[main_cols].iterrows():
        nums = tuple(sorted([int(x) for x in row.values]))
        triplet_counts.update(combinations(nums, 3))
    
    return triplet_counts.most_common(20)

def get_odd_even_ratio(df, lookback=None):
    if lookback:
        df = df.head(lookback)
    
    main_cols = ['DrawnNo1', 'DrawnNo2', 'DrawnNo3', 'DrawnNo4', 'DrawnNo5', 'DrawnNo6']
    
    def count_odd(row):
        return sum(1 for n in row if n % 2 != 0)
    
    odd_counts = df[main_cols].apply(count_odd, axis=1)
    ratio_counts = odd_counts.value_counts().sort_index()
    # Convert to labels like "3:3", "4:2", etc. (Odd:Even)
    labels = {i: f"{i}:{6-i}" for i in range(7)}
    ratio_counts.index = [labels.get(i, str(i)) for i in ratio_counts.index]
    
    return ratio_counts

def get_sum_analysis(df, lookback=None):
    if lookback:
        df = df.head(lookback)
    
    main_cols = ['DrawnNo1', 'DrawnNo2', 'DrawnNo3', 'DrawnNo4', 'DrawnNo5', 'DrawnNo6']
    sums = df[main_cols].sum(axis=1)
    return sums

def filter_by_date(df, year=None, month=None, dow=None):
    filtered_df = df.copy()
    if year:
        filtered_df = filtered_df[filtered_df['DrawDate'].dt.year == year]
    if month:
        filtered_df = filtered_df[filtered_df['DrawDate'].dt.month == month]
    if dow is not None:
        filtered_df = filtered_df[filtered_df['DrawDate'].dt.dayofweek == dow]
    return filtered_df
