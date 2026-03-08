import pandas as pd
import requests
import zipfile
import io
import os
from datetime import datetime

# URLs for Sports Toto results
URLS = {
    "6/50": "https://rst.sportstoto.com.my/upload/Toto650.zip",
    "6/55": "https://rst.sportstoto.com.my/upload/Toto655.zip",
    "6/58": "https://rst.sportstoto.com.my/upload/Toto658.zip"
}

DATA_DIR = "data"

def download_and_extract():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    for game, url in URLS.items():
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    z.extractall(DATA_DIR)
                print(f"Downloaded and extracted {game}")
            else:
                print(f"Failed to download {game}: {response.status_code}")
        except Exception as e:
            print(f"Error downloading {game}: {e}")

def load_data(game_type):
    file_map = {
        "6/50": "Toto650.txt",
        "6/55": "Toto655.txt",
        "6/58": "Toto658.txt"
    }
    
    file_path = os.path.join(DATA_DIR, file_map[game_type])
    if not os.path.exists(file_path):
        download_and_extract()
    
    # Read the text file
    df = pd.read_csv(file_path)
    
    # Basic cleaning
    df.columns = [c.strip() for c in df.columns]
    df['DrawDate'] = pd.to_datetime(df['DrawDate'], format='%Y%m%d')
    
    # Standardize column names for easier access
    # Star Toto 6/50: DrawNo,DrawDate,DrawnNo1,DrawnNo2, DrawnNo3, DrawnNo4, DrawnNo5, DrawnNo6,BonusNo,Jackpot1, Jackpot2
    # Power/Supreme: DrawNo,DrawDate,DrawnNo1,DrawnNo2, DrawnNo3, DrawnNo4, DrawnNo5, DrawnNo6,Jackpot
    
    main_numbers = ['DrawnNo1', 'DrawnNo2', 'DrawnNo3', 'DrawnNo4', 'DrawnNo5', 'DrawnNo6']
    for col in main_numbers:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(int)
    
    if 'BonusNo' in df.columns:
        df['BonusNo'] = pd.to_numeric(df['BonusNo'], errors='coerce').astype(int)
    
    # Sort by date descending
    df = df.sort_values('DrawDate', ascending=False).reset_index(drop=True)
    
    return df

def get_all_numbers(df):
    main_numbers = ['DrawnNo1', 'DrawnNo2', 'DrawnNo3', 'DrawnNo4', 'DrawnNo5', 'DrawnNo6']
    return df[main_numbers]

def get_bonus_numbers(df):
    if 'BonusNo' in df.columns:
        return df[['BonusNo']]
    return None
