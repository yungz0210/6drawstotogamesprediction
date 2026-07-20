import pandas as pd
import requests
import zipfile
import io
import os
from datetime import datetime

try:
    from curl_cffi import requests as cffi_requests
except ImportError:
    cffi_requests = None

# URLs for Sports Toto results
URLS = {
    "6/50": "https://rst.sportstoto.com.my/upload/Toto650.zip",
    "6/55": "https://rst.sportstoto.com.my/upload/Toto655.zip",
    "6/58": "https://rst.sportstoto.com.my/upload/Toto658.zip"
}

DATA_DIR = "data"

def fetch_zip_bytes(url):
    """Fetches zip file content using curl_cffi to bypass Cloudflare/WAF 403 blocks."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    if cffi_requests is not None:
        try:
            response = cffi_requests.get(url, impersonate='chrome', headers=headers, timeout=15)
            if response.status_code == 200:
                return response.content
        except Exception as e:
            print(f"curl_cffi fetch error for {url}: {e}")
    
    # Fallback to requests with browser headers
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            return response.content
    except Exception as e:
        print(f"requests fetch error for {url}: {e}")
    return None

def download_and_extract():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    for game, url in URLS.items():
        try:
            content = fetch_zip_bytes(url)
            if content:
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    z.extractall(DATA_DIR)
                print(f"Downloaded and extracted {game}")
            else:
                print(f"Failed to download {game}: empty or blocked response")
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

def get_live_jackpots():
    """Scrapes estimated jackpot values live from Sports Toto website."""
    try:
        from bs4 import BeautifulSoup
        import re
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        if cffi_requests is not None:
            r = cffi_requests.get('https://www.sportstoto.com.my', impersonate='chrome', headers=headers, timeout=10)
        else:
            r = requests.get('https://www.sportstoto.com.my', headers=headers, timeout=10)
            
        if r.status_code != 200:
            return None
            
        soup = BeautifulSoup(r.text, 'html.parser')
        results = {}
        for img_tag in soup.find_all('img'):
            src = img_tag.get('src', '')
            if 'estimatedj/' in src:
                game_code = src.split('estimatedj/')[-1].replace('.png', '')
                parent = img_tag.find_parent('div')
                if parent:
                    text = parent.get_text(strip=True)
                    matches = re.findall(r'RM\s*([\d,]+\.?\d*)', text)
                    if matches:
                        num_val = float(matches[0].replace(',', ''))
                        if '658' in game_code:
                            results['6/58'] = {'jackpot': num_val, 'raw': f"RM {matches[0]}"}
                        elif '655' in game_code:
                            results['6/55'] = {'jackpot': num_val, 'raw': f"RM {matches[0]}"}
                        elif '650' in game_code:
                            if '6/50' not in results:
                                results['6/50'] = {'jackpot1': num_val, 'raw_j1': f"RM {matches[0]}"}
                            elif len(matches) > 1 or 'Jackpot 2' in text:
                                results['6/50']['jackpot2'] = num_val
                                results['6/50']['raw_j2'] = f"RM {matches[0]}"
        return results
    except Exception as e:
        print(f"Error scraping live jackpots: {e}")
        return None

