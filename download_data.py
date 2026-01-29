"""Pobiera Auto MPG z UCI Repository"""
import requests
from pathlib import Path

def download():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url)
    with open('data/raw/auto_mpg.data', 'wb') as f:
        f.write(response.content)
    
    print("Dataset pobrany → data/raw/auto_mpg.data")

if __name__ == "__main__":
    download()
