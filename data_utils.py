import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Cache for loaded data
_data_cache = {}

# Emotion columns
EMOTION_COLUMNS = ['anger', 'contempt', 'disgust', 'fear', 'frustration',
                  'gratitude', 'joy', 'love', 'neutral', 'sadness', 'surprise']

EMOTION_LABELS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Frustration',
                 'Gratitude', 'Joy', 'Love', 'Neutral', 'Sadness', 'Surprise']

VALID_DECADES = [1810, 1820, 1830, 1840, 1850, 1860, 1870, 1880, 1890, 1900, 1910, 1920, 1930, 1940, 1950, 1960]

LANGUAGE_FILES = {
    'English': 'final_english_emotions.csv',
    'Hindi': 'final_hindi_emotions.csv',
    'Tamil': 'final_tamil_emotions.csv'
}

COLOR_PALETTE = {
    'English': '#1f77b4',
    'Hindi': "#E500FA",
    'Tamil': '#2ca02c'
}

MARKERS = {
    'English': 'o',
    'Hindi': '^',
    'Tamil': 's'
}

def normalize_decade(val):
    if pd.isna(val):
        return None
    val_str = str(val).strip()
    if len(val_str) >= 4 and val_str[:4].isdigit():
        return int(val_str[:4])
    return None

def load_all_data():
    """Load and cache all emotion datasets"""
    if _data_cache:
        return _data_cache
    
    dfs = {}
    for lang, file in LANGUAGE_FILES.items():
        df = pd.read_csv(file)
        time_col = 'time_period' if 'time_period' in df.columns else 'decade'
        df[time_col] = df[time_col].apply(normalize_decade)
        df = df.dropna(subset=[time_col])
        df[time_col] = df[time_col].astype(int)
        df['Language'] = lang
        # Rename columns to lowercase for consistency
        for col in EMOTION_COLUMNS:
            if col in df.columns:
                df[col.lower()] = df[col]
        dfs[lang] = df
    
    # Load historical data
    hist_df = pd.read_csv('historical_events_english.csv')
    hist_df['decade'] = hist_df['decade'].astype(int)
    dfs['Historical'] = hist_df
    
    _data_cache['all'] = dfs
    return dfs

def get_language_dataframe(language):
    """Get dataframe for specific language"""
    return load_all_data()['all'][language]

def get_emotions_by_decade(language, decade):
    """Get mean emotion values for a specific decade"""
    df = get_language_dataframe(language)
    if 'time_period' in df.columns:
        time_col = 'time_period'
    else:
        time_col = 'decade'
    
    decade_df = df[df[time_col] == decade]
    return decade_df[EMOTION_COLUMNS].mean()

def get_time_series_data(language, emotion):
    """Get smoothed time series data for an emotion"""
    df = get_language_dataframe(language)
    time_col = 'time_period' if 'time_period' in df.columns else 'decade'
    
    # Filter to valid decades
    df = df[df[time_col].isin(VALID_DECADES)]
    
    # Group and calculate mean
    grouped = df.groupby(time_col)[emotion.lower()].mean()
    
    # Apply rolling average
    smoothed = grouped.rolling(window=3, min_periods=1, center=True).mean()
    
    return smoothed

def get_time_series_multilang(languages, emotion):
    """Get time series for one or more languages"""
    data = {}
    for lang in languages:
        data[lang] = get_time_series_data(lang, emotion)
    
    # Combine into DataFrame
    combined = pd.DataFrame(data)
    combined.index.name = 'decade'
    return combined.dropna()

def get_decade_comparison(emotion, decade):
    """Get emotion values across all languages for a specific decade"""
    data = {}
    for lang in ['English', 'Hindi', 'Tamil']:
        data[lang] = get_emotions_by_decade(lang, decade)
    
    return pd.Series(data)

def get_correlation_matrix(language):
    """Get Pearson correlation matrix for a language"""
    df = get_language_dataframe(language)
    emotion_df = df[EMOTION_COLUMNS]
    emotion_df.columns = EMOTION_LABELS
    return emotion_df.corr(method='pearson')

def get_mirror_hypothesis_data(language, emotion):
    """Get data for dual-axis mirror hypothesis"""
    dfs = load_all_data()['all']
    
    # Use specified language (default to English if not found)
    lit_df = dfs.get(language, dfs['English'])
    lit_grouped = lit_df.groupby('time_period')[emotion.lower()].mean()
    lit_smoothed = lit_grouped.rolling(window=3, min_periods=1, center=True).mean()
    
    # Historical data (always English historical events)
    hist_df = dfs['Historical']
    hist_grouped = hist_df.groupby('decade')[emotion.lower()].mean()
    hist_smoothed = hist_grouped.rolling(window=3, min_periods=1, center=True).mean()
    
    # Merge
    merged = pd.DataFrame({
        'Literature': lit_smoothed,
        'Historical': hist_smoothed
    })
    
    return merged.dropna()