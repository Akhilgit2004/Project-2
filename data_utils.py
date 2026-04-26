import os
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

_data_cache = {}

EMOTION_COLUMNS = ['anger', 'contempt', 'disgust', 'fear', 'frustration',
                  'gratitude', 'joy', 'love', 'neutral', 'sadness', 'surprise']

EMOTION_LABELS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Frustration',
                 'Gratitude', 'Joy', 'Love', 'Neutral', 'Sadness', 'Surprise']

VALID_DECADES = [1810, 1820, 1830, 1840, 1850, 1860, 1870, 1880, 1890, 1900, 1910, 1920, 1930, 1940, 1950, 1960]

LITERARY_PERIODS = {
    'Romantic': (1810, 1840),
    'Victorian': (1840, 1900),
    'Edwardian': (1900, 1910),
    'Modern': (1910, 1960)
}

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
        for col in EMOTION_COLUMNS:
            if col in df.columns:
                df[col.lower()] = df[col]
        dfs[lang] = df
    
    hist_df = pd.read_csv('historical_events_english.csv')
    hist_df['decade'] = hist_df['decade'].astype(int)
    dfs['Historical'] = hist_df
    
    _data_cache['all'] = dfs
    return dfs

def get_language_dataframe(language):
    """Get dataframe for specific language"""
    data = load_all_data()
    if 'all' in data:
        return data['all'][language]
    return data[language]

def get_emotions_by_decade(language, decade):
    """Get mean emotion values for a specific decade"""
    df = get_language_dataframe(language)
    time_col = 'time_period' if 'time_period' in df.columns else 'decade'
    decade_df = df[df[time_col] == decade]
    return decade_df[EMOTION_COLUMNS].mean()

def get_time_series_data(language, emotion, decade_start=1810, decade_end=1960):
    """Get smoothed time series data for an emotion"""
    df = get_language_dataframe(language)
    time_col = 'time_period' if 'time_period' in df.columns else 'decade'
    
    df = df[(df[time_col] >= decade_start) & (df[time_col] <= decade_end)]
    df = df[df[time_col].isin(VALID_DECADES)]
    
    grouped = df.groupby(time_col)[emotion.lower()].mean()
    smoothed = grouped.rolling(window=3, min_periods=1, center=True).mean()
    
    return smoothed

def get_time_series_multilang(languages, emotion, decade_start=1810, decade_end=1960):
    """Get time series for one or more languages"""
    data = {}
    for lang in languages:
        data[lang] = get_time_series_data(lang, emotion.lower(), decade_start, decade_end)
    
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

def get_mirror_hypothesis_data(language, emotion, decade_start=1810, decade_end=1960):
    """Get data for dual-axis mirror hypothesis"""
    dfs = load_all_data()['all']
    
    lit_df = dfs.get(language, dfs['English'])
    time_col = 'time_period' if 'time_period' in lit_df.columns else 'decade'
    lit_df = lit_df[(lit_df[time_col] >= decade_start) & (lit_df[time_col] <= decade_end)]
    lit_grouped = lit_df.groupby(time_col)[emotion.lower()].mean()
    lit_smoothed = lit_grouped.rolling(window=3, min_periods=1, center=True).mean()
    
    hist_df = dfs['Historical']
    hist_df = hist_df[(hist_df['decade'] >= decade_start) & (hist_df['decade'] <= decade_end)]
    hist_grouped = hist_df.groupby('decade')[emotion.lower()].mean()
    hist_smoothed = hist_grouped.rolling(window=3, min_periods=1, center=True).mean()
    
    merged = pd.DataFrame({
        'Literature': lit_smoothed,
        'Historical': hist_smoothed
    })
    
    return merged.dropna()

def get_heatmap_data(language, decade_start=1810, decade_end=1960):
    """Get data for heatmap visualization"""
    if language == 'All':
        languages = ['English', 'Hindi', 'Tamil']
    else:
        languages = [language]
    
    result = {}
    for lang in languages:
        df = get_language_dataframe(lang)
        time_col = 'time_period' if 'time_period' in df.columns else 'decade'
        df = df[(df[time_col] >= decade_start) & (df[time_col] <= decade_end)]
        grouped = df.groupby(time_col)[EMOTION_COLUMNS].mean()
        result[lang] = grouped
    
    return result

def get_distribution_data(language, decade_start=1810, decade_end=1960):
    """Get data for distribution visualization"""
    if language == 'All':
        languages = ['English', 'Hindi', 'Tamil']
    else:
        languages = [language]
    
    result = {}
    for lang in languages:
        df = get_language_dataframe(lang)
        time_col = 'time_period' if 'time_period' in df.columns else 'decade'
        df = df[(df[time_col] >= decade_start) & (df[time_col] <= decade_end)]
        result[lang] = df[EMOTION_COLUMNS]
    
    return result

def calculate_significance(emotion, language1, language2, decade_start=1810, decade_end=1960):
    """Calculate statistical significance between two languages"""
    df1 = get_language_dataframe(language1)
    df2 = get_language_dataframe(language2)
    
    time_col = 'time_period' if 'time_period' in df1.columns else 'decade'
    
    df1 = df1[(df1[time_col] >= decade_start) & (df1[time_col] <= decade_end)]
    df2 = df2[(df2[time_col] >= decade_start) & (df2[time_col] <= decade_end)]
    
    data1 = df1[emotion.lower()].values
    data2 = df2[emotion.lower()].values
    
    if len(data1) > 1 and len(data2) > 1:
        t_stat, p_value = stats.ttest_ind(data1, data2)
        mean1 = np.mean(data1)
        mean2 = np.mean(data2)
        
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        
        return {
            'language1': language1,
            'language2': language2,
            'emotion': emotion,
            'mean1': float(mean1),
            'mean2': float(mean2),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significance': significance,
            'interpretation': f"The difference in {emotion} between {language1} and {language2} is {'statistically significant' if p_value < 0.05 else 'not statistically significant'} (p={p_value:.4f})"
        }
    
    return {'error': 'Insufficient data for statistical analysis'}

def detect_peaks(emotion, language, decade_start=1810, decade_end=1960, threshold=0.1):
    """Detect significant peaks in emotional trajectory"""
    df = get_time_series_data(language, emotion.lower(), decade_start, decade_end)
    
    if len(df) < 3:
        return {'peaks': [], 'troughs': []}
    
    values = df.values
    peaks = []
    troughs = []
    
    mean_val = np.mean(values)
    std_val = np.std(values)
    threshold_val = mean_val + std_val * threshold
    trough_threshold_val = mean_val - std_val * threshold
    
    for i in range(1, len(values) - 1):
        if values[i] > values[i-1] and values[i] > values[i+1] and values[i] > threshold_val:
            peaks.append({'decade': int(df.index[i]), 'value': float(values[i])})
        elif values[i] < values[i-1] and values[i] < values[i+1] and values[i] < trough_threshold_val:
            troughs.append({'decade': int(df.index[i]), 'value': float(values[i])})
    
    return {'peaks': peaks, 'troughs': troughs}

def calculate_trend(emotion, language, decade_start=1810, decade_end=1960):
    """Calculate linear regression trend"""
    df = get_time_series_data(language, emotion.lower(), decade_start, decade_end)
    
    if len(df) < 2:
        return {'error': 'Insufficient data'}
    
    x = np.arange(len(df))
    y = df.values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(r_value ** 2),
        'p_value': float(p_value),
        'std_error': float(std_err),
        'trend': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
        'interpretation': f"The {emotion} in {language} literature shows a {('positive' if slope > 0 else 'negative' if slope < 0 else 'no')} trend with R² = {r_value**2:.3f}"
    }

def calculate_cross_correlation(emotion1, emotion2, language, max_lag=5):
    """Calculate cross-correlation between two emotions"""
    try:
        df = get_language_dataframe(language)
        
        # Map emotion names to column names (handle case differences)
        emotion_map = {
            'Anger': 'anger', 'Contempt': 'contempt', 'Disgust': 'disgust',
            'Fear': 'fear', 'Frustration': 'frustration', 'Gratitude': 'gratitude',
            'Joy': 'joy', 'Love': 'love', 'Neutral': 'neutral',
            'Sadness': 'sadness', 'Surprise': 'surprise'
        }
        
        # Normalize emotion names to column names
        emotion1_col = emotion_map.get(emotion1, emotion1.lower())
        emotion2_col = emotion_map.get(emotion2, emotion2.lower())
        
        if emotion1_col not in df.columns or emotion2_col not in df.columns:
            return {'error': f'Emotion columns not found: {emotion1_col} or {emotion2_col}'}
        
        data1 = df[emotion1_col].values
        data2 = df[emotion2_col].values
        
        min_len = min(len(data1), len(data2))
        data1 = data1[:min_len]
        data2 = data2[:min_len]
        
        if min_len < 2:
            return {'error': 'Insufficient data for cross-correlation'}
        
        correlations = {}
        for lag in range(-max_lag, max_lag + 1):
            try:
                if lag < 0:
                    corr = np.corrcoef(data1[:lag], data2[-lag:])[0, 1]
                elif lag > 0:
                    corr = np.corrcoef(data1[lag:], data2[:-lag])[0, 1]
                else:
                    corr = np.corrcoef(data1, data2)[0, 1]
                
                correlations[f'lag_{lag}'] = float(corr) if not np.isnan(corr) else 0.0
            except:
                correlations[f'lag_{lag}'] = 0.0
        
        max_corr_val = max(correlations.values(), key=abs)
        # Handle floating point comparison issues
        max_corr_lag = None
        for k, v in correlations.items():
            if max_corr_lag is None or abs(v) >= abs(correlations[max_corr_lag]):
                max_corr_lag = k
        
        lag_0_val = correlations.get('lag_0', 0)
        
        return {
            'emotion1': emotion1,
            'emotion2': emotion2,
            'language': language,
            'correlations': correlations,
            'max_correlation': max_corr_val,
            'max_lag': max_corr_lag,
            'interpretation': f"{emotion1} and {emotion2} show {'positive' if lag_0_val > 0 else 'negative'} correlation (r={lag_0_val:.3f}) in {language} literature"
        }
    except Exception as e:
        return {'error': str(e)}

def get_literary_period_data(period):
    """Get data aggregated by literary period"""
    if period not in LITERARY_PERIODS:
        period = 'Romantic'
    
    start, end = LITERARY_PERIODS[period]
    
    result = {}
    for lang in ['English', 'Hindi', 'Tamil']:
        df = get_language_dataframe(lang)
        time_col = 'time_period' if 'time_period' in df.columns else 'decade'
        df = df[(df[time_col] >= start) & (df[time_col] <= end)]
        
        if len(df) > 0:
            result[lang] = df[EMOTION_COLUMNS].mean()
    
    return result

def get_multi_emotion_data(emotions, language, decade_start=1810, decade_end=1960):
    """Get data for multiple emotions comparison"""
    if language == 'All':
        languages = ['English', 'Hindi', 'Tamil']
    else:
        languages = [language]
    
    result = {}
    for lang in languages:
        df = get_language_dataframe(lang)
        time_col = 'time_period' if 'time_period' in df.columns else 'decade'
        df = df[(df[time_col] >= decade_start) & (df[time_col] <= decade_end)]
        
        emotion_data = {}
        for emotion in emotions:
            if emotion.lower() in EMOTION_COLUMNS:
                grouped = df.groupby(time_col)[emotion.lower()].mean()
                smoothed = grouped.rolling(window=3, min_periods=1, center=True).mean()
                emotion_data[emotion] = smoothed
        
        result[lang] = emotion_data
    
    return result

def detect_outliers(language, decade_start=1810, decade_end=1960):
    """Detect statistical outliers in the data"""
    df = get_language_dataframe(language)
    time_col = 'time_period' if 'time_period' in df.columns else 'decade'
    df = df[(df[time_col] >= decade_start) & (df[time_col] <= decade_end)]
    
    outliers = {}
    for emotion in EMOTION_COLUMNS:
        values = df[emotion]
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_mask = (values < lower_bound) | (values > upper_bound)
        outlier_indices = df[outlier_mask].index.tolist()
        
        if len(outlier_indices) > 0:
            outliers[emotion] = {
                'count': len(outlier_indices),
                'decades': [int(d) for d in outlier_indices],
                'values': values[outlier_mask].tolist()
            }
    
    return outliers

def get_timeseries_insights(emotion, language, decade_start=1810, decade_end=1960):
    """Calculate insights for time series graph - returns plain language + technical data"""
    try:
        if language == 'All':
            languages = ['English', 'Hindi', 'Tamil']
        else:
            languages = [language]
        
        results = {}
        for lang in languages:
            df = get_time_series_data(lang, emotion.lower(), decade_start, decade_end)
            if df is not None and len(df) > 0:
                values = df.values
                mean_val = float(np.mean(values))
                min_val = float(np.min(values))
                max_val = float(np.max(values))
                std_val = float(np.std(values))
                
                # Calculate trend
                if len(df) > 1:
                    x = np.arange(len(df))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                    trend = 'increasing' if slope > 0.001 else 'decreasing' if slope < -0.001 else 'stable'
                    
                    # Calculate percent change
                    if values[0] > 0:
                        pct_change = ((values[-1] - values[0]) / values[0]) * 100
                    else:
                        pct_change = 0
                else:
                    trend = 'stable'
                    slope = 0
                    pct_change = 0
                    r_value = 0
                    p_value = 1
                
                # Find peaks
                peaks = []
                for i in range(1, len(values) - 1):
                    if values[i] > values[i-1] and values[i] > values[i+1]:
                        peaks.append({'decade': int(df.index[i]), 'value': float(values[i])})
                peaks = sorted(peaks, key=lambda x: x['value'], reverse=True)[:3]
                
                peak_decade = peaks[0]['decade'] if peaks else None
                
                # Data coverage
                original_df = get_language_dataframe(lang)
                time_col = 'time_period' if 'time_period' in original_df.columns else 'decade'
                original_df = original_df[(original_df[time_col] >= decade_start) & (original_df[time_col] <= decade_end)]
                text_count = len(original_df)
                
                # Generate plain language insights
                trend_emoji = '📈' if trend == 'increasing' else '📉' if trend == 'decreasing' else '➡️'
                trend_text = 'Going Up' if trend == 'increasing' else 'Going Down' if trend == 'decreasing' else 'Staying Steady'
                
                # Create headline based on emotion
                if emotion.lower() == 'joy':
                    if trend == 'decreasing':
                        headline = f"Joy in {lang} literature has been declining"
                        what_means = "Literature became less openly joyful - modern books show more complex emotions"
                    elif trend == 'increasing':
                        headline = f"Joy in {lang} literature has been growing"
                        what_means = "Literature became more joyful over time - authors wrote with increasing optimism"
                    else:
                        headline = f"Joy in {lang} literature has remained steady"
                        what_means = "Joy levels have stayed consistent across the decades"
                elif emotion.lower() == 'sadness':
                    if trend == 'increasing':
                        headline = f"Sadness in {lang} literature has been growing"
                        what_means = "Literature reflects more melancholy - possibly influenced by historical events"
                    elif trend == 'decreasing':
                        headline = f"Sadness in {lang} literature has been declining"
                        what_means = "Literature became less sad - reflecting more optimistic eras"
                    else:
                        headline = f"Sadness in {lang} literature has remained steady"
                        what_means = "Sadness levels have stayed consistent"
                elif emotion.lower() == 'neutral':
                    headline = f"😐 Neutral is the baseline emotion in {lang}"
                    what_means = "This measures texts that express little strong emotion - a neutral baseline"
                else:
                    headline = f"{emotion} in {lang} shows a {trend} trend"
                    what_means = f"This emotion is {trend} over the {decade_start}s-{decade_end}s"
                
                # Peak description
                if peak_decade:
                    if peak_decade >= 1910 and peak_decade <= 1945:
                        peak_context = "during the World Wars"
                    elif peak_decade >= 1920 and peak_decade <= 1929:
                        peak_context = "in the Roaring Twenties"
                    elif peak_decade >= 1850 and peak_decade <= 1900:
                        peak_context = "during the Victorian era"
                    else:
                        peak_context = f"in the {peak_decade}s"
                    peak_desc = f"Highest in {peak_decade}s ({peak_context})"
                else:
                    peak_desc = "No significant peaks detected"
                
                results[lang] = {
                    'plain': {
                        'headline': headline,
                        'trend_emoji': trend_emoji,
                        'trend_text': trend_text,
                        'trend_direction': trend,
                        'peak_decade': peak_decade,
                        'peak_desc': peak_desc,
                        'pct_change': round(abs(pct_change), 1),
                        'change_direction': 'increased' if pct_change > 0 else 'decreased',
                        'what_means': what_means,
                        'data_sources': f"Based on {text_count} texts across {len(df)} decades"
                    },
                    'technical': {
                        'trend': trend,
                        'slope': float(slope),
                        'r_squared': float(r_value ** 2),
                        'p_value': float(p_value),
                        'mean': mean_val,
                        'min': min_val,
                        'max': max_val,
                        'std': std_val,
                        'peaks': peaks,
                        'pct_change': float(pct_change),
                        'text_count': text_count
                    }
                }
        
        return {'success': True, 'insights': results}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_multi_emotion_insights(emotions, language, decade_start=1810, decade_end=1960):
    """Calculate insights for multi-emotion comparison - returns plain + technical"""
    try:
        if language == 'All':
            languages = ['English', 'Hindi', 'Tamil']
        else:
            languages = [language]
        
        emotion_map = {
            'Anger': 'anger', 'Contempt': 'contempt', 'Disgust': 'disgust',
            'Fear': 'fear', 'Frustration': 'frustration', 'Gratitude': 'gratitude',
            'Joy': 'joy', 'Love': 'love', 'Neutral': 'neutral',
            'Sadness': 'sadness', 'Surprise': 'surprise'
        }
        
        results = {}
        for lang in languages:
            avg_intensities = {}
            for emotion in emotions:
                emotion_key = emotion_map.get(emotion, emotion.lower())
                df = get_time_series_data(lang, emotion_key, decade_start, decade_end)
                if df is not None and len(df) > 0:
                    avg_intensities[emotion] = float(np.mean(df.values))
            
            if avg_intensities:
                sorted_emotions = sorted(avg_intensities.items(), key=lambda x: x[1], reverse=True)
                # Skip Neutral to find the most meaningful highest emotion
                highest = None
                highest_val = 0
                for emotion, val in sorted_emotions:
                    if emotion.lower() != 'neutral':
                        highest = emotion
                        highest_val = val
                        break
                if highest is None:  # fallback if all are Neutral
                    highest = sorted_emotions[0][0] if sorted_emotions else None
                    highest_val = sorted_emotions[0][1] if sorted_emotions else 0
                lowest = sorted_emotions[-1][0] if sorted_emotions else None
                lowest_val = sorted_emotions[-1][1] if sorted_emotions else 0
                ranking = [{'emotion': e, 'avg': v} for e, v in sorted_emotions]
                
                # Plain language generation
                if highest:
                    if highest in ['Joy', 'Love', 'Gratitude']:
                        vibe = "optimistic and hopeful"
                    elif highest in ['Sadness', 'Fear', 'Anger']:
                        vibe = "somber and serious"
                    else:
                        vibe = "balanced and neutral"
                    
                    # Calculate ratio
                    if lowest_val > 0:
                        ratio = highest_val / lowest_val
                        ratio_text = f"about {int(ratio)}x more" if ratio > 1.5 else "slightly more"
                    else:
                        ratio_text = "significantly more"
                    
                    headline = f"🏆 {highest} is the most prominent emotion"
                    what_means = f"{lang} literature from this period feels {vibe}"
                    
                    if len(emotions) > 1:
                        insight = f"{highest} appears {ratio_text} than {lowest} - {vibe}"
                    else:
                        insight = f"This emotion is the dominant one in {lang} literature"
                else:
                    headline = "No clear emotion pattern"
                    what_means = "Not enough data to determine patterns"
                    insight = "Unable to analyze patterns"
                
                results[lang] = {
                    'plain': {
                        'headline': headline,
                        'highest_emotion': highest,
                        'lowest_emotion': lowest,
                        'ratio': ratio_text,
                        'vibe': vibe,
                        'insight': insight,
                        'what_means': what_means,
                        'emotion_count': len(emotions)
                    },
                    'technical': {
                        'highest': highest,
                        'lowest': lowest,
                        'highest_value': highest_val,
                        'lowest_value': lowest_val,
                        'ranking': ranking,
                        'avg_intensities': avg_intensities
                    }
                }
        
        return {'success': True, 'insights': results}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_radar_insights(decade, language):
    """Calculate insights for radar chart - returns plain + technical"""
    try:
        if language == 'All':
            languages = ['English', 'Hindi', 'Tamil']
        else:
            languages = [language]
        
        results = {}
        for lang in languages:
            emotions = get_emotions_by_decade(lang, decade)
            total = emotions.sum()
            if total > 0:
                normalized = (emotions / total) * 100
                sorted_emotions = sorted(zip(normalized.index, normalized.values), key=lambda x: x[1], reverse=True)

                # Skip Neutral to find the most meaningful dominant emotion
                dominant = None
                dominant_pct = 0
                for emotion, pct in sorted_emotions:
                    if emotion.lower() != 'neutral':
                        dominant = emotion
                        dominant_pct = float(pct)
                        break
                # If all emotions are Neutral (unlikely), fall back
                if dominant is None:
                    dominant = sorted_emotions[0][0] if sorted_emotions else None
                    dominant_pct = float(sorted_emotions[0][1]) if sorted_emotions else 0
                
                lowest = sorted_emotions[-1][0] if sorted_emotions else None
                lowest_pct = float(sorted_emotions[-1][1]) if sorted_emotions else 0
                
                top_three = [{'emotion': e, 'percentage': float(p)} for e, p in sorted_emotions[:3]]
                
                # Plain language insights
                if dominant in ['Joy', 'Love', 'Gratitude']:
                    vibe = "positive and hopeful"
                    cultural_note = f"This culture emphasized happiness and gratitude in their writing"
                elif dominant in ['Sadness', 'Fear', 'Anger']:
                    vibe = "somber and serious"
                    cultural_note = f"This era reflected difficult times through literature"
                elif dominant in ['Neutral']:
                    vibe = "balanced and practical"
                    cultural_note = f"Writers focused on objective, matter-of-fact storytelling"
                else:
                    vibe = "emotionally diverse"
                    cultural_note = f"Literature showed a mix of various emotional tones"
                
                headline = f"🏆 {dominant} ({dominant_pct:.1f}%) is the dominant emotion"
                what_means = f"This means writers in the {decade}s emphasized {vibe.lower()} themes"
                
                results[lang] = {
                    'plain': {
                        'headline': headline,
                        'dominant': dominant,
                        'dominant_pct': dominant_pct,
                        'lowest': lowest,
                        'lowest_pct': lowest_pct,
                        'vibe': vibe,
                        'cultural_note': cultural_note,
                        'what_means': what_means,
                        'top_three': [e['emotion'] for e in top_three]
                    },
                    'technical': {
                        'dominant': dominant,
                        'dominant_pct': dominant_pct,
                        'lowest': lowest,
                        'lowest_pct': lowest_pct,
                        'top_three': top_three,
                        'all_emotions': {e: float(normalized[e]) for e in normalized.index},
                        'raw_values': {e: float(emotions[e]) for e in emotions.index}
                    }
                }
        
        return {'success': True, 'insights': results}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_correlation_insights(language):
    """Calculate insights for correlation matrix - returns plain + technical"""
    try:
        if language == 'All':
            languages = ['English', 'Hindi', 'Tamil']
        else:
            languages = [language]
        
        results = {}
        for lang in languages:
            corr = get_correlation_matrix(lang)
            
            # Find correlations
            corr_values = []
            for i in range(len(corr)):
                for j in range(i+1, len(corr)):
                    corr_values.append({
                        'emotion1': corr.index[i],
                        'emotion2': corr.columns[j],
                        'value': float(corr.iloc[i, j])
                    })
            
            sorted_corr = sorted(corr_values, key=lambda x: x['value'], reverse=True)
            
            strongest_positive = sorted_corr[0] if sorted_corr and sorted_corr[0]['value'] > 0.1 else None
            strongest_negative = sorted_corr[-1] if sorted_corr and sorted_corr[-1]['value'] < -0.1 else None
            
            # Find independent emotions
            independence = {}
            for emotion in corr.index:
                other_corrs = [abs(corr.loc[emotion, other]) for other in corr.columns if other != emotion]
                independence[emotion] = float(np.mean(other_corrs))
            
            most_independent = min(independence.items(), key=lambda x: x[1])[0] if independence else None
            
            # Plain language insights
            if strongest_positive and strongest_positive['value'] > 0.5:
                relationship = f"{strongest_positive['emotion1']} and {strongest_positive['emotion2']} are best friends"
                simple_expl = "When one goes up, the other tends to go up too"
            elif strongest_positive:
                relationship = f"{strongest_positive['emotion1']} and {strongest_positive['emotion2']} often appear together"
                simple_expl = "They have a moderate tendency to appear together"
            else:
                relationship = "No strong positive relationships found"
                simple_expl = "No emotions strongly increase together"
            
            if strongest_negative and strongest_negative['value'] < -0.3:
                opposite = f"{strongest_negative['emotion1']} and {strongest_negative['emotion2']} are opposites"
                opposite_expl = "When one is high, the other tends to be low"
            elif strongest_negative:
                opposite = f"{strongest_negative['emotion1']} and {strongest_negative['emotion2']} tend to fight"
                opposite_expl = "They somewhat cancel each other out"
            else:
                opposite = "No strong opposing relationships"
                opposite_expl = "No emotions strongly oppose each other"
            
            results[lang] = {
                'plain': {
                    'relationship': relationship,
                    'simple_expl': simple_expl,
                    'opposite': opposite,
                    'opposite_expl': opposite_expl,
                    'independent': f"{most_independent} does its own thing" if most_independent else "All emotions show some connection",
                    'headline': f"Emotions in {lang} literature show predictable patterns",
                    'what_means': "Some emotions cluster together while others oppose each other"
                },
                'technical': {
                    'strongest_positive': strongest_positive,
                    'strongest_negative': strongest_negative,
                    'most_independent': most_independent,
                    'correlation_matrix': corr.to_dict(),
                    'all_correlations': sorted_corr[:10]
                }
            }
        
        return {'success': True, 'insights': results}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_mirror_insights(emotion, language, decade_start=1810, decade_end=1960):
    """Calculate insights for mirror hypothesis - returns plain + technical"""
    try:
        df = get_mirror_hypothesis_data(language, emotion.lower(), decade_start, decade_end)
        
        if df is None or len(df) < 2:
            return {'success': False, 'error': 'Insufficient data'}
        
        lit_values = df['Literature'].values
        hist_values = df['Historical'].values
        
        # Calculate correlation
        if len(lit_values) > 1:
            corr = np.corrcoef(lit_values, hist_values)[0, 1]
            corr = float(corr) if not np.isnan(corr) else 0
        else:
            corr = 0
        
        # Determine finding
        if abs(corr) > 0.5:
            finding = "Strong connection"
            evidence = "positive" if corr > 0 else "negative"
            emoji = "✅"
            simple_finding = f"YES! Books strongly reflect their era"
        elif abs(corr) > 0.3:
            finding = "Moderate connection"
            evidence = "positive" if corr > 0 else "negative"
            emoji = "⚠️"
            simple_finding = "Somewhat - literature partly reflects historical events"
        else:
            finding = "Weak or no clear connection"
            evidence = "none"
            emoji = "❌"
            simple_finding = "Not really - books don't clearly reflect their times"
        
        # Find peaks
        lit_peak_idx = np.argmax(lit_values)
        hist_peak_idx = np.argmax(hist_values)
        lit_peak_decade = int(df.index[lit_peak_idx]) if lit_peak_idx < len(df) else None
        hist_peak_decade = int(df.index[hist_peak_idx]) if hist_peak_idx < len(df) else None
        
        # Plain language
        strength_pct = int(abs(corr) * 100)
        
        if evidence == "positive":
            what_means = f"What happened in the world showed up in books - when history was intense, literature matched that mood"
            simple_expl = f"{strength_pct}% of the time, book emotions match historical events"
        elif evidence == "negative":
            what_means = f"Books sometimes went against the grain - when history was happy, writers explored darker themes"
            simple_expl = f"{strength_pct}% of the time, book emotions went opposite to history"
        else:
            what_means = f"Literature seems to be independent of historical events during this period"
            simple_expl = "Books don't seem to reflect historical events"
        
        return {
            'success': True,
            'insights': {
                'plain': {
                    'emoji': emoji,
                    'simple_finding': simple_finding,
                    'strength_pct': strength_pct,
                    'literature_peak': f"{lit_peak_decade}s" if lit_peak_decade else "N/A",
                    'history_peak': f"{hist_peak_decade}s" if hist_peak_decade else "N/A",
                    'what_means': what_means,
                    'simple_expl': simple_expl,
                    'headline': f"Writers were {'' if evidence != 'none' else 'not '}influenced by their times"
                },
                'technical': {
                    'correlation': corr,
                    'finding': finding,
                    'evidence': evidence,
                    'literature_peak': lit_peak_decade,
                    'history_peak': hist_peak_decade,
                    'literature_values': [float(v) for v in lit_values],
                    'history_values': [float(v) for v in hist_values]
                }
            }
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_heatmap_insights(language, decade_start=1810, decade_end=1960):
    """Calculate insights for heatmap - returns plain + technical"""
    try:
        if language == 'All':
            languages = ['English', 'Hindi', 'Tamil']
        else:
            languages = [language]
        
        results = {}
        for lang in languages:
            data = get_heatmap_data(lang, decade_start, decade_end)
            
            if lang in data:
                df = data[lang]
                
                # Find hottest decade
                decade_totals = df.sum(axis=1)
                hottest_decade = int(decade_totals.idxmax()) if len(decade_totals) > 0 else None
                
                # Find hottest emotion (skip Neutral)
                emotion_totals = df.sum(axis=0)
                # Exclude Neutral (try both capitalizations)
                emotion_without_neutral = emotion_totals.drop('neutral', errors='ignore')
                if len(emotion_without_neutral) == len(emotion_totals):  # 'neutral' wasn't found, try 'Neutral'
                    emotion_without_neutral = emotion_totals.drop('Neutral', errors='ignore')
                hottest_emotion = emotion_without_neutral.idxmax() if len(emotion_without_neutral) > 0 else None
                
                # Calculate overall trend
                if len(decade_totals) > 1:
                    x = np.arange(len(decade_totals))
                    slope, _, _, _, _ = stats.linregress(x, decade_totals.values)
                    if slope > 0.01:
                        trend = "heating up"
                        trend_emoji = "🔥"
                        trend_desc = "Emotions in literature have been getting more intense"
                    elif slope < -0.01:
                        trend = "cooling down"
                        trend_emoji = "❄️"
                        trend_desc = "Emotions in literature have been getting less intense"
                    else:
                        trend = "stable"
                        trend_emoji = "➡️"
                        trend_desc = "Emotion levels have remained consistent"
                else:
                    trend = "variable"
                    trend_emoji = "〰️"
                    trend_desc = "Emotion patterns vary by decade"
                
                # Plain language
                if hottest_decade:
                    if hottest_decade >= 1910 and hottest_decade <= 1945:
                        era = "the World Wars period"
                    elif hottest_decade >= 1920 and hottest_decade <= 1929:
                        era = "the Roaring Twenties"
                    elif hottest_decade >= 1850 and hottest_decade <= 1900:
                        era = "the Victorian era"
                    else:
                        era = f"the {hottest_decade}s"
                    peak_desc = f"🔥 Hottest was {hottest_decade}s ({era})"
                else:
                    peak_desc = "No clear peak detected"
                
                headline = f"{trend_emoji} Emotions have been {trend} over time"
                what_means = trend_desc
                
                results[lang] = {
                    'plain': {
                        'headline': headline,
                        'peak_desc': peak_desc,
                        'top_emotion': f"{hottest_emotion} was the strongest emotion" if hottest_emotion else "No dominant emotion",
                        'trend_emoji': trend_emoji,
                        'trend': trend,
                        'what_means': what_means,
                        'emotion_count': len(emotion_totals),
                        'decade_count': len(decade_totals)
                    },
                    'technical': {
                        'hottest_decade': hottest_decade,
                        'hottest_emotion': hottest_emotion,
                        'trend': trend,
                        'decade_totals': {str(k): float(v) for k, v in decade_totals.items()},
                        'emotion_totals': {str(k): float(v) for k, v in emotion_totals.items()},
                        'ranking': [{'emotion': e, 'total': float(v)} for e, v in sorted(zip(emotion_totals.index, emotion_totals.values), key=lambda x: x[1], reverse=True)]
                    }
                }
        
        return {'success': True, 'insights': results}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_distribution_insights(language, decade_start=1810, decade_end=1960):
    """Calculate insights for distribution plot - returns plain + technical"""
    try:
        if language == 'All':
            languages = ['English', 'Hindi', 'Tamil']
        else:
            languages = [language]
        
        results = {}
        for lang in languages:
            data = get_distribution_data(lang, decade_start, decade_end)
            
            if lang in data:
                df = data[lang]
                
                # Calculate stats for each emotion
                stats_dict = {}
                outliers = []
                overall_mean = 0
                emotion_count = 0
                
                for col in df.columns:
                    values = df[col].dropna()
                    if len(values) > 0:
                        mean_val = float(np.mean(values))
                        median_val = float(np.median(values))
                        std_val = float(np.std(values))
                        min_val = float(np.min(values))
                        max_val = float(np.max(values))
                        
                        # Detect outliers
                        q1 = values.quantile(0.25)
                        q3 = values.quantile(0.75)
                        iqr = q3 - q1
                        lower = q1 - 1.5 * iqr
                        upper = q3 + 1.5 * iqr
                        outlier_mask = (values < lower) | (values > upper)
                        outlier_count = int(outlier_mask.sum())
                        
                        # Determine skewness
                        if mean_val > median_val:
                            shape = "slightly higher in some texts"
                        elif mean_val < median_val:
                            shape = "more balanced"
                        else:
                            shape = "evenly distributed"
                        
                        stats_dict[col] = {
                            'mean': mean_val,
                            'median': median_val,
                            'std': std_val,
                            'min': min_val,
                            'max': max_val,
                            'outliers': outlier_count,
                            'shape': shape
                        }
                        
                        if outlier_count > 0:
                            outliers.append({'emotion': col, 'count': outlier_count})
                        
                        overall_mean += mean_val
                        emotion_count += 1
                
                avg_intensity = overall_mean / emotion_count if emotion_count > 0 else 0

                # Find highest and most volatile emotions (skip Neutral)
                emotions_without_neutral = {k: v for k, v in stats_dict.items() if k.lower() != 'neutral'}
                highest_emotion = None
                most_volatile = None
                if emotions_without_neutral:
                    highest_emotion = max(emotions_without_neutral.items(), key=lambda x: x[1]['mean'])[0]
                    most_volatile = max(emotions_without_neutral.items(), key=lambda x: x[1]['std'])[0]

                # Plain language
                range_min = min(s['min'] for s in stats_dict.values()) if stats_dict else 0
                range_max = max(s['max'] for s in stats_dict.values()) if stats_dict else 0
                
                if outliers:
                    outlier_text = f"{len(outliers)} emotions have unusual peaks"
                    outlier_desc = "Some texts are much more expressive than typical"
                else:
                    outlier_text = "No extreme outliers"
                    outlier_desc = "All texts fall within normal emotional ranges"
                
                headline = f"📊 Average emotion level is {avg_intensity*100:.1f}%"
                what_means = f"Most texts in {lang} show {shape.lower()}. {outlier_desc}"
                if highest_emotion:
                    what_means += f" {highest_emotion} is the most prevalent emotion."
                if most_volatile:
                    what_means += f" {most_volatile} shows the most variation."

                results[lang] = {
                    'plain': {
                        'headline': headline,
                        'avg_intensity': f"{avg_intensity*100:.1f}%",
                        'typical_range': f"Most texts between {range_min*100:.0f}-{range_max*100:.0f}%",
                        'outlier_text': outlier_text,
                        'outlier_desc': outlier_desc,
                        'what_means': what_means,
                        'emotion_count': emotion_count,
                        'highest_emotion': highest_emotion,
                        'most_volatile': most_volatile
                    },
                    'technical': {
                        'stats': stats_dict,
                        'outliers': outliers,
                        'overall_mean': overall_mean,
                        'avg_intensity': avg_intensity,
                        'highest_emotion': highest_emotion,
                        'most_volatile': most_volatile
                    }
                }
        
        return {'success': True, 'insights': results}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_period_insights(period):
    """Calculate insights for literary period comparison - returns plain + technical"""
    try:
        valid_periods = LITERARY_PERIODS
        if period not in valid_periods:
            period = 'Romantic'
        
        period_data = get_literary_period_data(period)
        
        if not period_data:
            return {'success': False, 'error': 'No data for period'}
        
        # Find most intense period overall
        period_totals = {}
        for lang, values in period_data.items():
            period_totals[lang] = float(values.sum())
        
        # Calculate avg intensity
        avg_intensity = float(np.mean([v for vals in period_data.values() for v in vals.values])) if period_data else 0
        start, end = valid_periods.get(period, (1810, 1840))
        
        # Plain language insights
        if period == 'Romantic':
            period_desc = "focused on emotion, nature, and individualism"
            period_emoji = "💕"
            recommendation = "If you want passionate, heartfelt stories"
        elif period == 'Victorian':
            period_desc = "emphasized social order, morality, and detail"
            period_emoji = "🎩"
            recommendation = "If you want complex plots and moral dilemmas"
        elif period == 'Edwardian':
            period_desc = "bridged Victorian and Modern, with wit and social commentary"
            period_emoji = "🥂"
            recommendation = "If you want light social satire"
        else:  # Modern
            period_desc = "broke traditional rules with experimental styles"
            period_emoji = "🔬"
            recommendation = "If you want innovative, boundary-pushing works"
        
        # Find which emotion is highest
        highest_emotion = None
        highest_value = 0
        for lang, values in period_data.items():
            for emotion, val in values.items():
                if val > highest_value:
                    highest_value = val
                    highest_emotion = emotion
        
        if highest_emotion:
            if highest_emotion in ['Joy', 'Love']:
                emotion_note = f"notably {highest_emotion.lower()}ful"
            elif highest_emotion in ['Sadness']:
                emotion_note = f"notably {highest_emotion.lower()}"
            else:
                emotion_note = f"high in {highest_emotion.lower()}"
        else:
            emotion_note = "varied emotions"
        
        headline = f"{period_emoji} The {period} era ({start}-{end}) was characterized by writers who {period_desc}"
        what_means = f"This period shows distinctive emotional patterns. {recommendation}, try {period} literature."
        
        return {
            'success': True,
            'insights': {
                'plain': {
                    'headline': headline,
                    'period_desc': period_desc,
                    'period_emoji': period_emoji,
                    'recommendation': recommendation,
                    'highest_emotion': highest_emotion,
                    'emotion_note': emotion_note,
                    'avg_intensity': f"{avg_intensity*100:.1f}%",
                    'languages': list(period_data.keys()),
                    'what_means': what_means
                },
                'technical': {
                    'period': period,
                    'year_range': valid_periods[period],
                    'languages': list(period_data.keys()),
                    'avg_intensity': avg_intensity,
                    'emotion_data': {lang: {str(k): float(v) for k, v in vals.items()} for lang, vals in period_data.items()}
                }
            }
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_cross_corr_insights(emotion1, emotion2, language, max_lag=5):
    """Calculate insights for cross-correlation - returns plain + technical"""
    try:
        result = calculate_cross_correlation(emotion1, emotion2, language, max_lag)
        
        if 'error' in result:
            return {'success': False, 'error': result['error']}
        
        correlations = result.get('correlations', {})
        lag_0 = correlations.get('lag_0', 0)
        
        # Determine strength
        abs_corr = abs(lag_0)
        if abs_corr > 0.7:
            strength = "Strong"
            strength_emoji = "⚡⚡⚡"
            strength_desc = "very strongly connected"
        elif abs_corr > 0.4:
            strength = "Moderate"
            strength_emoji = "⚡⚡"
            strength_desc = "somewhat connected"
        elif abs_corr > 0.2:
            strength = "Weak"
            strength_emoji = "⚡"
            strength_desc = "loosely connected"
        else:
            strength = "Very weak"
            strength_emoji = "🙈"
            strength_desc = "barely related"
        
        # Find best lag
        best_lag = 0
        best_corr = 0
        for lag, val in correlations.items():
            if val is not None and abs(val) > abs(best_corr):
                best_lag = int(lag.replace('lag_', ''))
                best_corr = val
        
        # Plain language
        if lag_0 > 0:
            relationship = "🤝 best friends - they go up and down together"
            simple_expl = f"When {emotion1} is high, {emotion2} tends to be high too"
            metaphor = "Like two dancers moving in sync"
        elif lag_0 < 0:
            relationship = "⚔️ opposites - when one rises, the other falls"
            simple_expl = f"When {emotion1} is high, {emotion2} tends to be low"
            metaphor = "Like a seesaw - up on one side means down on the other"
        else:
            relationship = "🙈 strangers - they don't really affect each other"
            simple_expl = f"{emotion1} and {emotion2} don't seem related"
            metaphor = "Like two unrelated things happening by coincidence"
        
        if best_lag == 0:
            timing_text = "They react at the same time"
        elif best_lag > 0:
            timing_text = f"{emotion2} peaks about {best_lag} decade(s) after {emotion1}"
        else:
            timing_text = f"{emotion1} peaks about {abs(best_lag)} decade(s) after {emotion2}"
        
        strength_pct = int(abs_corr * 100)
        headline = f"{strength_emoji} {emotion1} and {emotion2} are {strength_desc}"
        what_means = f"{simple_expl}. {metaphor}."
        
        return {
            'success': True,
            'insights': {
                'plain': {
                    'headline': headline,
                    'relationship': relationship,
                    'simple_expl': simple_expl,
                    'metaphor': metaphor,
                    'timing_text': timing_text,
                    'strength_pct': strength_pct,
                    'direction': "together" if lag_0 > 0 else "opposite" if lag_0 < 0 else "unrelated",
                    'what_means': what_means
                },
                'technical': {
                    'correlation': lag_0,
                    'strength': strength,
                    'best_lag': best_lag,
                    'best_lag_corr': best_corr,
                    'correlations': correlations
                }
            }
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}