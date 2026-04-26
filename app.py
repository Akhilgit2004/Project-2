from flask import Flask, render_template, request, send_file, jsonify, make_response
import os
import uuid
import json
import data_utils
import graph_generator

app = Flask(__name__)

GRAPH_DIR = os.path.join(os.path.dirname(__file__), 'static', 'graphs')
os.makedirs(GRAPH_DIR, exist_ok=True)

@app.route('/')
def index():
    """Main dashboard"""
    emotions = data_utils.EMOTION_LABELS
    decades = data_utils.VALID_DECADES
    languages = ['English', 'Hindi', 'Tamil', 'All']
    literary_periods = data_utils.LITERARY_PERIODS
    return render_template('index.html', 
                         emotions=emotions, 
                         decades=decades,
                         languages=languages,
                         literary_periods=literary_periods)

@app.route('/generate', methods=['POST'])
def generate():
    """Generate graph based on request"""
    data = request.json
    
    graph_type = data.get('type', 'timeseries')
    emotion = data.get('emotion', 'Joy')
    decade = int(data.get('decade', 1940))
    language = data.get('language', 'All')
    decade_start = int(data.get('decade_start', 1810))
    decade_end = int(data.get('decade_end', 1960))
    emotions = data.get('emotions', [emotion])
    show_trendline = data.get('show_trendline', False)
    show_peaks = data.get('show_peaks', False)
    window_size = int(data.get('window_size', 3))
    
    try:
        if graph_type == 'timeseries':
            graph_json = graph_generator.generate_time_series(
                emotion, language,
                decade_start=decade_start,
                decade_end=decade_end,
                show_trendline=show_trendline,
                window_size=window_size,
                return_json=True
            )
        elif graph_type == 'multi_emotion':
            graph_json = graph_generator.generate_multi_emotion_comparison(
                emotions, language,
                decade_start=decade_start,
                decade_end=decade_end,
                show_trendline=show_trendline,
                return_json=True
            )
        elif graph_type == 'radar':
            graph_json = graph_generator.generate_radar_chart(decade, language, return_json=True)
        elif graph_type == 'correlation':
            graph_json = graph_generator.generate_correlation_matrix(language, return_json=True)
        elif graph_type == 'mirror':
            mirror_language = language if language != 'All' else 'English'
            graph_json = graph_generator.generate_mirror_hypothesis(
                emotion, mirror_language,
                decade_start=decade_start,
                decade_end=decade_end,
                return_json=True
            )
        elif graph_type == 'heatmap':
            graph_json = graph_generator.generate_heatmap_timeline(language, decade_start, decade_end, return_json=True)
        elif graph_type == 'distribution':
            graph_json = graph_generator.generate_distribution_plot(language, decade_start, decade_end, return_json=True)
        elif graph_type == 'literary_period':
            period = data.get('period', 'Romantic')
            graph_json = graph_generator.generate_literary_period_comparison(period, return_json=True)
        elif graph_type == 'cross_correlation':
            emotion1 = data.get('emotion1', 'Joy')
            emotion2 = data.get('emotion2', 'Sadness')
            graph_json = graph_generator.generate_cross_correlation(
                emotion1, emotion2, language, max_lag=int(data.get('max_lag', 5)), return_json=True
            )
        else:
            return jsonify({'error': 'Invalid graph type'}), 400
        
        return jsonify({'success': True, 'graph_data': graph_json})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_all_correlation', methods=['POST'])
def generate_all_correlation():
    """Generate all three correlation matrices"""
    data = request.json
    language = data.get('language', 'English')
    
    filenames = {}
    for lang in ['English', 'Hindi', 'Tamil']:
        filename = f"correlation_{lang}_{uuid.uuid4().hex[:8]}.png"
        filepath = os.path.join(GRAPH_DIR, filename)
        graph_generator.generate_correlation_matrix(lang, filepath)
        filenames[lang] = f'/static/graphs/{filename}'
    
    return jsonify({'success': True, 'graphs': filenames})

@app.route('/api/data', methods=['POST'])
def get_data():
    """API endpoint to get raw data for export/analysis"""
    data = request.json
    
    data_type = data.get('data_type', 'timeseries')
    emotion = data.get('emotion', 'Joy')
    language = data.get('language', 'All')
    decade_start = int(data.get('decade_start', 1810))
    decade_end = int(data.get('decade_end', 1960))
    
    try:
        if data_type == 'timeseries':
            if language == 'All':
                languages = ['English', 'Hindi', 'Tamil']
            else:
                languages = [language]
            
            result = {}
            for lang in languages:
                df = data_utils.get_time_series_data(lang, emotion.lower())
                df = df[(df.index >= decade_start) & (df.index <= decade_end)]
                result[lang] = {str(k): float(v) for k, v in df.items()}
            
            return jsonify({'success': True, 'data': result})
        
        elif data_type == 'heatmap':
            if language == 'All':
                languages = ['English', 'Hindi', 'Tamil']
            else:
                languages = [language]
            
            result = {}
            for lang in languages:
                df = data_utils.get_language_dataframe(lang)
                time_col = 'time_period' if 'time_period' in df.columns else 'decade'
                df = df[(df[time_col] >= decade_start) & (df[time_col] <= decade_end)]
                grouped = df.groupby(time_col)[data_utils.EMOTION_COLUMNS].mean()
                result[lang] = {str(k): {e: float(v) for e, v in row.items()} for k, row in grouped.iterrows()}
            
            return jsonify({'success': True, 'data': result})
        
        elif data_type == 'all_emotions':
            if language == 'All':
                languages = ['English', 'Hindi', 'Tamil']
            else:
                languages = [language]
            
            result = {}
            for lang in languages:
                df = data_utils.get_language_dataframe(lang)
                time_col = 'time_period' if 'time_period' in df.columns else 'decade'
                df = df[(df[time_col] >= decade_start) & (df[time_col] <= decade_end)]
                grouped = df.groupby(time_col)[data_utils.EMOTION_COLUMNS].mean()
                result[lang] = {str(k): {e: float(v) for e, v in row.items()} for k, row in grouped.iterrows()}
            
            return jsonify({'success': True, 'data': result})
        
        return jsonify({'error': 'Invalid data type'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics', methods=['POST'])
def get_statistics():
    """API endpoint for statistical analysis"""
    data = request.json
    
    analysis_type = data.get('analysis_type', 'significance')
    emotion = data.get('emotion', 'Joy')
    language1 = data.get('language1', 'English')
    language2 = data.get('language2', 'Hindi')
    decade_start = int(data.get('decade_start', 1810))
    decade_end = int(data.get('decade_end', 1960))
    
    try:
        if analysis_type == 'significance':
            result = data_utils.calculate_significance(
                emotion, language1, language2, decade_start, decade_end
            )
            return jsonify({'success': True, 'data': result})
        
        elif analysis_type == 'peaks':
            language = data.get('language', 'English')
            result = data_utils.detect_peaks(
                emotion, language, decade_start, decade_end
            )
            return jsonify({'success': True, 'data': result})
        
        elif analysis_type == 'trend':
            language = data.get('language', 'English')
            result = data_utils.calculate_trend(
                emotion, language, decade_start, decade_end
            )
            return jsonify({'success': True, 'data': result})
        
        elif analysis_type == 'cross_correlation':
            emotion1 = data.get('emotion1', 'Joy')
            emotion2 = data.get('emotion2', 'Sadness')
            language = data.get('language', 'English')
            max_lag = data.get('max_lag', 5)
            result = data_utils.calculate_cross_correlation(
                emotion1, emotion2, language, max_lag
            )
            return jsonify({'success': True, 'data': result})
        
        return jsonify({'error': 'Invalid analysis type'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/insights', methods=['POST'])
def get_insights():
    """API endpoint for graph insights"""
    data = request.json
    
    graph_type = data.get('graph_type')
    emotion = data.get('emotion', 'Joy')
    language = data.get('language', 'All')
    decade_start = int(data.get('decade_start', 1810))
    decade_end = int(data.get('decade_end', 1960))
    emotions = data.get('emotions', [emotion])
    decade = int(data.get('decade', 1940))
    period = data.get('period', 'Romantic')
    emotion1 = data.get('emotion1', 'Joy')
    emotion2 = data.get('emotion2', 'Sadness')
    max_lag = int(data.get('max_lag', 5))
    
    try:
        if graph_type == 'timeseries':
            result = data_utils.get_timeseries_insights(emotion, language, decade_start, decade_end)
        
        elif graph_type == 'multi_emotion':
            result = data_utils.get_multi_emotion_insights(emotions, language, decade_start, decade_end)
        
        elif graph_type == 'radar':
            result = data_utils.get_radar_insights(decade, language)
        
        elif graph_type == 'correlation':
            result = data_utils.get_correlation_insights(language)
        
        elif graph_type == 'mirror':
            mirror_lang = language if language != 'All' else 'English'
            result = data_utils.get_mirror_insights(emotion, mirror_lang, decade_start, decade_end)
        
        elif graph_type == 'heatmap':
            result = data_utils.get_heatmap_insights(language, decade_start, decade_end)
        
        elif graph_type == 'distribution':
            result = data_utils.get_distribution_insights(language, decade_start, decade_end)
        
        elif graph_type == 'literary_period':
            result = data_utils.get_period_insights(period)
        
        elif graph_type == 'cross_correlation':
            result = data_utils.get_cross_corr_insights(emotion1, emotion2, language, max_lag)
        
        else:
            return jsonify({'success': False, 'error': 'Invalid graph type'}), 400
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/preferences', methods=['POST'])
def save_preferences():
    """Save user preferences"""
    data = request.json
    preferences = data.get('preferences', {})
    
    try:
        prefs_file = os.path.join(os.path.dirname(__file__), 'user_preferences.json')
        with open(prefs_file, 'w') as f:
            json.dump(preferences, f)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/preferences', methods=['GET'])
def get_preferences():
    """Get user preferences"""
    try:
        prefs_file = os.path.join(os.path.dirname(__file__), 'user_preferences.json')
        if os.path.exists(prefs_file):
            with open(prefs_file, 'r') as f:
                preferences = json.load(f)
            return jsonify({'success': True, 'preferences': preferences})
        return jsonify({'success': True, 'preferences': {}})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/annotations', methods=['POST'])
def save_annotations():
    """Save graph annotations"""
    data = request.json
    annotations = data.get('annotations', {})
    
    try:
        annotations_file = os.path.join(os.path.dirname(__file__), 'annotations.json')
        with open(annotations_file, 'w') as f:
            json.dump(annotations, f)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/annotations', methods=['GET'])
def get_annotations():
    """Get graph annotations"""
    try:
        annotations_file = os.path.join(os.path.dirname(__file__), 'annotations.json')
        if os.path.exists(annotations_file):
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)
            return jsonify({'success': True, 'annotations': annotations})
        return jsonify({'success': True, 'annotations': {}})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/png/<path:filename>')
def download_png(filename):
    """Download graph as PNG"""
    filepath = os.path.join(GRAPH_DIR, filename)
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='image/png', as_attachment=True)
    return "File not found", 404

@app.route('/methodology')
def methodology():
    """Methodology page"""
    return render_template('methodology.html')

@app.route('/glossary')
def glossary():
    """Glossary page"""
    return render_template('glossary.html')

@app.route('/discoveries')
def discoveries():
    """Interesting findings page"""
    return render_template('discoveries.html')

if __name__ == '__main__':
    print("=" * 60)
    print("Loading data...")
    data_utils.load_all_data()
    print("Data loaded successfully!")
    print("=" * 60)
    app.run(debug=True, port=8000)