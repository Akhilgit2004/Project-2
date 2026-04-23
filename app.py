from flask import Flask, render_template, request, send_file, jsonify
import os
import uuid
import data_utils
import graph_generator

app = Flask(__name__)

# Ensure graphs directory exists
GRAPH_DIR = os.path.join(os.path.dirname(__file__), 'static', 'graphs')
os.makedirs(GRAPH_DIR, exist_ok=True)

@app.route('/')
def index():
    """Main dashboard"""
    emotions = data_utils.EMOTION_LABELS
    decades = data_utils.VALID_DECADES
    languages = ['English', 'Hindi', 'Tamil', 'All']
    return render_template('index.html', 
                         emotions=emotions, 
                         decades=decades,
                         languages=languages)

@app.route('/generate', methods=['POST'])
def generate():
    """Generate graph based on request"""
    data = request.json
    
    graph_type = data.get('type', 'timeseries')
    emotion = data.get('emotion', 'Joy')
    decade = int(data.get('decade', 1940))
    language = data.get('language', 'All')
    
    # Generate unique filename
    filename = f"{graph_type}_{emotion}_{decade}_{language}_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(GRAPH_DIR, filename)
    
    try:
        if graph_type == 'timeseries':
            graph_generator.generate_time_series(emotion, language, filepath)
        elif graph_type == 'radar':
            graph_generator.generate_radar_chart(decade, language, filepath)
        elif graph_type == 'correlation':
            graph_generator.generate_correlation_matrix(language, filepath)
        elif graph_type == 'mirror':
            # Mirror hypothesis uses specific language (not 'All')
            mirror_language = language if language != 'All' else 'English'
            graph_generator.generate_mirror_hypothesis(emotion, mirror_language, filepath)
        else:
            return jsonify({'error': 'Invalid graph type'}), 400
        
        # Return the path for the frontend to load
        return jsonify({'success': True, 'graph_url': f'/static/graphs/{filename}'})
    
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

if __name__ == '__main__':
    print("=" * 60)
    print("Loading data...")
    data_utils.load_all_data()
    print("Data loaded successfully!")
    print("=" * 60)
    app.run(debug=True, port=5000)