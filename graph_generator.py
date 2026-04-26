import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import data_utils
import kaleido
import os

PLOTLY_TEMPLATE = "plotly_white"

# Universal hover configuration - fixes hover text positioning issues
UNIVERSAL_HOVERCONFIG = dict(
    hoverlabel=dict(
        bgcolor='rgba(255,255,255,0.95)',
        bordercolor='rgba(0,0,0,0.1)',
        font_size=11,
        align='left',
        namelength=-1
    )
)

# Convert matplotlib markers to Plotly markers
MARKER_MAP = {
    'o': 'circle',
    's': 'square',
    '^': 'triangle-up',
    'v': 'triangle-down',
    'd': 'diamond',
    'p': 'pentagon',
    'h': 'hexagon',
    '*': 'star',
    'x': 'x-thin',
    '+': 'plus',
    '.': 'circle-open',
}

def generate_time_series(emotion, language, output_path=None, decade_start=1810, decade_end=1960, 
                         show_trendline=False, window_size=3, return_json=False):
    """Generate Time Series with Rolling Average chart using Plotly"""
    if language == 'All':
        languages = ['English', 'Hindi', 'Tamil']
    else:
        languages = [language]
    
    df = data_utils.get_time_series_multilang(languages, emotion.lower(), decade_start, decade_end)
    
    fig = go.Figure()
    
    for lang in languages:
        if lang in df.columns:
            lang_data = df[lang].dropna()
            
            # Add main line
            fig.add_trace(go.Scatter(
                x=lang_data.index.tolist(),
                y=lang_data.values.tolist(),
                mode='lines+markers',
                name=lang,
                line=dict(
                    color=data_utils.COLOR_PALETTE[lang],
                    width=3
                ),
                marker=dict(
                    size=10,
                    symbol=MARKER_MAP.get(data_utils.MARKERS.get(lang, 'circle'), 'circle')
                ),
                hovertemplate=f'<b>{lang}</b><br>' +
                              'Decade: %{x}<br>' +
                              'Intensity: %{y:.4f}<extra></extra>',
                hoverlabel=UNIVERSAL_HOVERCONFIG['hoverlabel']
            ))
            
            if show_trendline and len(lang_data) > 1:
                x = np.arange(len(lang_data))
                y = lang_data.values
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                
                fig.add_trace(go.Scatter(
                    x=lang_data.index.tolist(),
                    y=p(x).tolist(),
                    mode='lines',
                    name=f'{lang} Trend',
                    line=dict(
                        color=data_utils.COLOR_PALETTE[lang],
                        width=2,
                        dash='dash'
                    ),
                    hoverinfo='skip'
                ))
    
    fig.update_layout(
        title=dict(
            text=f'Cross-Cultural Trajectory of {emotion.capitalize()} in Literature ({decade_start}–{decade_end})',
            font=dict(size=18, color='#1f77b4'),
            x=0.5
        ),
        xaxis=dict(
            title='Time Period (Decade)',
            tickmode='array',
            tickvals=data_utils.VALID_DECADES,
            ticktext=[str(d) for d in data_utils.VALID_DECADES],
            tickangle=45,
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)'
        ),
        yaxis=dict(
            title='3-Decade Rolling Mean Intensity',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)'
        ),
        legend=dict(
            title='Language',
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        ),
        hovermode='x unified',
        hoverdistance=100,
        template=PLOTLY_TEMPLATE,
        margin=dict(t=120, l=60, r=40, b=100),
        height=500,
        dragmode='zoom'
    )
    
    if return_json:
        return fig.to_json()
    
    # Save as static PNG
    if output_path:
        fig.write_image(output_path, width=1400, height=800, scale=2)
    
    return output_path
def generate_multi_emotion_comparison(emotions, language, output_path=None, decade_start=1810, decade_end=1960,
                                       show_trendline=False, return_json=False):
    """Generate Multi-Emotion Comparison chart using Plotly"""
    if not emotions or len(emotions) == 0:
        emotions = ['Joy']
    
    emotion_map = {
        'Anger': 'anger', 'Contempt': 'contempt', 'Disgust': 'disgust',
        'Fear': 'fear', 'Frustration': 'frustration', 'Gratitude': 'gratitude',
        'Joy': 'joy', 'Love': 'love', 'Neutral': 'neutral',
        'Sadness': 'sadness', 'Surprise': 'surprise'
    }
    
    if language == 'All':
        languages = ['English', 'Hindi', 'Tamil']
    else:
        languages = [language]
    
    emotion_colors = list(data_utils.COLOR_PALETTE.values())[:len(emotions)]
    line_styles = ['solid', 'dash', 'dot', 'dashdot']
    
    fig = go.Figure()
    has_data = False
    
    for idx, emotion in enumerate(emotions):
        emotion_key = emotion_map.get(emotion, emotion.lower())
        for lang_idx, lang in enumerate(languages):
            df = data_utils.get_time_series_data(lang, emotion_key, decade_start, decade_end)
            
            if df is None or len(df) == 0:
                continue
            
            has_data = True
            label = f"{emotion} - {lang}" if len(languages) > 1 else emotion
            
            fig.add_trace(go.Scatter(
                x=df.index.tolist(),
                y=df.values.tolist(),
                mode='lines+markers',
                name=label,
                line=dict(
                    color=emotion_colors[idx],
                    width=2.5,
                    dash=line_styles[idx % len(line_styles)]
                ),
                marker=dict(size=7),
                hovertemplate=f'<b>{label}</b><br>' +
                              'Decade: %{x}<br>' +
                              'Intensity: %{y:.4f}<extra></extra>',
                hoverlabel=UNIVERSAL_HOVERCONFIG['hoverlabel']
            ))
    
    if not has_data:
        fig.add_annotation(
            text='No data available for selected parameters',
            showarrow=False,
            font=dict(size=16)
        )
    
    fig.update_layout(
        title=dict(
            text=f'Multi-Emotion Comparison in {language} Literature ({decade_start}–{decade_end})',
            font=dict(size=18, color='#1f77b4'),
            x=0.5
        ),
        xaxis=dict(
            title='Time Period (Decade)',
            tickmode='array',
            tickvals=data_utils.VALID_DECADES,
            ticktext=[str(d) for d in data_utils.VALID_DECADES],
            tickangle=45,
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title=dict(text='Emotion Intensity', font=dict(size=14)),
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        legend=dict(
            title='Emotion - Language',
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        ),
        hovermode='x unified',
        hoverdistance=100,
        template=PLOTLY_TEMPLATE,
        margin=dict(t=120, l=60, r=40, b=100),
        height=500,
        dragmode='zoom'
    )
    
    if return_json:
        return fig.to_json()
    
    if output_path:
        fig.write_image(output_path, width=1400, height=800, scale=2)
    
    return output_path
def generate_radar_chart(decade, language, output_path=None, return_json=False):
    """Generate Radar Chart using Plotly"""
    if language == 'All':
        languages = ['English', 'Hindi', 'Tamil']
    else:
        languages = [language]
    
    data = {}
    for lang in languages:
        data[lang] = data_utils.get_emotions_by_decade(lang, decade)
    
    normalized = {}
    for lang, vals in data.items():
        total = vals.sum()
        normalized[lang] = (vals / total) * 100
    
    categories = data_utils.EMOTION_LABELS
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + angles[:1]
    
    fig = go.Figure()
    
    for lang in languages:
        values = normalized[lang].values.tolist()
        values_closed = values + values[:1]

        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories + [categories[0]],
            fill='toself',
            name=lang,
            line_color=data_utils.COLOR_PALETTE[lang],
            fillcolor=f"rgba({int(data_utils.COLOR_PALETTE[lang][1:3], 16)}, {int(data_utils.COLOR_PALETTE[lang][3:5], 16)}, {int(data_utils.COLOR_PALETTE[lang][5:7], 16)}, 0.3)",
            hovertemplate='<b>%{fullData.name}</b><br>%{theta}: %{r:.1f}%<extra></extra>',
            hoverlabel=UNIVERSAL_HOVERCONFIG['hoverlabel']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                title='Percentage (%)',
                tickvals=[5, 10, 15, 20],
                ticktext=['5%', '10%', '15%', '20%'],
                range=[0, 20],
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)'
            ),
            angularaxis=dict(
                tickfont=dict(size=11),
                direction='clockwise',
                rotation=90
            )
        ),
        title=dict(
            text=f'Relative Emotional Composition by Culture ({decade}s)',
            font=dict(size=18, color='#1f77b4'),
            x=0.5
        ),
        legend=dict(
            title='Language',
            orientation='h',
            yanchor='bottom',
            y=1.05,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        ),
        hoverdistance=100,
        template=PLOTLY_TEMPLATE,
        margin=dict(t=100, l=60, r=60, b=60),
        height=600,
        width=600
    )
    
    if return_json:
        return fig.to_json()
    
    if output_path:
        fig.write_image(output_path, width=1000, height=1000, scale=2)
    
    return output_path
def generate_correlation_matrix(language, output_path=None, return_json=False):
    """Generate Correlation Matrix heatmap using Plotly"""
    if language == 'All':
        languages = ['English', 'Hindi', 'Tamil']
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=languages,
            horizontal_spacing=0.05
        )
        
        for idx, lang in enumerate(languages):
            corr = data_utils.get_correlation_matrix(lang)

            fig.add_trace(
                go.Heatmap(
                    z=corr.values.tolist(),
                    x=corr.index.tolist(),
                    y=corr.columns.tolist(),
                    colorscale='RdBu_r',
                    zmin=-1, zmax=1,
                    text=np.round(corr.values, 2).tolist(),
                    texttemplate='%{text}',
                    textfont=dict(size=8),
                    hovertemplate='Emotion 1: %{x}<br>Emotion 2: %{y}<br>Correlation: %{z:.3f}<extra></extra>',
                    showscale=idx == 2,
                    hoverlabel=UNIVERSAL_HOVERCONFIG['hoverlabel']
                ),
                row=1, col=idx+1
            )
        
        fig.update_layout(
            title=dict(
                text='Cross-Cultural Emotion Correlation Comparison',
                font=dict(size=18, color='#1f77b4'),
                x=0.5
            ),
            hoverdistance=100,
            template=PLOTLY_TEMPLATE,
            margin=dict(t=80, l=40, r=40, b=60),
            height=400,
            width=1800
        )
    else:
        corr = data_utils.get_correlation_matrix(language)

        fig = go.Figure(data=go.Heatmap(
            z=corr.values.tolist(),
            x=corr.index.tolist(),
            y=corr.columns.tolist(),
            colorscale='RdBu_r',
            zmin=-1, zmax=1,
            text=np.round(corr.values, 2).tolist(),
            texttemplate='%{text}',
            textfont=dict(size=10),
            hovertemplate='Emotion 1: %{x}<br>Emotion 2: %{y}<br>Correlation: %{z:.3f}<extra></extra>',
            colorbar=dict(title='Pearson Correlation'),
            hoverlabel=UNIVERSAL_HOVERCONFIG['hoverlabel']
        ))
        
        fig.update_layout(
            title=dict(
                text=f'{language} Emotion Correlation',
                font=dict(size=18, color='#1f77b4'),
                x=0.5
            ),
            hoverdistance=100,
            template=PLOTLY_TEMPLATE,
            margin=dict(t=80, l=60, r=60, b=60),
            height=600,
            width=800
        )
    
    if return_json:
        return fig.to_json()
    
    if output_path:
        fig.write_image(output_path, width=fig.layout.width, height=fig.layout.height, scale=2)
    
    return output_path
def generate_mirror_hypothesis(emotion, language, output_path=None, decade_start=1810, decade_end=1960, return_json=False):
    """Generate Dual-Axis Mirror Hypothesis chart using Plotly"""
    df = data_utils.get_mirror_hypothesis_data(language, emotion.lower(), decade_start, decade_end)
    
    lit_color = data_utils.COLOR_PALETTE.get(language, '#1f77b4')
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Literature line
    fig.add_trace(
        go.Scatter(
            x=df.index.tolist(),
            y=df['Literature'].tolist(),
            mode='lines+markers',
            name=f'{language} Literature',
            line=dict(color=lit_color, width=3),
            marker=dict(size=10),
            hovertemplate=f'<b>{language} Literature</b><br>' +
                          'Decade: %{x}<br>' +
                          'Intensity: %{y:.4f}<extra></extra>',
            hoverlabel=UNIVERSAL_HOVERCONFIG['hoverlabel']
        ),
        secondary_y=False
    )
    
    # Historical events line
    fig.add_trace(
        go.Scatter(
            x=df.index.tolist(),
            y=df['Historical'].tolist(),
            mode='lines+markers',
            name='Historical Events',
            line=dict(color='#d62728', width=3, dash='dash'),
            marker=dict(size=10, symbol='square'),
            hovertemplate=f'<b>Historical Events</b><br>' +
                          'Decade: %{x}<br>' +
                          'Index: %{y:.4f}<extra></extra>',
            hoverlabel=UNIVERSAL_HOVERCONFIG['hoverlabel']
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title=dict(
            text=f'The {language} Mirror Hypothesis - Literary vs. Historical Trajectory of {emotion.capitalize()} ({decade_start}-{decade_end})',
            font=dict(size=16, color='#1f77b4'),
            x=0.5
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        ),
        hoverdistance=100,
        template=PLOTLY_TEMPLATE,
        margin=dict(t=100, l=60, r=60, b=80),
        height=500,
        dragmode='zoom'
    )
    
    fig.update_yaxes(
        title=f'{language} Literary {emotion.capitalize()} Intensity',
        tickfont=dict(color=lit_color),
        showgrid=True,
        gridcolor='rgba(0,0,0,0.1)',
        secondary_y=False
    )
    
    fig.update_yaxes(
        title=f'Historical {emotion.capitalize()} Index',
        tickfont=dict(color='#d62728'),
        showgrid=False,
        secondary_y=True
    )
    
    fig.update_xaxes(
        title='Time Period (Decade)',
        tickangle=45,
        showgrid=True,
        gridcolor='rgba(0,0,0,0.1)'
    )
    
    if return_json:
        return fig.to_json()
    
    if output_path:
        fig.write_image(output_path, width=1400, height=800, scale=2)
    
    return output_path
def generate_heatmap_timeline(language, output_path=None, decade_start=1810, decade_end=1960, return_json=False):
    """Generate Heatmap Timeline visualization using Plotly"""
    heatmap_data = data_utils.get_heatmap_data(language, decade_start, decade_end)
    
    if language == 'All':
        languages = ['English', 'Hindi', 'Tamil']
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=languages,
            horizontal_spacing=0.05
        )
        
        for idx, lang in enumerate(languages):
            if lang in heatmap_data:
                data = heatmap_data[lang]

                fig.add_trace(
                    go.Heatmap(
                        z=data.values.tolist(),
                        x=data.columns.tolist(),
                        y=data.index.tolist(),
                        colorscale='YlOrRd',
                        text=np.round(data.values, 3).tolist(),
                        texttemplate='%{text:.2f}' if data.shape[0] <= 5 else '%{text}',
                        textfont=dict(size=8),
                        hovertemplate='Decade: %{y}<br>Emotion: %{x}<br>Intensity: %{z:.4f}<extra></extra>',
                        showscale=idx == 2,
                        hoverlabel=UNIVERSAL_HOVERCONFIG['hoverlabel']
                    ),
                    row=1, col=idx+1
                )
        
        fig.update_layout(
            title=dict(
                text=f'Emotion Intensity Heatmap Timeline ({decade_start}-{decade_end})',
                font=dict(size=18, color='#1f77b4'),
                x=0.5
            ),
            hoverdistance=100,
            template=PLOTLY_TEMPLATE,
            margin=dict(t=80, l=40, r=40, b=80),
            height=400,
            width=1800
        )
    else:
        if language in heatmap_data:
            data = heatmap_data[language]
            
            fig = go.Figure(data=go.Heatmap(
                z=data.values.tolist(),
                x=data.columns.tolist(),
                y=data.index.tolist(),
                colorscale='YlOrRd',
                text=np.round(data.values, 3).tolist(),
                texttemplate='%{text:.2f}',
                textfont=dict(size=9),
                hovertemplate='Decade: %{y}<br>Emotion: %{x}<br>Intensity: %{z:.4f}<extra></extra>',
                colorbar=dict(title='Intensity'),
                hoverlabel=UNIVERSAL_HOVERCONFIG['hoverlabel']
            ))
            
            fig.update_layout(
                title=dict(
                    text=f'{language} Literature - Emotion Intensity Heatmap ({decade_start}-{decade_end})',
                    font=dict(size=18, color='#1f77b4'),
                    x=0.5
                ),
                xaxis=dict(
                    title='Decade',
                    tickangle=45,
                    showgrid=True
                ),
                yaxis=dict(
                    title='Emotion',
                    showgrid=True
                ),
                hoverdistance=100,
                template=PLOTLY_TEMPLATE,
                margin=dict(t=80, l=100, r=60, b=80),
                height=500,
                width=1000
            )
        else:
            fig = go.Figure()
            fig.add_annotation(text=f'No data available for {language}', showarrow=False, font=dict(size=16))
            if return_json:
                return fig.to_json()
            if output_path:
                fig.write_image(output_path, width=800, height=400, scale=2)
            return output_path
    
    if return_json:
        return fig.to_json()
    
    if output_path:
        fig.write_image(output_path, width=fig.layout.width, height=fig.layout.height, scale=2)
    
    return output_path
def generate_distribution_plot(language, output_path=None, decade_start=1810, decade_end=1960, return_json=False):
    """Generate Distribution Plot (Box/Violin plot) using Plotly"""
    dist_data = data_utils.get_distribution_data(language, decade_start, decade_end)
    
    if language == 'All':
        languages = ['English', 'Hindi', 'Tamil']
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=languages,
            horizontal_spacing=0.05
        )
        
        for idx, lang in enumerate(languages):
            if lang in dist_data:
                data = dist_data[lang]
                data_melted = data.melt(var_name='Emotion', value_name='Intensity')
                
                for emotion in data_melted['Emotion'].unique():
                    emo_data = data_melted[data_melted['Emotion'] == emotion]

                    fig.add_trace(
                        go.Box(
                            y=emo_data['Intensity'].tolist(),
                            name=emotion,
                            boxmean='sd',
                            marker_color=data_utils.COLOR_PALETTE.get(lang, '#1f77b4'),
                            hovertemplate='<b>%{fullData.name}</b><br>Intensity: %{y:.4f}<extra></extra>',
                            legendgroup=emotion,
                            showlegend=(idx == 0),
                            hoverlabel=UNIVERSAL_HOVERCONFIG['hoverlabel']
                        ),
                        row=1, col=idx+1
                    )
        
        fig.update_layout(
            title=dict(
                text=f'Emotion Distribution by Language ({decade_start}-{decade_end})',
                font=dict(size=18, color='#1f77b4'),
                x=0.5
            ),
            hoverdistance=100,
            template=PLOTLY_TEMPLATE,
            margin=dict(t=80, l=40, r=40, b=60),
            height=450,
            width=1800,
            showlegend=True
        )
    else:
        if language in dist_data:
            data = dist_data[language]
            data_melted = data.melt(var_name='Emotion', value_name='Intensity')
            
            fig = go.Figure()
            
            for emotion in data_melted['Emotion'].unique():
                emo_data = data_melted[data_melted['Emotion'] == emotion]

                fig.add_trace(go.Violin(
                    y=emo_data['Intensity'].tolist(),
                    name=emotion,
                    box_visible=True,
                    meanline_visible=True,
                    points='all',
                    jitter=0.05,
                    marker_color=data_utils.COLOR_PALETTE.get(language, '#1f77b4'),
                    line_color=data_utils.COLOR_PALETTE.get(language, '#1f77b4'),
                    fillcolor=f"rgba({int(data_utils.COLOR_PALETTE.get(language, '#1f77b4')[1:3], 16)}, {int(data_utils.COLOR_PALETTE.get(language, '#1f77b4')[3:5], 16)}, {int(data_utils.COLOR_PALETTE.get(language, '#1f77b4')[5:7], 16)}, 0.5)",
                    hovertemplate='<b>%{fullData.name}</b><br>Intensity: %{y:.4f}<extra></extra>',
                    hoverlabel=UNIVERSAL_HOVERCONFIG['hoverlabel']
                ))
            
            fig.update_layout(
                title=dict(
                    text=f'{language} Literature - Emotion Distribution ({decade_start}-{decade_end})',
                    font=dict(size=18, color='#1f77b4'),
                    x=0.5
                ),
                xaxis=dict(
                    title='Emotion',
                    tickangle=45
                ),
                yaxis=dict(
                    title='Intensity',
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.1)'
                ),
                hoverdistance=100,
                template=PLOTLY_TEMPLATE,
                margin=dict(t=80, l=60, r=60, b=80),
                height=500,
                width=1000,
                violinmode='group'
            )
        else:
            fig = go.Figure()
            fig.add_annotation(text=f'No data available for {language}', showarrow=False, font=dict(size=16))
            if return_json:
                return fig.to_json()
            if output_path:
                fig.write_image(output_path, width=800, height=400, scale=2)
            return output_path
    
    if return_json:
        return fig.to_json()
    
    if output_path:
        fig.write_image(output_path, width=fig.layout.width, height=fig.layout.height, scale=2)
    
    return output_path
def generate_literary_period_comparison(period, output_path=None, return_json=False):
    """Generate Literary Period Comparison chart using Plotly"""
    valid_periods = data_utils.LITERARY_PERIODS
    if period not in valid_periods:
        period = 'Romantic'
    
    period_years = valid_periods.get(period, (1810, 1840))
    period_data = data_utils.get_literary_period_data(period)
    
    if not period_data:
        fig = go.Figure()
        fig.add_annotation(
            text=f'No data available for {period} period',
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=f'Literary Period: {period}',
            template=PLOTLY_TEMPLATE
        )
        if return_json:
            return fig.to_json()
        if output_path:
            fig.write_image(output_path, width=1200, height=800, scale=2)
        return output_path
    
    fig = go.Figure()
    
    x = list(data_utils.EMOTION_LABELS)
    languages_list = list(period_data.keys())
    colors = [data_utils.COLOR_PALETTE.get(lang, '#1f77b4') for lang in languages_list]
    
    for idx, (lang, values) in enumerate(period_data.items()):
        fig.add_trace(go.Bar(
            name=lang,
            x=x,
            y=values.values.tolist(),
            marker_color=colors[idx],
            marker_line_color='rgba(0,0,0,0.2)',
            marker_line_width=1,
            hovertemplate='<b>%{fullData.name}</b><br>%{x}: %{y:.4f}<extra></extra>',
            hoverlabel=UNIVERSAL_HOVERCONFIG['hoverlabel']
        ))
    
    fig.update_layout(
        title=dict(
            text=f'Emotion Composition During {period} Period ({period_years[0]}-{period_years[1]})',
            font=dict(size=18, color='#1f77b4'),
            x=0.5
        ),
        xaxis=dict(
            title='Emotion',
            tickangle=45
        ),
        yaxis=dict(
            title='Average Intensity',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        legend=dict(
            title='Language',
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        ),
        barmode='group',
        hoverdistance=100,
        template=PLOTLY_TEMPLATE,
        margin=dict(t=100, l=60, r=40, b=100),
        height=500,
        dragmode='zoom'
    )
    
    if return_json:
        return fig.to_json()
    
    if output_path:
        fig.write_image(output_path, width=1200, height=800, scale=2)
    
    return output_path
def generate_cross_correlation(emotion1, emotion2, language, output_path=None, max_lag=5, return_json=False):
    """Generate Cross-Correlation visualization using Plotly"""
    try:
        result = data_utils.calculate_cross_correlation(emotion1, emotion2, language, max_lag)
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f'Error calculating cross-correlation:<br>{str(e)}',
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(
            title=f'Cross-Correlation: {emotion1} vs {emotion2}',
            template=PLOTLY_TEMPLATE,
            hoverdistance=100,
        )
        if return_json:
            return fig.to_json()
        if output_path:
            fig.write_image(output_path, width=1000, height=600, scale=2)
        return output_path
    
    if not result or 'error' in result:
        fig = go.Figure()
        fig.add_annotation(
            text='Insufficient data for cross-correlation analysis',
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=f'Cross-Correlation: {emotion1} vs {emotion2}',
            template=PLOTLY_TEMPLATE,
            hoverdistance=100,
        )
        if return_json:
            return fig.to_json()
        if output_path:
            fig.write_image(output_path, width=1000, height=600, scale=2)
        return output_path
    
    lags = list(range(-max_lag, max_lag + 1))
    correlations = []
    for lag in lags:
        val = result['correlations'].get(f'lag_{lag}', 0)
        correlations.append(val if val is not None else 0)
    
    colors = ['#28a745' if c > 0 else '#dc3545' for c in correlations]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=lags,
        y=correlations,
        marker_color=colors,
        marker_line_color='rgba(0,0,0,0.3)',
        marker_line_width=1,
        hovertemplate='<b>Lag %{x}</b><br>Correlation: %{y:.4f}<extra></extra>',
        hoverlabel=UNIVERSAL_HOVERCONFIG['hoverlabel']
    ))
    
    # Add reference lines
    fig.add_hline(y=0, line_dash='solid', line_color='black', line_width=1)
    fig.add_hline(y=0.5, line_dash='dash', line_color='gray', line_width=1, annotation_text='Strong positive')
    fig.add_hline(y=-0.5, line_dash='dash', line_color='gray', line_width=1, annotation_text='Strong negative')
    
    fig.update_layout(
        title=dict(
            text=f'Cross-Correlation: {emotion1} vs {emotion2} in {language} Literature',
            font=dict(size=18, color='#1f77b4'),
            x=0.5
        ),
        xaxis=dict(
            title='Lag (Decades)',
            tickvals=lags,
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title='Correlation Coefficient',
            range=[-1.1, 1.1],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.3)'
        ),
        hoverdistance=100,
        template=PLOTLY_TEMPLATE,
        margin=dict(t=80, l=60, r=60, b=80),
        height=500,
        dragmode='zoom'
    )
    
    if return_json:
        return fig.to_json()
    
    if output_path:
        fig.write_image(output_path, width=1000, height=600, scale=2)
    
    return output_path