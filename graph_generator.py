import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import data_utils

# Set style globally
sns.set_style('whitegrid')

def generate_time_series(emotion, language, output_path):
    """Generate the Time Series with Rolling Average chart"""
    # Get data - determine languages to include
    if language == 'All':
        languages = ['English', 'Hindi', 'Tamil']
    else:
        languages = [language]
    
    # Get data
    df = data_utils.get_time_series_multilang(languages, emotion.lower())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    
    # Plot each language
    for lang in languages:
        if lang in df.columns:
            lang_data = df[lang].dropna()
            ax.plot(lang_data.index, lang_data.values,
                   marker=data_utils.MARKERS[lang],
                   color=data_utils.COLOR_PALETTE[lang],
                   linewidth=2.5,
                   markersize=8,
                   label=lang,
                   alpha=0.9)
    
    # Set labels and title
    ax.set_xlabel('Time Period (Decade)', fontsize=12, fontweight='bold')
    ax.set_ylabel('3-Decade Rolling Mean Intensity', fontsize=12, fontweight='bold')
    ax.set_title(f'Fig X: Smoothed Cross-Cultural Trajectory of {emotion.capitalize()} in Literature (1810–1960)',
                 fontsize=14, fontweight='bold', pad=20)
    
    # X-axis ticks
    ax.set_xticks(data_utils.VALID_DECADES)
    ax.set_xticklabels(data_utils.VALID_DECADES, rotation=45, ha='right')
    
    # Legend
    if len(languages) > 1:
        ax.legend(title='Language', loc='upper right', fontsize=10, title_fontsize=11,
                frameon=True, fancybox=True, shadow=True)
    else:
        ax.legend(title='Language', loc='upper right', fontsize=10,
                frameon=True, fancybox=True, shadow=True)
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Y-axis limits
    y_max = df.max().max() * 1.15
    ax.set_ylim(0, y_max)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    return output_path

def generate_radar_chart(decade, language, output_path):
    """Generate the Radar Chart (emotional footprint)"""
    # Determine languages to include
    if language == 'All':
        languages = ['English', 'Hindi', 'Tamil']
    else:
        languages = [language]
    
    # Get data for each language
    data = {}
    for lang in languages:
        data[lang] = data_utils.get_emotions_by_decade(lang, decade)
    
    # Proportional normalization
    normalized = {}
    for lang, vals in data.items():
        total = vals.sum()
        normalized[lang] = (vals / total) * 100
    
    # Radar math
    categories = data_utils.EMOTION_LABELS
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + angles[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Plot each language
    for lang in languages:
        values = normalized[lang].values.tolist()
        values_closed = values + values[:1]
        
        ax.plot(angles_closed, values_closed, 
               color=data_utils.COLOR_PALETTE[lang], 
               linewidth=2.5, 
               label=lang, 
               marker='o', 
               markersize=6)
        ax.fill(angles_closed, values_closed, 
               color=data_utils.COLOR_PALETTE[lang], 
               alpha=0.1)
    
    # Clean up center
    ax.set_yticklabels([])
    ax.set_thetagrids(np.degrees(angles), categories, fontsize=12, fontweight='bold')
    ax.tick_params(pad=20)
    
    # Title
    ax.set_title(f'Fig X: Relative Emotional Composition by Culture ({decade}s)',
                fontsize=14, fontweight='bold', pad=30)
    
    # Legend
    if len(languages) > 1:
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), fontsize=10, 
                 title='Language', title_fontsize=11)
    else:
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), fontsize=10)
    
    # Radial limits
    ax.set_ylim(0, 20)
    ax.set_rticks([5, 10, 15, 20])
    ax.set_yticklabels(['5%', '10%', '15%', '20%'], fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    return output_path

def generate_correlation_matrix(language, output_path):
    """Generate the Correlation Matrix heatmap"""
    # Get correlation matrix
    corr = data_utils.get_correlation_matrix(language)
    
    # Mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # Heatmap
    sns.heatmap(corr, 
                mask=mask,
                cmap='RdBu_r',
                vmin=-1, vmax=1,
                annot=True,
                fmt='.2f',
                square=True,
                linewidths=0.5,
                cbar_kws={'shrink': 0.8, 'label': 'Pearson Correlation'},
                ax=ax)
    
    ax.set_title(f'{language} Emotion Correlation', fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    return output_path

def generate_mirror_hypothesis(emotion, language, output_path):
    """Generate the Dual-Axis Mirror Hypothesis chart"""
    # Get data with specified language
    df = data_utils.get_mirror_hypothesis_data(language, emotion.lower())
    
    # Get the color for the selected language
    lit_color = data_utils.COLOR_PALETTE.get(language, '#1f77b4')
    
    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=300)
    
    # Axis 1 (Literature) - left Y-axis
    ax1.set_xlabel('Time Period (Decade)', fontsize=12, fontweight='bold')
    
    line1, = ax1.plot(df.index, df['Literature'], 
                     color=lit_color, linewidth=2.5, linestyle='-',
                     marker='o', markersize=8, label=f'{language} Literature')
    
    ax1.set_ylabel(f'{language} Literary {emotion.capitalize()} Intensity', 
                   color=lit_color, fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=lit_color)
    
    # Axis 2 (History) - right Y-axis
    ax2 = ax1.twinx()
    
    line2, = ax2.plot(df.index, df['Historical'], 
                     color='#d62728', linewidth=2.5, linestyle='--',
                     marker='s', markersize=8, label='Historical Events')
    
    ax2.set_ylabel(f'Historical {emotion.capitalize()} Index', 
                   color='#d62728', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    
    # Title
    ax1.set_title(f'Fig X: The {language} Mirror Hypothesis - Literary vs. Historical Trajectory of {emotion.capitalize()} (1810-1960)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # X-axis
    ax1.set_xticks(df.index)
    ax1.set_xticklabels(df.index, rotation=45, ha='right')
    
    # Legend
    lines = [line1, line2]
    labels = [f'{language} Literature', 'Historical Events']
    ax1.legend(lines, labels, loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    # Grid
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    return output_path