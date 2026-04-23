# CROSS-CULTURAL EMOTION CORRELATION MATRIX FOR IEEE ACADEMIC PAPER
# Publication-ready 1x3 comparative heatmap
# ===================== EDIT THIS CELL =====================
# Figure number for the publication
FIG_NUMBER = 'X'
# ============================================================

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Define the 11 emotion categories (lowercase for column matching)
EMOTION_COLUMNS = ['anger', 'contempt', 'disgust', 'fear', 'frustration',
                  'gratitude', 'joy', 'love', 'neutral', 'sadness', 'surprise']

# Display labels (capitalized for academic presentation)
EMOTION_LABELS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Frustration',
                 'Gratitude', 'Joy', 'Love', 'Neutral', 'Sadness', 'Surprise']

print("="*60)
print("IEEE CROSS-CULTURAL CORRELATION MATRIX")
print("="*60)

# ===================== DATA LOADING =====================
# Load all three emotion datasets
df_eng = pd.read_csv('final_english_emotions.csv')
df_hin = pd.read_csv('final_hindi_emotions.csv')
df_tam = pd.read_csv('final_tamil_emotions.csv')

# Extract just the emotion columns
eng_emotions = df_eng[EMOTION_COLUMNS]
hin_emotions = df_hin[EMOTION_COLUMNS]
tam_emotions = df_tam[EMOTION_COLUMNS]

print(f"English documents: {len(df_eng)}")
print(f"Hindi documents: {len(df_hin)}")
print(f"Tamil documents: {len(df_tam)}")

# Rename columns to display labels
eng_emotions.columns = EMOTION_LABELS
hin_emotions.columns = EMOTION_LABELS
tam_emotions.columns = EMOTION_LABELS

# ===================== CORRELATION MATRICES =====================
# Calculate Pearson correlation matrices
corr_eng = eng_emotions.corr(method='pearson')
corr_hin = hin_emotions.corr(method='pearson')
corr_tam = tam_emotions.corr(method='pearson')

print("\n--- English Correlation Matrix ---")
print(corr_eng.round(2))

print("\n--- Hindi Correlation Matrix ---")
print(corr_hin.round(2))

print("\n--- Tamil Correlation Matrix ---")
print(corr_tam.round(2))

# ===================== TRIANGLE MASK =====================
# Generate mask for upper triangle (to avoid redundant data)
mask = np.triu(np.ones_like(corr_eng, dtype=bool))

# ===================== PLOTTING =====================
# Set seaborn style
sns.set_style('whitegrid')

# Initialize wide figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# Diverging colormap (negative=blue, positive=red, zero=white)
cmap = 'RdBu_r'

# Common heatmap parameters
heatmap_kwargs = {
    'mask': mask,
    'cmap': cmap,
    'vmin': -1,
    'vmax': 1,
    'annot': True,
    'fmt': '.2f',
    'square': True,
    'linewidths': 0.5,
    'cbar_kws': {'shrink': 0.8, 'label': 'Pearson Correlation'}
}

# Plot English correlation heatmap
sns.heatmap(corr_eng, ax=axes[0], **heatmap_kwargs)
axes[0].set_title('English Emotion Correlation', fontsize=14, fontweight='bold', pad=15)

# Plot Hindi correlation heatmap
sns.heatmap(corr_hin, ax=axes[1], **heatmap_kwargs)
axes[1].set_title('Hindi Emotion Correlation', fontsize=14, fontweight='bold', pad=15)

# Plot Tamil correlation heatmap
sns.heatmap(corr_tam, ax=axes[2], **heatmap_kwargs)
axes[2].set_title('Tamil Emotion Correlation', fontsize=14, fontweight='bold', pad=15)

# Add master title
fig.suptitle(f'Fig {FIG_NUMBER}: Comparative Psychological Structures in Cross-Cultural Literature', 
             fontsize=20, fontweight='bold', y=1.02)

# Adjust layout
plt.tight_layout()

# Save as high-resolution PNG (300 DPI)
output_filename = 'cross_cultural_correlation_matrix.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"\nGraph saved as: {output_filename}")
print(f"Resolution: 300 DPI (publication-ready)")

# Display
# plt.show()