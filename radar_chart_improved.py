# CROSS-CULTURAL RADAR CHART FOR IEEE ACADEMIC PAPER
# Publication-ready spider chart with proportional normalization
# ===================== EDIT THIS CELL =====================
# Target decade (change here to swap: 1920, 1850, etc.)
target_decade = 1940
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

# Vibrant, distinct palette
COLOR_PALETTE = {
    'English': '#1f77b4',   # Vibrant blue
    'Hindi': '#d62728',    # Vibrant red
    'Tamil': '#2ca02c'     # Vibrant green
}

print("="*60)
print(f"IEEE PUBLICATION-READY RADAR CHART: {target_decade}s")
print("="*60)

# ===================== DATA LOADING =====================
# Load all three emotion datasets
english_emotions = pd.read_csv('final_english_emotions.csv')
hindi_emotions = pd.read_csv('final_hindi_emotions.csv')
tamil_emotions = pd.read_csv('final_tamil_emotions.csv')

# Add Language column to identify source
english_emotions['Language'] = 'English'
hindi_emotions['Language'] = 'Hindi'
tamil_emotions['Language'] = 'Tamil'

# Combine into single master dataframe
master_df = pd.concat([english_emotions, hindi_emotions, tamil_emotions], ignore_index=True)

# Determine time_period column name
time_col = 'time_period' if 'time_period' in master_df.columns else 'decade'

# Normalize time_period values to decade integer
def normalize_decade(val):
    if pd.isna(val):
        return None
    val_str = str(val).strip()
    if len(val_str) >= 4 and val_str[:4].isdigit():
        return int(val_str[:4])
    return None

master_df[time_col] = master_df[time_col].apply(normalize_decade)
master_df = master_df.dropna(subset=[time_col])
master_df[time_col] = master_df[time_col].astype(int)

# Filter to target decade
decade_df = master_df[master_df[time_col] == target_decade].copy()

# Group by Language and calculate mean for all emotion columns
emotion_by_lang = decade_df.groupby('Language')[EMOTION_COLUMNS].mean()

# ===================== PROPORTIONAL NORMALIZATION =====================
# For each language, divide each emotion by the sum of all 11 emotions
# This converts raw scores to percentages (all languages on same scale)
emotion_sums = emotion_by_lang.sum(axis=1)
normalized_df = emotion_by_lang.div(emotion_sums, axis=0) * 100

print(f"Documents in {target_decade}s: {len(decade_df)}")
print(f"  - English: {len(decade_df[decade_df['Language'] == 'English'])}")
print(f"  - Hindi: {len(decade_df[decade_df['Language'] == 'Hindi'])}")
print(f"  - Tamil: {len(decade_df[decade_df['Language'] == 'Tamil'])}")
print(f"\nNormalized emotion percentages for {target_decade}s:")
print(normalized_df.round(2).T)

# ===================== RADAR MATH =====================
# Extract emotion categories as labels
categories = EMOTION_LABELS
N = len(categories)

# Calculate angle for each axis
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

# Append the first value to close the polygon
angles_closed = angles + angles[:1]

# Get normalized values for each language and close polygons
english_values = normalized_df.loc['English'].values.tolist()
english_values_closed = english_values + english_values[:1]

hindi_values = normalized_df.loc['Hindi'].values.tolist()
hindi_values_closed = hindi_values + hindi_values[:1]

tamil_values = normalized_df.loc['Tamil'].values.tolist()
tamil_values_closed = tamil_values + tamil_values[:1]

# ===================== HIGH-END IEEE AESTHETICS =====================
# Set seaborn style
sns.set_style('whitegrid')

# Initialize figure with polar projection (larger for clarity)
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# Plot each language with improved styling
for language, values, color in [
    ('English', english_values_closed, COLOR_PALETTE['English']),
    ('Hindi', hindi_values_closed, COLOR_PALETTE['Hindi']),
    ('Tamil', tamil_values_closed, COLOR_PALETTE['Tamil'])
]:
    # Plot the line with thick weight
    ax.plot(angles_closed, values, color=color, linewidth=2.5, label=language, marker='o', markersize=6)
    # Fill with very light transparency to prevent muddy overlap
    ax.fill(angles_closed, values, color=color, alpha=0.1)

# Hide concentric Y-axis numbers to clean up the center
ax.set_yticklabels([])

# Push emotion labels further out with padding
ax.set_thetagrids(np.degrees(angles), categories, fontsize=12, fontweight='bold')
ax.tick_params(pad=20)

# Set title with proper positioning
ax.set_title(
    f'Fig {FIG_NUMBER}: Relative Emotional Composition by Culture ({target_decade}s)',
    fontsize=14, fontweight='bold', pad=30
)

# Place legend completely outside the plot
ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), fontsize=10, title='Language', title_fontsize=11)

# Set radial limits (0-20% for proportional data)
ax.set_ylim(0, 20)

# Use radial grid ticks
ax.set_rticks([5, 10, 15, 20])
ax.set_yticklabels(['5%', '10%', '15%', '20%'], fontsize=9)

# Tight layout
plt.tight_layout()

# Save as high-resolution PNG (300 DPI)
output_filename = f'radar_chart_{target_decade}s.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"\nGraph saved as: {output_filename}")
print(f"Resolution: 300 DPI (publication-ready)")

# Display
plt.show()