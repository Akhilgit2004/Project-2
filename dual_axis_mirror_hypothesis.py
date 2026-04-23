# DUAL-AXIS REFLECTION TIME SERIES FOR IEEE ACADEMIC PAPER
# Tests 'Mirror vs. Escapism' hypothesis - Literature vs. Historical events
# ===================== EDIT THIS CELL =====================
# Target emotion (lowercase to match historical CSV)
target_emotion = 'joy'
# Map to literature CSV column (capitalization may differ)
literature_emotion_col = 'joy'
# Figure number for publication
FIG_NUMBER = 'X'
# ============================================================

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Valid decades for time series
VALID_DECADES = [1810, 1820, 1830, 1840, 1850, 1860, 1870, 1880, 1890, 1900, 1910, 1920, 1930, 1940, 1950, 1960]

# Color scheme
LIT_COLOR = '#1f77b4'  # Dark Blue
HIST_COLOR = '#d62728'  # Dark Red

print("="*60)
print(f"DUAL-AXIS MIRROR HYPOTHESIS: {target_emotion.upper()}")
print("="*60)
print("Make a choice between Hindi,Tamil,English (h,t,e) ")
ch=input("")
# ===================== LITERATURE DATA =====================
# Load literature emotions
if ch=="e":
    print("chosen english.")
    lit_df = pd.read_csv('final_english_emotions.csv')
    hist_df = pd.read_csv('historical_events_english.csv')
    output_filename = 'dual_axis_mirror_hypothesis_joy_E.png'

elif ch=="h":
    lit_df = pd.read_csv('final_hindi_emotions.csv')
    hist_df = pd.read_csv('historical_events_hindi.csv')
    output_filename = 'dual_axis_mirror_hypothesis_joy_H.png'

elif ch=="t":
    lit_df = pd.read_csv('final_tamil_emotions.csv')
    hist_df = pd.read_csv('historical_events_tamil.csv')
    output_filename = 'dual_axis_mirror_hypothesis_joy_T.png'
# Determine time_period column
time_col = 'time_period' if 'time_period' in lit_df.columns else 'decade'

# Normalize decades
def normalize_decade(val):
    if pd.isna(val):
        return None
    val_str = str(val).strip()
    if len(val_str) >= 4 and val_str[:4].isdigit():
        return int(val_str[:4])
    return None

lit_df[time_col] = lit_df[time_col].apply(normalize_decade)
lit_df = lit_df.dropna(subset=[time_col])
lit_df[time_col] = lit_df[time_col].astype(int)

# Filter to valid decades
lit_df = lit_df[lit_df[time_col].isin(VALID_DECADES)]

# Group by decade and calculate mean
lit_grouped = lit_df.groupby(time_col)[literature_emotion_col].mean()

# Rename index to decade
lit_grouped.index.name = 'decade'

# Apply 3-period rolling average (center=True)
lit_smoothed = lit_grouped.rolling(window=3, min_periods=1, center=True).mean()

print(f"Literature data points: {len(lit_smoothed)}")

# ===================== HISTORICAL DATA =====================
# Load historical events


# Group by decade and calculate mean (in case of multiple events per decade)
hist_grouped = hist_df.groupby('decade')[target_emotion].mean()

# Apply 3-period rolling average (center=True)
hist_smoothed = hist_grouped.rolling(window=3, min_periods=1, center=True).mean()

print(f"Historical data points: {len(hist_smoothed)}")

# ===================== MERGE DATA =====================
# Combine into single dataframe
merged_df = pd.DataFrame({
    'Literature': lit_smoothed,
    'Historical': hist_smoothed
})

# Drop rows with NaN
merged_df = merged_df.dropna()

print(f"\nMerged data points: {len(merged_df)}")
print("\n--- Merged Data ---")
print(merged_df.round(4))

# ===================== PLOTTING =====================
# Set seaborn style
sns.set_style('whitegrid')

# Initialize plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# ========== Axis 1 (Literature) ==========
ax1.set_xlabel('Time Period (Decade)', fontsize=12, fontweight='bold')

# Plot literature data
line1, = ax1.plot(merged_df.index, merged_df['Literature'], 
                   color=LIT_COLOR, linewidth=2.5, linestyle='-',
                   marker='o', markersize=8, label='Literature')

# Y-axis label
ax1.set_ylabel(f'Literary {target_emotion.capitalize()} Intensity', 
               color=LIT_COLOR, fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=LIT_COLOR)
ax1.tick_params(axis='y', labelsize=10)

# ========== Axis 2 (History) ==========
ax2 = ax1.twinx()

# Plot historical data
line2, = ax2.plot(merged_df.index, merged_df['Historical'], 
                  color=HIST_COLOR, linewidth=2.5, linestyle='--',
                  marker='s', markersize=8, label='Historical Events')

# Y-axis label
ax2.set_ylabel(f'Historical {target_emotion.capitalize()} Index', 
              color=HIST_COLOR, fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=HIST_COLOR)
ax2.tick_params(axis='y', labelsize=10)

# ========== Academic Aesthetics ==========
# Master title
ax1.set_title(f'Fig {FIG_NUMBER}: The Mirror Hypothesis - Literary vs. Historical Trajectory of {target_emotion.capitalize()} (1810-1960)', 
              fontsize=14, fontweight='bold', pad=20)

# X-axis ticks
ax1.set_xticks(merged_df.index)
ax1.set_xticklabels(merged_df.index, rotation=45, ha='right')

# Legend (combined)
lines = [line1, line2]
labels = ['Literature', 'Historical Events']
ax1.legend(lines, labels, loc='upper right', fontsize=10, frameon=True, fancybox=True, shadow=True)

# Grid
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_axisbelow(True)

# Set y-axis limits
y_lit_max = merged_df['Literature'].max() * 1.15
y_hist_max = merged_df['Historical'].max() * 1.15
ax1.set_ylim(0, y_lit_max)
ax2.set_ylim(0, y_hist_max)

# Tight layout
plt.tight_layout()

# Save as high-resolution PNG (300 DPI)

plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"\nGraph saved as: {output_filename}")
print(f"Resolution: 300 DPI (publication-ready)")

# Display
# plt.show()