import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Data
# Assuming your file is named 'scouting_data.csv'
df = pd.read_csv('bethelGameScoutingReport.csv')

# 2. Descriptive Statistics: Special Teams Play Frequencies
# Counts how many times they ran each specific play type
play_counts = df['Play type'].value_counts()
print("--- Frequency of Special Teams Plays ---")
print(play_counts)
print("\n")

# 3. Cross-Tab: Play Type vs. Result
# Maps out exactly what happens when they execute specific kicks/punts
xtab_results = pd.crosstab(df['Play type'], df['Result'])
print("--- Play Type vs. Result Matrix ---")
print(xtab_results)

# 4. Visualization: Heatmap
# Creates a color-coded visual of the cross-tab so tendencies pop out instantly
plt.figure(figsize=(10, 6))
sns.heatmap(xtab_results, annot=True, cmap='Reds', fmt='g', cbar=False)

# Formatting the chart for the coach
plt.title('Scouting Report: Special Teams Results Tendency', fontsize=14, fontweight='bold')
plt.ylabel('Play Type (Kick/Punt)', fontsize=12)
plt.xlabel('Play Result', fontsize=12)
plt.tight_layout()

# Display the chart
plt.show()