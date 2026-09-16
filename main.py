import pandas as pd
import numpy as np
from pathlib import Path

# Helper function to format tables cleanly as percentages (e.g., "84%")
def to_pct(data):
    return (data * 100).fillna(0).astype(int).astype(str) + '%'

def write_to_excel(writer, sheet_name, table, title, start_row):
    """Helper function to write a title and a table, returning the next open row."""
    # Write the title
    pd.DataFrame([title]).to_excel(writer, sheet_name=sheet_name, startrow=start_row, startcol=0, header=False, index=False)
    # Write the actual table
    table.to_excel(writer, sheet_name=sheet_name, startrow=start_row + 1)
    # Return the row number for the next table (adds padding)
    return start_row + len(table) + 4

# ==========================================
# 1. LOAD AND PREP DATA
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / 'bethelGameScoutingReport.csv'

def main() -> None:
    df = pd.read_csv(CSV_PATH)

    # Fill text blanks with 'Unknown' to protect Special Teams plays
    text_cols = ['Play type', 'Off form', 'Backfield', 'Play dir', 'Eff', 'Result', 'Hash', 'Qtr', 'Def Front', 'Coverage']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str).str.strip().str.title()

    df['ODK'] = df['ODK'].fillna('Unknown').astype(str).str.strip().str.upper()

    # Map Efficiency from y/n to Success/Not Success (also makes everything lowercase)
    df['Eff'] = df['Eff'].str.lower().replace({'y': 'Success', 'n': 'Not Success'})

    # Handle numeric columns - using 'Int64' removes the .0 from Downs
    df['Dn'] = pd.to_numeric(df['Dn'], errors='coerce').astype('Int64')
    df['Dist'] = pd.to_numeric(df['Dist'], errors='coerce')
    df['Yard ln'] = pd.to_numeric(df['Yard ln'], errors='coerce')
    df['gn/ls'] = pd.to_numeric(df['gn/ls'], errors='coerce').fillna(0)

    # Filter out timeouts
    df = df[df['Play type'] != 'Timeout']

    # Global Distance Buckets
    dist_bins = [0, 3, 7, 100]
    dist_labels = ['Short (1-3)', 'Medium (4-7)', 'Long (8+)']
    df['Dist Bucket'] = pd.cut(df['Dist'], bins=dist_bins, labels=dist_labels)

    # ==========================================
    # ==========================================
    #                OFFENSE
    # ==========================================
    # ==========================================
    print("\n" + "="*40)
    print("           OFFENSIVE ANALYSIS")
    print("="*40 + "\n")

    # FIXED: Now catches uppercase Unknown, lowercase unknown, and empty strings
    offense_df = df[df['ODK'] == 'O'].copy().replace(['Unknown', 'unknown', ''], np.nan)

    # Create Field Zones
    zone_bins = [-100, -20, 0, 20, 100]
    zone_labels = ['Backed Up (Own 1-20)', 'Own 21 - Midfield', 'Opponent 40 - RedZone', 'RedZone (Inside 20)']
    offense_df['Field Zone'] = pd.cut(offense_df['Yard ln'], bins=zone_bins, labels=zone_labels)

    # Filter Frequent Formations (Min 5 Plays)
    form_counts = offense_df['Off form'].value_counts()
    freq_offense_df = offense_df[offense_df['Off form'].isin(form_counts[form_counts >= 5].index)]

    print("--- 1. Play Type Distribution ---")
    print(to_pct(offense_df['Play type'].value_counts(normalize=True)))
    print("\n")

    print("--- 2. Down vs. Play Type ---")
    print(to_pct(pd.crosstab(offense_df['Dn'], offense_df['Play type'], normalize='index')))
    print("\n")

    print("--- 3. Down & Distance vs. Play Type ---")
    print(to_pct(pd.crosstab([offense_df['Dn'], offense_df['Dist Bucket']], offense_df['Play type'], normalize='index')))
    print("\n")

    print("--- 4. Formation vs. Play Type (Min 5 Plays) ---")
    print(to_pct(pd.crosstab(freq_offense_df['Off form'], freq_offense_df['Play type'], normalize='index')))
    print("\n")

    print("--- 5. Field Position vs. Play Type ---")
    print(to_pct(pd.crosstab(offense_df['Field Zone'], offense_df['Play type'], normalize='index')))
    print("\n")

    # --- OFFENSE EFFICIENCY ---
    is_pass = offense_df['Play type'].str.contains('Pass|Dropback|Screen|Shot', case=False, na=False)
    is_run = offense_df['Play type'].str.contains('Run|Zone|Power|Gap|Counter', case=False, na=False)
    pass_plays = offense_df[is_pass]
    run_plays = offense_df[is_run]

    ypc = run_plays['gn/ls'].mean() if len(run_plays) > 0 else 0
    ypa = pass_plays['gn/ls'].mean() if len(pass_plays) > 0 else 0
    completed_passes = pass_plays[pass_plays['Result'].str.contains('Complete|Caught|Good', case=False, na=False)]
    comp_pct = (len(completed_passes) / len(pass_plays) * 100) if len(pass_plays) > 0 else 0

    print("--- 6. Key Efficiency Metrics ---")
    # 1. Calculate Scrambles
    scrambles = pass_plays[pass_plays['Result'].str.contains('Scramble', case=False, na=False)]
    scramble_rate = (len(scrambles) / len(pass_plays) * 100) if len(pass_plays) > 0 else 0

    # 2. Print Metrics
    print(f"Yards Per Carry (YPC):      {ypc:.1f} yards")
    print(f"Yards Per Attempt (YPA):    {ypa:.1f} yards")
    print(f"Completion Percentage:      {int(comp_pct)}%")
    print(f"Scramble Rate on Pass Calls:{int(scramble_rate)}%")
    print("\n")
    print("--- 7. Overall Success Rate ---")
    print(to_pct(offense_df['Eff'].value_counts(normalize=True, dropna=True)))
    print("\n")

    print("--- 8. Efficiency by Formation (Min 5 Plays) ---")
    formation_eff = pd.crosstab(freq_offense_df['Off form'], freq_offense_df['Eff'], normalize='index')
    if 'Success' in formation_eff.columns:
        success_table = formation_eff[['Success']].copy().sort_values(by='Success', ascending=False)
        success_table.columns = ['Success Rate']
        print(to_pct(success_table))
    print("\n")

    # --- OFFENSE EXPLOSIVES ---
    explosive_runs = run_plays[run_plays['gn/ls'] >= 12]
    explosive_passes = pass_plays[pass_plays['gn/ls'] >= 16]
    all_explosive = pd.concat([explosive_runs, explosive_passes])
    total_runs, total_passes = len(run_plays), len(pass_plays)
    exp_run_pct = (len(explosive_runs) / total_runs * 100) if total_runs > 0 else 0
    exp_pass_pct = (len(explosive_passes) / total_passes * 100) if total_passes > 0 else 0

    print("--- 9. Explosive Play Rates ---")
    print(f"Explosive Run Rate (12+ Yds):   {int(exp_run_pct)}%  ({len(explosive_runs)} out of {total_runs} runs)")
    print(f"Explosive Pass Rate (16+ Yds):  {int(exp_pass_pct)}%  ({len(explosive_passes)} out of {total_passes} passes)")
    print("\n")


    # ==========================================
    # ==========================================
    #                DEFENSE
    # ==========================================
    # ==========================================
    print("\n" + "="*40)
    print("           DEFENSIVE ANALYSIS")
    print("="*40 + "\n")

    # FIXED: Now catches uppercase Unknown, lowercase unknown, and empty strings
    defense_df = df[df['ODK'] == 'D'].copy().replace(['Unknown', 'unknown', ''], np.nan)

    # Yards Allowed Buckets
    yards_bins = [-100, -1, 3, 7, 14, 100]
    yards_labels = ['Loss (<0)', 'Stuffed (0-3)', 'Medium (4-7)', 'Long (8-14)', 'Explosive (15+)']
    defense_df['Yards Result'] = pd.cut(defense_df['gn/ls'], bins=yards_bins, labels=yards_labels)

    # Filter Frequent Formations for Defense (Min 7 Plays)
    def_form_counts = defense_df['Off form'].value_counts()
    freq_defense_df = defense_df[defense_df['Off form'].isin(def_form_counts[def_form_counts >= 7].index)]

    print("--- 1. Defensive Front Distribution ---")
    print(to_pct(defense_df['Def Front'].value_counts(normalize=True)))
    print("\n")

    print("--- 2. Coverage Distribution ---")
    print(to_pct(defense_df['Coverage'].value_counts(normalize=True)))
    print("\n")

    print("--- 3. Down & Distance vs. Coverage ---")
    print(to_pct(pd.crosstab([defense_df['Dn'], defense_df['Dist Bucket']], defense_df['Coverage'], normalize='index')))
    print("\n")

    print("--- 4. Down & Distance vs. Def Front ---")
    print(to_pct(pd.crosstab([defense_df['Dn'], defense_df['Dist Bucket']], defense_df['Def Front'], normalize='index')))
    print("\n")

    print("--- 5. Off Formation vs. Def Front (Min 7 Plays) ---")
    print(to_pct(pd.crosstab(freq_defense_df['Off form'], freq_defense_df['Def Front'], normalize='index')))
    print("\n")

    print("--- 6. Off Formation vs. Coverage (Min 7 Plays, Max Coverage > 20%) ---")
    formation_coverage = pd.crosstab(freq_defense_df['Off form'], freq_defense_df['Coverage'], normalize='index')
    formation_coverage = formation_coverage[formation_coverage.max(axis=1) > 0.20]
    print(to_pct(formation_coverage))
    print("\n")

    # --- DEFENSE EFFICIENCY ---
    print("--- 7. Overall Defensive Efficiency (Allowed) ---")
    print(to_pct(defense_df['Eff'].value_counts(normalize=True, dropna=True)))
    print("\n")

    print("--- 8. Def Front vs. Efficiency (Allowed) ---")
    print(to_pct(pd.crosstab(defense_df['Def Front'], defense_df['Eff'], normalize='index')))
    print("\n")

    print("--- 9. Coverage vs. Efficiency (Allowed) ---")
    print(to_pct(pd.crosstab(defense_df['Coverage'], defense_df['Eff'], normalize='index')))
    print("\n")

    print("--- 10. Def Front vs. Yards Allowed ---")
    print(to_pct(pd.crosstab(defense_df['Def Front'], defense_df['Yards Result'], normalize='index')))
    print("\n")

    print("--- 11. Coverage vs. Yards Allowed ---")
    print(to_pct(pd.crosstab(defense_df['Coverage'], defense_df['Yards Result'], normalize='index')))
    print("\n")

    # ==========================================
    # 14. EXPORT TO EXCEL (FULL REPORT)
    # ==========================================
    print("Exporting full data to Excel...")

    with pd.ExcelWriter('Scouting_Report.xlsx', engine='xlsxwriter') as writer:
        
        # ==========================================
        # --- OFFENSE TAB ---
        # ==========================================
        o_row = 0
        
        # 1-5. Distributions & Tendencies
        o_row = write_to_excel(writer, 'Offense_Stats', to_pct(offense_df['Play type'].value_counts(normalize=True).to_frame('Percentage')), '--- 1. Play Type Distribution ---', o_row)
        o_row = write_to_excel(writer, 'Offense_Stats', to_pct(pd.crosstab(offense_df['Dn'], offense_df['Play type'], normalize='index')), '--- 2. Down vs. Play Type ---', o_row)
        o_row = write_to_excel(writer, 'Offense_Stats', to_pct(pd.crosstab([offense_df['Dn'], offense_df['Dist Bucket']], offense_df['Play type'], normalize='index')), '--- 3. Down & Distance vs. Play Type ---', o_row)
        o_row = write_to_excel(writer, 'Offense_Stats', to_pct(pd.crosstab(freq_offense_df['Off form'], freq_offense_df['Play type'], normalize='index')), '--- 4. Formation vs. Play Type (Min 5 Plays) ---', o_row)
        o_row = write_to_excel(writer, 'Offense_Stats', to_pct(pd.crosstab(offense_df['Field Zone'], offense_df['Play type'], normalize='index')), '--- 5. Field Position vs. Play Type ---', o_row)
        
        # 6. Key Efficiency (Converted to DataFrame for Excel)
        eff_data = {
            'Metric': ['Yards Per Carry (YPC)', 'Yards Per Attempt (YPA)', 'Completion %', 'Scramble Rate %'],
            'Value': [f"{ypc:.1f}", f"{ypa:.1f}", f"{int(comp_pct)}%", f"{int(scramble_rate)}%"]
        }
        o_row = write_to_excel(writer, 'Offense_Stats', pd.DataFrame(eff_data).set_index('Metric'), '--- 6. Key Efficiency Metrics ---', o_row)
        
        # 7-8. Success Rates
        o_row = write_to_excel(writer, 'Offense_Stats', to_pct(offense_df['Eff'].value_counts(normalize=True, dropna=True).to_frame('Percentage')), '--- 7. Overall Success Rate ---', o_row)
        
        if 'Success' in formation_eff.columns:
            o_row = write_to_excel(writer, 'Offense_Stats', to_pct(success_table), '--- 8. Efficiency by Formation (Min 5 Plays) ---', o_row)
            
        # 9. Explosive Plays (Converted to DataFrame)
        exp_data = {
            'Metric': ['Explosive Run Rate (12+)', 'Explosive Pass Rate (16+)'],
            'Percentage': [f"{int(exp_run_pct)}%", f"{int(exp_pass_pct)}%"],
            'Details': [f"{len(explosive_runs)} of {total_runs} runs", f"{len(explosive_passes)} of {total_passes} passes"]
        }
        o_row = write_to_excel(writer, 'Offense_Stats', pd.DataFrame(exp_data).set_index('Metric'), '--- 9. Explosive Play Rates ---', o_row)

        # ==========================================
        # --- DEFENSE TAB ---
        # ==========================================
        d_row = 0
        
        # 1-6. Base & Situational Tendencies
        d_row = write_to_excel(writer, 'Defense_Stats', to_pct(defense_df['Def Front'].value_counts(normalize=True).to_frame('Percentage')), '--- 1. Defensive Front Distribution ---', d_row)
        d_row = write_to_excel(writer, 'Defense_Stats', to_pct(defense_df['Coverage'].value_counts(normalize=True).to_frame('Percentage')), '--- 2. Coverage Distribution ---', d_row)
        d_row = write_to_excel(writer, 'Defense_Stats', to_pct(pd.crosstab([defense_df['Dn'], defense_df['Dist Bucket']], defense_df['Coverage'], normalize='index')), '--- 3. Down & Distance vs. Coverage ---', d_row)
        d_row = write_to_excel(writer, 'Defense_Stats', to_pct(pd.crosstab([defense_df['Dn'], defense_df['Dist Bucket']], defense_df['Def Front'], normalize='index')), '--- 4. Down & Distance vs. Def Front ---', d_row)
        d_row = write_to_excel(writer, 'Defense_Stats', to_pct(pd.crosstab(freq_defense_df['Off form'], freq_defense_df['Def Front'], normalize='index')), '--- 5. Off Formation vs. Def Front (Min 7 Plays) ---', d_row)
        d_row = write_to_excel(writer, 'Defense_Stats', to_pct(formation_coverage), '--- 6. Off Formation vs. Coverage (Min 7 Plays, Max Coverage > 20%) ---', d_row)
        
        # 7-11. Efficiency & Yards Allowed
        d_row = write_to_excel(writer, 'Defense_Stats', to_pct(defense_df['Eff'].value_counts(normalize=True, dropna=True).to_frame('Percentage')), '--- 7. Overall Defensive Efficiency (Allowed) ---', d_row)
        d_row = write_to_excel(writer, 'Defense_Stats', to_pct(pd.crosstab(defense_df['Def Front'], defense_df['Eff'], normalize='index')), '--- 8. Def Front vs. Efficiency (Allowed) ---', d_row)
        d_row = write_to_excel(writer, 'Defense_Stats', to_pct(pd.crosstab(defense_df['Coverage'], defense_df['Eff'], normalize='index')), '--- 9. Coverage vs. Efficiency (Allowed) ---', d_row)
        d_row = write_to_excel(writer, 'Defense_Stats', to_pct(pd.crosstab(defense_df['Def Front'], defense_df['Yards Result'], normalize='index')), '--- 10. Def Front vs. Yards Allowed ---', d_row)
        d_row = write_to_excel(writer, 'Defense_Stats', to_pct(pd.crosstab(defense_df['Coverage'], defense_df['Yards Result'], normalize='index')), '--- 11. Coverage vs. Yards Allowed ---', d_row)

    print("Export Complete! Check your folder for 'Scouting_Report.xlsx'.")

if __name__ == '__main__':
    main()