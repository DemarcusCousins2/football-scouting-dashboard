from __future__ import annotations

import io
import os
from pathlib import Path

import anthropic
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


TEXT_COLUMNS = [
    "Play type",
    "Off form",
    "Backfield",
    "Play dir",
    "Eff",
    "Result",
    "Hash",
    "Qtr",
    "Def Front",
    "Coverage",
]

# A run gaining >= 12 yards or a pass gaining >= 16 is treated as "explosive".
EXPLOSIVE_RUN_YDS = 12
EXPLOSIVE_PASS_YDS = 16
# Coverage columns are shown only if used at least this often in some row.
MIN_COVERAGE_SHARE = 0.20
# A formation is a pre-snap "tell" if it leans run or pass at least this strongly.
TELL_THRESHOLD = 0.70

AI_SUMMARY_MODEL = "claude-opus-5"
AI_SUMMARY_SYSTEM = (
    "You are a football defensive and offensive coordinator's scouting analyst. You are given "
    "stat tables built from Hudl film of an OPPONENT. 'Offense' is the opponent's offense (what "
    "our defense faces); 'Defense' is the opponent's defense (what our offense faces). Write a "
    "short game-plan summary in Markdown with exactly two sections: '#### 🛡️ Stopping their "
    "offense' and '#### 🏈 Attacking their defense'. Give 3-5 bullets per section, most "
    "important first. Each bullet must be a concrete, actionable key backed by a number from "
    "the tables (e.g. formation tells, down-and-distance tendencies, coverages that give up "
    "yards). Flag small samples instead of overstating them. No intro or closing text."
)


def to_pct(data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Format ratios as whole-number percentages for display."""
    return (data * 100).fillna(0).round().astype(int).astype(str) + "%"


def success_rate(eff: pd.Series) -> float:
    """Success ratio over plays with a known efficiency (ignores blank/unknown Eff)."""
    known = eff[eff.notna()]
    return known.eq("Success").mean() if len(known) else 0.0


def add_field_position(offense_df: pd.DataFrame) -> pd.DataFrame:
    """Map Hudl's signed yard line to 0-100 field position (own goal 0, opponent goal 100)."""
    positioned = offense_df.copy()
    positioned["Field Pos"] = pd.to_numeric(positioned["Yard ln"], errors="coerce") + 50
    return positioned.dropna(subset=["Field Pos"])


def build_field_position_figure(offense_df: pd.DataFrame):
    """Density of where the opponent runs vs. passes along the length of the field.

    Hudl scouting data has no true (x, y) ball coordinates, so a 2-D heatmap is not
    meaningful. The yard line *is* reliable, so we plot a 1-D field-position density.
    """
    positioned = add_field_position(offense_df)
    positioned = positioned[(positioned["Field Pos"] >= 0) & (positioned["Field Pos"] <= 100)]
    if len(positioned) < 5 or positioned["Field Pos"].nunique() < 2:
        return None

    is_pass = positioned["Play type"].str.contains("Pass|Dropback|Screen|Shot", case=False, na=False)
    is_run = positioned["Play type"].str.contains("Run|Zone|Power|Gap|Counter", case=False, na=False)

    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_xlim(0, 100)
    ax.axvspan(0, 20, color="#C62828", alpha=0.07)    # own territory (backed up)
    ax.axvspan(80, 100, color="#2E7D32", alpha=0.12)  # opponent red zone (inside the 20)
    for yard in range(10, 100, 10):
        ax.axvline(yard, color="#DDDDDD", lw=0.8, zorder=0)
    ax.axvline(50, color="#9E9E9E", lw=1.3, zorder=0)

    plotted = False
    for series, color, label in [
        (positioned.loc[is_run, "Field Pos"], "#1f77b4", "Run"),
        (positioned.loc[is_pass, "Field Pos"], "#d62728", "Pass"),
    ]:
        if series.nunique() >= 2:
            sns.kdeplot(x=series, ax=ax, fill=True, alpha=0.4, color=color,
                        label=label, clip=(0, 100), bw_adjust=0.8, linewidth=1.5)
            plotted = True

    top = ax.get_ylim()[1]
    ax.text(10, top * 0.9, "Backed up", ha="center", color="#C62828", fontsize=9)
    ax.text(90, top * 0.9, "Red zone", ha="center", color="#2E7D32", fontsize=9)
    ax.set_yticks([])
    ax.set_ylabel("Relative play frequency")
    ax.set_xlabel("Field position  (own goal 0  →  midfield 50  →  opponent goal 100)")
    if plotted:
        ax.legend(loc="upper center", frameon=False, ncol=2)
    ax.margins(x=0)
    fig.tight_layout()
    return fig


def render_field_position_density(offense_df: pd.DataFrame) -> None:
    """Render the opponent's run/pass field-position density."""
    st.subheader("Opponent Field-Position Density")
    figure = build_field_position_figure(offense_df)
    if figure is None:
        st.info("Not enough plays with a valid yard line to plot field-position density.")
        return
    st.caption(
        "Each curve is a density estimate: taller means a larger share of that play type happens "
        "at that field position (area under each curve totals 100%). The y-axis is relative "
        "frequency, not a raw play count."
    )
    st.pyplot(figure)
    plt.close(figure)


def build_yards_allowed_figure(defense_df: pd.DataFrame):
    """Density of yards the defense gives up per play, split by run vs. pass defense."""
    work = defense_df.assign(_yards=pd.to_numeric(defense_df["gn/ls"], errors="coerce"))
    work = work.dropna(subset=["_yards"])
    if len(work) < 5 or work["_yards"].nunique() < 2:
        return None

    is_pass = work["Play type"].str.contains("Pass|Dropback|Screen|Shot", case=False, na=False)
    is_run = work["Play type"].str.contains("Run|Zone|Power|Gap|Counter", case=False, na=False)
    low = min(-10.0, float(work["_yards"].quantile(0.01)))
    high = max(20.0, float(work["_yards"].quantile(0.99)))

    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_xlim(low, high)
    ax.axvspan(low, 0, color="#2E7D32", alpha=0.06)   # no gain / loss = good for the defense
    ax.axvline(0, color="#9E9E9E", lw=1.2, zorder=0)
    ax.axvline(EXPLOSIVE_RUN_YDS, color="#1f77b4", lw=1.0, ls="--", zorder=0)   # run explosive
    ax.axvline(EXPLOSIVE_PASS_YDS, color="#d62728", lw=1.0, ls="--", zorder=0)  # pass explosive

    plotted = False
    for series, color, label in [
        (work.loc[is_run, "_yards"], "#1f77b4", "Run defense"),
        (work.loc[is_pass, "_yards"], "#d62728", "Pass defense"),
    ]:
        if series.nunique() >= 2:
            sns.kdeplot(x=series, ax=ax, fill=True, alpha=0.4, color=color,
                        label=label, clip=(low, high), bw_adjust=0.8, linewidth=1.5)
            plotted = True

    top = ax.get_ylim()[1]
    ax.text(EXPLOSIVE_RUN_YDS, top * 0.96, f"Run explosive {EXPLOSIVE_RUN_YDS}+", color="#1f77b4", fontsize=8, ha="left")
    ax.text(EXPLOSIVE_PASS_YDS, top * 0.86, f"Pass explosive {EXPLOSIVE_PASS_YDS}+", color="#d62728", fontsize=8, ha="left")
    ax.set_yticks([])
    ax.set_ylabel("Relative play frequency")
    ax.set_xlabel("Yards allowed on the play  (0 = no gain; left = loss)")
    if plotted:
        ax.legend(loc="upper right", frameon=False)
    ax.margins(x=0)
    fig.tight_layout()
    return fig


def render_yards_allowed_density(defense_df: pd.DataFrame) -> None:
    """Render the defense's yards-allowed density for run vs. pass."""
    st.subheader("Yards Allowed Density (Run vs. Pass Defense)")
    figure = build_yards_allowed_figure(defense_df)
    if figure is None:
        st.info("Not enough defensive plays with a valid gain/loss to plot the yards-allowed density.")
        return
    st.caption(
        "Distribution of yards the defense gives up per play, split by run and pass. A tall peak "
        "near or below 0 means stops; a fatter right tail means they surrender chunk plays. Dashed "
        "lines mark the explosive thresholds (run 12+, pass 16+). Y-axis is relative frequency."
    )
    st.pyplot(figure)
    plt.close(figure)


def excel_numeric_table(table: pd.DataFrame) -> pd.DataFrame:
    """Convert percentage text back to numeric ratios before writing Excel."""
    numeric_table = table.copy()
    for column in numeric_table.columns:
        numeric_table[column] = numeric_table[column].map(
            lambda value: float(value.rstrip("%")) / 100
            if isinstance(value, str) and value.endswith("%") else value
        )
    return numeric_table


def write_to_excel(
    writer: pd.ExcelWriter,
    sheet_name: str,
    table: pd.DataFrame,
    title: str,
    start_row: int,
) -> int:
    """Write a titled Excel table and return the next available row."""
    numeric_table = excel_numeric_table(table).reset_index()
    numeric_table.to_excel(
        writer,
        sheet_name=sheet_name,
        startrow=start_row + 2,
        header=False,
        index=False,
    )
    worksheet = writer.sheets[sheet_name]
    worksheet.merge_range(
        start_row,
        0,
        start_row,
        len(numeric_table.columns) - 1,
        title,
        writer.book.add_format(
            {"bold": True, "bg_color": "#1F4E78", "font_color": "#FFFFFF", "border": 1}
        ),
    )
    worksheet.add_table(
        start_row + 1,
        0,
        start_row + len(numeric_table) + 1,
        len(numeric_table.columns) - 1,
        {
            "name": f"{sheet_name}_{start_row}".replace("-", "_"),
            "style": "Table Style Medium 2",
            "columns": [{"header": str(column)} for column in numeric_table.columns],
        },
    )

    percentage_format = writer.book.add_format({"num_format": "0%"})
    index_width = table.index.nlevels
    for row_position, (_, row) in enumerate(table.iterrows()):
        for column_index, column in enumerate(table.columns):
            value = row[column]
            is_percentage = (
                isinstance(value, str) and value.endswith("%")
                or column in {"Percentage", "Success Rate"}
            )
            numeric_value = numeric_table.iloc[row_position, index_width + column_index]
            if is_percentage and pd.notna(numeric_value):
                worksheet.write(
                    start_row + 2 + row_position,
                    index_width + column_index,
                    numeric_value,
                    percentage_format,
                )
    return start_row + len(table) + 4


def clean_data(uploaded_file) -> pd.DataFrame:
    """Read and normalize a Hudl scouting CSV."""
    df = pd.read_csv(uploaded_file)
    required_columns = {"ODK", "Eff", "Dn", "Dist", "Yard ln", "gn/ls", "Play type"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required Hudl columns: {', '.join(missing_columns)}")

    for column in TEXT_COLUMNS:
        if column not in df.columns:
            df[column] = "Unknown"
        df[column] = df[column].fillna("Unknown").astype(str).str.strip().str.title()

    df["ODK"] = df["ODK"].fillna("Unknown").astype(str).str.strip().str.upper()
    # Map only y/n to Success/Not Success; anything else (blanks, stray "5"/"-5") becomes NaN
    # so junk values do not silently count against success rates.
    df["Eff"] = (
        df["Eff"].astype(str).str.strip().str.lower().map({"y": "Success", "n": "Not Success"})
    )
    df["Dn"] = pd.to_numeric(df["Dn"], errors="coerce").astype("Int64")
    df["Dist"] = pd.to_numeric(df["Dist"], errors="coerce")
    df["Yard ln"] = pd.to_numeric(df["Yard ln"], errors="coerce")
    # Keep missing gains as NaN (do NOT fill with 0): a blank gn/ls means the yardage is
    # unknown, so it should be excluded from averages/densities rather than counted as a 0-yard play.
    df["gn/ls"] = pd.to_numeric(df["gn/ls"], errors="coerce")
    df = df[df["Play type"] != "Timeout"].copy()

    df["Dist Bucket"] = pd.cut(
        df["Dist"],
        bins=[0, 3, 7, 100],
        labels=["Short (1-3)", "Medium (4-7)", "Long (8+)"]
    )
    df["Field Zone"] = pd.cut(
        df["Yard ln"],
        bins=[-100, -20, 0, 20, 100],
        labels=[
            "Backed Up (Own 1-20)",
            "Own 21 - Midfield",
            "Opponent 40 - RedZone",
            "RedZone (Inside 20)",
        ],
    )
    return df


def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Render sidebar filters and return the selected plays and threshold."""
    st.sidebar.header("Filters")
    min_plays = st.sidebar.number_input(
        "Min Plays Threshold",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )
    filtered_df = df

    down_options = sorted(df["Dn"].dropna().unique().tolist())
    selected_downs = st.sidebar.multiselect("Down", down_options, default=down_options)
    if selected_downs:
        filtered_df = filtered_df[filtered_df["Dn"].isin(selected_downs)]

    quarter_options = sorted(df["Qtr"].dropna().unique().tolist())
    selected_quarters = st.sidebar.multiselect("Quarter", quarter_options, default=quarter_options)
    if selected_quarters:
        filtered_df = filtered_df[filtered_df["Qtr"].isin(selected_quarters)]

    zone_options = [str(zone) for zone in df["Field Zone"].dropna().unique()]
    selected_zones = st.sidebar.multiselect("Field Zone", zone_options, default=zone_options)
    if selected_zones:
        filtered_df = filtered_df[filtered_df["Field Zone"].astype(str).isin(selected_zones)]
    return filtered_df, int(min_plays)


def offense_analysis(df: pd.DataFrame, min_plays: int) -> dict:
    offense_df = df[df["ODK"] == "O"].copy().replace(["Unknown", "unknown", ""], np.nan)
    form_counts = offense_df["Off form"].value_counts()
    frequent_forms = form_counts[form_counts >= min_plays].index
    frequent_df = offense_df[offense_df["Off form"].isin(frequent_forms)]
    down_distance_columns = ["Dn", "Dist Bucket"]
    down_distance_counts = offense_df.groupby(down_distance_columns, observed=False).size()
    frequent_down_distance = down_distance_counts[down_distance_counts >= min_plays].index
    down_distance_table = pd.crosstab(
        [offense_df["Dn"], offense_df["Dist Bucket"]],
        offense_df["Play type"],
        normalize="index",
    )
    down_distance_table = down_distance_table.loc[
        down_distance_table.index.isin(frequent_down_distance)
    ]
    is_pass = offense_df["Play type"].str.contains("Pass|Dropback|Screen|Shot", case=False, na=False)
    is_run = offense_df["Play type"].str.contains("Run|Zone|Power|Gap|Counter", case=False, na=False)
    pass_plays, run_plays = offense_df[is_pass], offense_df[is_run]
    total_plays = len(offense_df)
    total_classified_plays = len(run_plays) + len(pass_plays)
    overall_success_rate = success_rate(offense_df["Eff"]) if total_plays else 0
    explosive_runs = run_plays[run_plays["gn/ls"] >= EXPLOSIVE_RUN_YDS]
    explosive_passes = pass_plays[pass_plays["gn/ls"] >= EXPLOSIVE_PASS_YDS]
    explosive_plays = pd.concat([explosive_runs, explosive_passes])
    # Rate over plays with a known gain only — unknown-yardage plays can't be judged explosive.
    plays_with_yardage = int(offense_df["gn/ls"].notna().sum())
    explosive_rate = len(explosive_plays) / plays_with_yardage if plays_with_yardage else 0
    run_share = len(run_plays) / total_classified_plays if total_classified_plays else 0
    pass_share = len(pass_plays) / total_classified_plays if total_classified_plays else 0

    call_type = pd.Series("Other", index=offense_df.index)
    call_type.loc[is_run] = "Run"
    call_type.loc[is_pass] = "Pass"
    call_counts = call_type[call_type.isin(["Run", "Pass"])].value_counts()
    play_type_distribution = pd.DataFrame({"Calls": call_counts})
    play_type_distribution["Usage %"] = percentage_column(
        call_counts / call_counts.sum() if call_counts.sum() else call_counts * 0,
        "Usage %",
    )

    formation_work = frequent_df.copy()
    formation_work["_Run"] = is_run.loc[frequent_df.index]
    formation_work["_Pass"] = is_pass.loc[frequent_df.index]
    formation_summary = formation_work.groupby("Off form", observed=False).agg(
        **{
            "Total Plays": ("Off form", "size"),
            "Usage %": ("Off form", lambda values: len(values) / len(frequent_df) if len(frequent_df) else 0),
            "Run %": ("_Run", "mean"),
            "Pass %": ("_Pass", "mean"),
            "Efficiency / Success Rate %": ("Eff", success_rate),
        }
    )
    for column in ["Usage %", "Run %", "Pass %", "Efficiency / Success Rate %"]:
        formation_summary[column] = percentage_column(formation_summary[column], column)
    formation_summary = formation_summary.reset_index().rename(columns={"Off form": "Off Formation"})

    # Pre-snap "tells": formations that lean strongly run or pass (>= TELL_THRESHOLD of their
    # classified calls). These are the actionable keys a defense can read before the snap.
    tell_rows = []
    for formation in frequent_forms:
        formation_rows = formation_work[formation_work["Off form"] == formation]
        run_calls = int(formation_rows["_Run"].sum())
        pass_calls = int(formation_rows["_Pass"].sum())
        classified = run_calls + pass_calls
        if not classified:
            continue
        run_lean = run_calls / classified
        lean = max(run_lean, 1 - run_lean)
        if lean >= TELL_THRESHOLD:
            tell_rows.append(
                {
                    "Off Formation": formation,
                    "Total Plays": len(formation_rows),
                    "Predicts": "Run" if run_lean >= 0.5 else "Pass",
                    "Confidence": lean,
                }
            )
    tells_table = pd.DataFrame(tell_rows)
    if not tells_table.empty:
        tells_table = tells_table.sort_values("Confidence", ascending=False).reset_index(drop=True)
        tells_table["Confidence"] = percentage_column(tells_table["Confidence"], "Confidence")

    explosive_rows = []
    for formation in frequent_forms:
        formation_plays = offense_df[offense_df["Off form"] == formation]
        for play_label, play_mask, explosive_frame in [
            ("Run", is_run, explosive_runs),
            ("Pass", is_pass, explosive_passes),
        ]:
            total_type_plays = len(formation_plays[play_mask.loc[formation_plays.index]])
            explosive_count = len(explosive_frame[explosive_frame["Off form"] == formation])
            explosive_rows.append(
                {
                    "Off Formation": formation,
                    "Play Type": play_label,
                    "Explosive Plays": explosive_count,
                    "Rate %": explosive_count / total_type_plays if total_type_plays else 0,
                }
            )
    explosive_breakdown = pd.DataFrame(explosive_rows)
    if not explosive_breakdown.empty:
        explosive_breakdown["Rate %"] = percentage_column(explosive_breakdown["Rate %"], "Rate %")
        explosive_breakdown = explosive_breakdown[explosive_breakdown["Rate %"] != "0%"].copy()

    down_distance_title = f"Down & Distance vs. Play Type (Min {min_plays} Plays)"
    formation_title = "Formation Tendencies & Efficiency"
    tells_title = f"🎯 Pre-Snap Tells: Formation → Run/Pass (Min {min_plays} Plays, ≥ {int(TELL_THRESHOLD * 100)}% Lean)"
    return {
        "offense_df": offense_df,
        "metrics": {
            "Overall Success Rate (%)": f"{overall_success_rate * 100:.1f}%",
            "Explosive Play Rate (%)": f"{explosive_rate * 100:.1f}%",
            "Overall Avg Yards Per Play": offense_df["gn/ls"].mean() if total_plays else 0,
            "Run / Pass Ratio": f"{run_share * 100:.0f}% Run / {pass_share * 100:.0f}% Pass",
        },
        "filtered_titles": {
            down_distance_title,
        },
        "formation_filtered_titles": {formation_title},
        "tell_titles": {tells_title},
        "tables": [
            (tells_title, tells_table),
            (formation_title, formation_summary),
            (down_distance_title, to_pct(down_distance_table)),
            ("Play Type Distribution", play_type_distribution),
            ("Field Position vs. Play Type", to_pct(pd.crosstab(offense_df["Field Zone"], offense_df["Play type"], normalize="index"))),
            ("Explosive Play Rates & Breakdown", explosive_breakdown),
        ],
    }


def percentage_column(data: pd.Series, name: str) -> pd.Series:
    """Format one ratio column with the shared percentage formatter."""
    return to_pct(data).rename(name)


def distribution_table(df: pd.DataFrame, group_column: str, percentage_name: str) -> pd.DataFrame:
    """Build a count, usage, and success summary for a defensive grouping."""
    summary = df.groupby(group_column, dropna=False, observed=False).agg(
        **{
            "Total Plays": (group_column, "size"),
            "Success Rate Allowed %": ("Eff", success_rate),
        }
    )
    summary[percentage_name] = summary["Total Plays"] / summary["Total Plays"].sum()
    return summary


def filter_coverage_columns(table: pd.DataFrame, minimum_share: float = MIN_COVERAGE_SHARE) -> pd.DataFrame:
    """Keep coverage columns used at or above the minimum share in any row."""
    if table.empty:
        return table
    return table.loc[:, (table >= minimum_share).any(axis=0)]


def defense_analysis(df: pd.DataFrame, min_plays: int) -> dict:
    defense_df = df[df["ODK"] == "D"].copy().replace(["Unknown", "unknown", ""], np.nan)
    third_down_plays = defense_df[defense_df["Dn"] == 3]
    third_down_stops = third_down_plays[third_down_plays["Eff"].astype(str).str.strip().str.lower().isin(
        {"not success", "no", "n", "0"}
    )]
    # Yard ln is signed (+50 = the goal line the offense is attacking), so the red zone
    # (offense inside the opponent 20) is Yard ln >= 30 — not <= 20, which tagged most of the field.
    defense_df["Field Zone"] = np.where(
        defense_df["Yard ln"] >= 30,
        "Red Zone",
        "Field",
    )
    is_pass = defense_df["Play type"].str.contains("Pass|Dropback|Screen|Shot", case=False, na=False)
    is_run = defense_df["Play type"].str.contains("Run|Zone|Power|Gap|Counter", case=False, na=False)
    run_plays = defense_df[is_run]
    pass_plays = defense_df[is_pass]
    explosive = defense_df[
        (is_run & (defense_df["gn/ls"] >= EXPLOSIVE_RUN_YDS))
        | (is_pass & (defense_df["gn/ls"] >= EXPLOSIVE_PASS_YDS))
    ]

    # Exclude uncharted (blank) defensive fronts so the summary doesn't show a "None" row.
    charted_front = defense_df[defense_df["Def Front"].notna()]
    front_summary = distribution_table(charted_front, "Def Front", "Usage %")
    front_summary["Avg YPC Allowed"] = (
        run_plays[run_plays["Def Front"].notna()]
        .groupby("Def Front", observed=False)["gn/ls"].mean().round(1)
    )
    front_summary["Usage %"] = percentage_column(front_summary["Usage %"], "Usage %")
    front_summary["Success Rate Allowed %"] = percentage_column(
        front_summary["Success Rate Allowed %"], "Success Rate Allowed %"
    )
    front_summary = front_summary.reset_index()[
        ["Def Front", "Total Plays", "Usage %", "Avg YPC Allowed", "Success Rate Allowed %"]
    ]

    coverage_counts = defense_df["Coverage"].value_counts()
    frequent_coverages = coverage_counts[coverage_counts >= min_plays].index
    coverage_df = defense_df[defense_df["Coverage"].isin(frequent_coverages)]
    coverage_summary = distribution_table(coverage_df, "Coverage", "Usage %")
    coverage_summary["Avg YPA Allowed"] = pass_plays[pass_plays["Coverage"].isin(frequent_coverages)].groupby(
        "Coverage", dropna=False, observed=False
    )["gn/ls"].mean().round(1)
    coverage_summary["Usage %"] = percentage_column(coverage_summary["Usage %"], "Usage %")
    coverage_summary["Success Rate Allowed %"] = percentage_column(
        coverage_summary["Success Rate Allowed %"], "Success Rate Allowed %"
    )
    coverage_summary = coverage_summary.reset_index()[
        ["Coverage", "Total Plays", "Usage %", "Avg YPA Allowed", "Success Rate Allowed %"]
    ]

    form_counts = defense_df["Off form"].value_counts()
    frequent_df = defense_df[defense_df["Off form"].isin(form_counts[form_counts >= min_plays].index)]
    explosive_coverage = explosive["Coverage"].value_counts().rename("Count").to_frame()
    explosive_front = explosive["Def Front"].value_counts().rename("Count").to_frame()
    for table in (explosive_coverage, explosive_front):
        table["% of total explosives"] = percentage_column(
            table["Count"] / len(explosive) if len(explosive) else table["Count"] * 0,
            "% of total explosives",
        )
        table.reset_index(inplace=True)

    down_distance_coverage = pd.crosstab(
        [defense_df["Dn"], defense_df["Dist Bucket"]], defense_df["Coverage"], normalize="index"
    )
    down_distance_front = pd.crosstab(
        [defense_df["Dn"], defense_df["Dist Bucket"]], defense_df["Def Front"], normalize="index"
    )
    down_distance_counts = defense_df.groupby(
        ["Dn", "Dist Bucket"], observed=False
    ).size()
    frequent_down_distance = down_distance_counts[down_distance_counts >= min_plays].index
    down_distance_coverage = down_distance_coverage.loc[
        down_distance_coverage.index.isin(frequent_down_distance)
    ]
    down_distance_coverage = filter_coverage_columns(down_distance_coverage)
    down_distance_front = down_distance_front.loc[
        down_distance_front.index.isin(frequent_down_distance)
    ]

    # Defensive pre-snap tells: down & distance situations where the defense strongly favors one
    # coverage — the offense can key these before the snap (mirror of the offensive formation tells).
    def_tell_rows = []
    coverage_known = defense_df.dropna(subset=["Coverage"])
    situation_counts = coverage_known.groupby(["Dn", "Dist Bucket"], observed=False).size()
    for situation, situation_plays in situation_counts.items():
        if situation_plays < min_plays:
            continue
        down_value, dist_bucket = situation
        subset = coverage_known[
            (coverage_known["Dn"] == down_value) & (coverage_known["Dist Bucket"] == dist_bucket)
        ]
        coverage_share = subset["Coverage"].value_counts(normalize=True)
        if coverage_share.empty:
            continue
        lean = float(coverage_share.iloc[0])
        if lean >= TELL_THRESHOLD:
            def_tell_rows.append(
                {
                    "Situation": f"{int(down_value)} & {dist_bucket}",
                    "Total Plays": int(len(subset)),
                    "Predicts Coverage": coverage_share.index[0],
                    "Confidence": lean,
                }
            )
    def_tells_table = pd.DataFrame(def_tell_rows)
    if not def_tells_table.empty:
        def_tells_table = def_tells_table.sort_values("Confidence", ascending=False).reset_index(drop=True)
        def_tells_table["Confidence"] = percentage_column(def_tells_table["Confidence"], "Confidence")
    def_tells_title = f"🎯 Defensive Tells: Down & Distance → Coverage (Min {min_plays} Plays, ≥ {int(TELL_THRESHOLD * 100)}% Lean)"

    coverage_summary_title = f"Coverage Summary (Min {min_plays} Plays)"
    down_distance_coverage_title = f"Down & Distance vs. Coverage (Min {min_plays} Plays)"
    down_distance_front_title = f"Down & Distance vs. Def Front (Min {min_plays} Plays)"
    field_zone_coverage = filter_coverage_columns(
        pd.crosstab(defense_df["Field Zone"], defense_df["Coverage"], normalize="index")
    )
    formation_coverage = filter_coverage_columns(
        pd.crosstab(frequent_df["Off form"], frequent_df["Coverage"], normalize="index")
    )
    coverage_filtered_titles = {
        down_distance_coverage_title,
        "Field Zone vs. Coverage",
        f"Off Formation vs. Coverage (Min {min_plays} Plays)",
    }
    return {
        "defense_df": defense_df,
        "metrics": {
            "Overall Avg Yards Allowed per Play": defense_df["gn/ls"].mean() if len(defense_df) else 0,
            "Overall Defensive Success Rate Allowed (%)": success_rate(defense_df["Eff"]) * 100 if len(defense_df) else 0,
            "Explosive Plays Allowed Count": len(explosive),
            "3rd Down Stop Rate (%)": len(third_down_stops) / len(third_down_plays) * 100 if len(third_down_plays) else 0,
        },
        "filtered_titles": {
            coverage_summary_title,
            down_distance_coverage_title,
            down_distance_front_title,
            f"Off Formation vs. Coverage (Min {min_plays} Plays)",
            f"Off Formation vs. Def Front (Min {min_plays} Plays)",
        },
        "coverage_filtered_titles": coverage_filtered_titles,
        "tell_titles": {def_tells_title},
        "tables": [
            (def_tells_title, def_tells_table),
            (coverage_summary_title, coverage_summary),
            ("Defensive Front Summary", front_summary),
            (down_distance_coverage_title, to_pct(down_distance_coverage)),
            (down_distance_front_title, to_pct(down_distance_front)),
            ("Field Zone vs. Coverage", to_pct(field_zone_coverage)),
            ("Field Zone vs. Def Front", to_pct(pd.crosstab(defense_df["Field Zone"], defense_df["Def Front"], normalize="index"))),
            (f"Off Formation vs. Coverage (Min {min_plays} Plays)", to_pct(formation_coverage)),
            (f"Off Formation vs. Def Front (Min {min_plays} Plays)", to_pct(pd.crosstab(frequent_df["Off form"], frequent_df["Def Front"], normalize="index"))),
            ("Explosive Plays Allowed by Coverage", explosive_coverage),
            ("Explosive Plays Allowed by Def Front", explosive_front),
        ],
    }


def build_excel_report(offense: dict, defense: dict) -> bytes:
    """Build the filtered offense and defense report in memory."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet_name, analysis in [("Offense_Stats", offense), ("Defense_Stats", defense)]:
            row = 0
            for title, table in analysis["tables"]:
                if not table.empty:
                    row = write_to_excel(writer, sheet_name, table, title, row)
    return output.getvalue()


def render_analysis(analysis: dict, min_plays: int, show_metrics: bool = False) -> None:
    if show_metrics:
        metric_columns = st.columns(len(analysis["metrics"]))
        for column, (label, value) in zip(metric_columns, analysis["metrics"].items()):
            suffix = "%" if "%" in label else ""
            display_value = value if isinstance(value, str) else f"{value:.1f}{suffix}"
            column.metric(label, display_value)
    for title, table in analysis["tables"]:
        st.subheader(title)
        if table.empty:
            if title in analysis.get("tell_titles", set()):
                st.info("No strong tells at this threshold — the group mixes its calls fairly evenly.")
            else:
                st.info("No plays match this table after filtering.")
        else:
            st.dataframe(table, use_container_width=True)
        if title in analysis.get("filtered_titles", set()):
            st.caption(f"Showing formations/scenarios with at least {min_plays} plays")
        if title in analysis.get("formation_filtered_titles", set()):
            st.caption(f"Showing formations with at least {min_plays} plays")
        if title in analysis.get("tell_titles", set()):
            st.caption(
                "Confidence = the share of that group's run/pass (or coverage) calls that go the "
                "predicted way — e.g. 80% means 4 of every 5 calls. From a single game these can "
                "read 100%; adding more games will settle them to realistic values."
            )
        if title in analysis.get("coverage_filtered_titles", set()):
            st.caption(
                f"Showing coverage columns reaching at least {int(MIN_COVERAGE_SHARE * 100)}% in one or more rows"
            )


def build_summary_digest(offense: dict, defense: dict, play_count: int, min_plays: int) -> str:
    """Flatten the filtered metrics and tables into plain text for the AI summary prompt."""
    lines = [f"Plays in current filter: {play_count} (tables require at least {min_plays} plays)."]
    for side, analysis in [("OFFENSE", offense), ("DEFENSE", defense)]:
        lines.append(f"\n=== {side} ===")
        for label, value in analysis["metrics"].items():
            lines.append(f"{label}: {value if isinstance(value, str) else f'{value:.1f}'}")
        for title, table in analysis["tables"]:
            if not table.empty:
                lines.append(f"\n## {title}\n{table.to_csv()}")
    return "\n".join(lines)


def anthropic_api_key() -> str | None:
    """API key from Streamlit secrets (hosted app) or the environment (local runs)."""
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
    except FileNotFoundError:
        key = None
    return key or os.environ.get("ANTHROPIC_API_KEY")


@st.cache_data(show_spinner=False)
def generate_ai_summary(digest: str, api_key: str) -> str:
    """Ask Claude for the key offense/defense points; cached so each filter combo is billed once."""
    client = anthropic.Anthropic(api_key=api_key)
    response = client.beta.messages.create(
        model=AI_SUMMARY_MODEL,
        max_tokens=16000,
        system=AI_SUMMARY_SYSTEM,
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": digest}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined to summarize this data.")
    return "".join(block.text for block in response.content if block.type == "text").strip()


def render_ai_summary(offense: dict, defense: dict, play_count: int, min_plays: int) -> None:
    """Render the AI game-plan summary above the offense/defense tabs."""
    st.subheader("🤖 AI Scouting Summary")
    api_key = anthropic_api_key()
    if not api_key:
        st.info("Set ANTHROPIC_API_KEY (environment variable or Streamlit secret) to enable the AI summary.")
        return
    digest = build_summary_digest(offense, defense, play_count, min_plays)
    try:
        with st.spinner("Summarizing the most important tendencies..."):
            summary = generate_ai_summary(digest, api_key)
    except (anthropic.APIError, RuntimeError) as error:
        st.warning(f"AI summary unavailable: {error}")
        return
    with st.container(border=True):
        st.markdown(summary)
    st.caption("AI-generated from the filtered tables below. Double-check key calls against the film.")


def latest_csv_path() -> Path | None:
    """Most recently modified CSV in the app folder, or None if there aren't any."""
    base_dir = Path(__file__).resolve().parent
    csv_files = sorted(base_dir.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    return csv_files[0] if csv_files else None


def main() -> None:
    st.set_page_config(page_title="Football Scouting Dashboard", page_icon="🏈", layout="wide")
    st.title("🏈 Football Scouting Dashboard")
    uploaded_file = st.file_uploader("Upload a Hudl scouting CSV", type="csv")

    # If nothing is uploaded, fall back to the newest CSV sitting in the app folder,
    # so you can just drop a new team's export in and reload.
    data_source = uploaded_file
    auto_loaded_name = None
    if data_source is None:
        latest = latest_csv_path()
        if latest is None:
            st.info("Upload a CSV file — or drop one in the app folder — to begin the scouting analysis.")
            return
        data_source = str(latest)
        auto_loaded_name = latest.name

    try:
        df = clean_data(data_source)
        filtered_df, min_plays = apply_filters(df)
        offense = offense_analysis(filtered_df, min_plays)
        defense = defense_analysis(filtered_df, min_plays)
    except (ValueError, pd.errors.ParserError) as error:
        st.error(str(error))
        return

    st.sidebar.download_button(
        "📥 Download Excel Report",
        data=build_excel_report(offense, defense),
        file_name="Scouting_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    if auto_loaded_name:
        st.caption(f"Auto-loaded newest CSV in the app folder: {auto_loaded_name}")
    st.caption(f"Showing {len(filtered_df):,} of {len(df):,} plays")
    render_ai_summary(offense, defense, len(filtered_df), min_plays)
    offense_tab, defense_tab = st.tabs(["Offense", "Defense"])
    with offense_tab:
        render_analysis(offense, min_plays, show_metrics=True)
        render_field_position_density(offense["offense_df"])
    with defense_tab:
        render_analysis(defense, min_plays, show_metrics=True)
        render_yards_allowed_density(defense["defense_df"])


if __name__ == "__main__":
    main()
