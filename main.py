from __future__ import annotations

import io

import numpy as np
import pandas as pd
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


def to_pct(data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Format ratios as whole-number percentages for display."""
    return (data * 100).fillna(0).round().astype(int).astype(str) + "%"


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
    df["Eff"] = df["Eff"].str.lower().replace({"y": "Success", "n": "Not Success"})
    df["Dn"] = pd.to_numeric(df["Dn"], errors="coerce").astype("Int64")
    df["Dist"] = pd.to_numeric(df["Dist"], errors="coerce")
    df["Yard ln"] = pd.to_numeric(df["Yard ln"], errors="coerce")
    df["gn/ls"] = pd.to_numeric(df["gn/ls"], errors="coerce").fillna(0)
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
    overall_success_rate = (offense_df["Eff"] == "Success").mean() if total_plays else 0
    explosive_runs = run_plays[run_plays["gn/ls"] >= 12]
    explosive_passes = pass_plays[pass_plays["gn/ls"] >= 16]
    explosive_plays = pd.concat([explosive_runs, explosive_passes])
    explosive_rate = len(explosive_plays) / total_plays if total_plays else 0
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
    formation_work["_Success"] = formation_work["Eff"].eq("Success")
    formation_summary = formation_work.groupby("Off form", observed=False).agg(
        **{
            "Total Plays": ("Off form", "size"),
            "Usage %": ("Off form", lambda values: len(values) / len(frequent_df) if len(frequent_df) else 0),
            "Run %": ("_Run", "mean"),
            "Pass %": ("_Pass", "mean"),
            "Efficiency / Success Rate %": ("_Success", "mean"),
        }
    )
    for column in ["Usage %", "Run %", "Pass %", "Efficiency / Success Rate %"]:
        formation_summary[column] = percentage_column(formation_summary[column], column)
    formation_summary = formation_summary.reset_index().rename(columns={"Off form": "Off Formation"})

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

    explosive_runs = run_plays[run_plays["gn/ls"] >= 12]
    explosive_passes = pass_plays[pass_plays["gn/ls"] >= 16]
    down_distance_title = f"Down & Distance vs. Play Type (Min {min_plays} Plays)"
    formation_title = "Formation Tendencies & Efficiency"
    return {
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
        "tables": [
            ("Play Type Distribution", play_type_distribution),
            (down_distance_title, to_pct(down_distance_table)),
            ("Field Position vs. Play Type", to_pct(pd.crosstab(offense_df["Field Zone"], offense_df["Play type"], normalize="index"))),
            (formation_title, formation_summary),
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
            "Success Rate Allowed %": ("Eff", lambda values: (values == "Success").mean()),
        }
    )
    summary[percentage_name] = summary["Total Plays"] / summary["Total Plays"].sum()
    return summary


def filter_coverage_columns(table: pd.DataFrame, minimum_share: float = 0.20) -> pd.DataFrame:
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
    defense_df["Field Zone"] = np.where(
        defense_df["Yard ln"] <= 20,
        "Red Zone",
        "Field",
    )
    is_pass = defense_df["Play type"].str.contains("Pass|Dropback|Screen|Shot", case=False, na=False)
    is_run = defense_df["Play type"].str.contains("Run|Zone|Power|Gap|Counter", case=False, na=False)
    run_plays = defense_df[is_run]
    pass_plays = defense_df[is_pass]
    explosive = defense_df[(is_run & (defense_df["gn/ls"] >= 12)) | (is_pass & (defense_df["gn/ls"] >= 16))]

    front_summary = distribution_table(defense_df, "Def Front", "Usage %")
    front_summary["Avg YPC Allowed"] = run_plays.groupby("Def Front", dropna=False, observed=False)["gn/ls"].mean()
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
    )["gn/ls"].mean()
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
        "metrics": {
            "Overall Avg Yards Allowed per Play": defense_df["gn/ls"].mean() if len(defense_df) else 0,
            "Overall Defensive Success Rate Allowed (%)": (defense_df["Eff"] == "Success").mean() * 100 if len(defense_df) else 0,
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
        "tables": [
            ("Defensive Front Summary", front_summary),
            (coverage_summary_title, coverage_summary),
            (down_distance_coverage_title, to_pct(down_distance_coverage)),
            (down_distance_front_title, to_pct(down_distance_front)),
            ("Field Zone vs. Coverage", to_pct(field_zone_coverage)),
            ("Field Zone vs. Def Front", to_pct(pd.crosstab(defense_df["Field Zone"], defense_df["Def Front"], normalize="index"))),
            ("Explosive Plays Allowed by Coverage", explosive_coverage),
            ("Explosive Plays Allowed by Def Front", explosive_front),
            (f"Off Formation vs. Coverage (Min {min_plays} Plays)", to_pct(formation_coverage)),
            (f"Off Formation vs. Def Front (Min {min_plays} Plays)", to_pct(pd.crosstab(frequent_df["Off form"], frequent_df["Def Front"], normalize="index"))),
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
            st.info("No plays match this table after filtering.")
        else:
            st.dataframe(table, use_container_width=True)
        if title in analysis.get("filtered_titles", set()):
            st.caption(f"Showing formations/scenarios with at least {min_plays} plays")
        if title in analysis.get("formation_filtered_titles", set()):
            st.caption(f"Showing formations with at least {min_plays} plays")
        if title in analysis.get("coverage_filtered_titles", set()):
            st.caption("Showing coverage columns reaching at least 20% in one or more rows")


def main() -> None:
    st.set_page_config(page_title="Football Scouting Dashboard", page_icon="🏈", layout="wide")
    st.title("🏈 Football Scouting Dashboard")
    uploaded_file = st.file_uploader("Upload a Hudl scouting CSV", type="csv")
    if uploaded_file is None:
        st.info("Upload a CSV file to begin the scouting analysis.")
        return

    try:
        df = clean_data(uploaded_file)
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
    st.caption(f"Showing {len(filtered_df):,} of {len(df):,} plays")
    offense_tab, defense_tab = st.tabs(["Offense", "Defense"])
    with offense_tab:
        render_analysis(offense, min_plays, show_metrics=True)
    with defense_tab:
        render_analysis(defense, min_plays, show_metrics=True)


if __name__ == "__main__":
    main()
