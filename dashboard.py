import streamlit as st
from PIL import Image
import pandas as pd
import json
import os
import numpy as np
from datetime import datetime

##############################################

# Custom CSS for styling sidebar
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            width: 600px !important;
            background-color: #f0f2f6;
        }
        [data-testid="stSidebar"] > div:first-child {
            width: 100% !important;
            padding: 2rem;
            box-sizing: border-box;
        }
        .sidebar-content {
            width: 100%;
            text-align: left;
            font-family: 'Segoe UI', sans-serif;
        }
        .sidebar-content > * {
            width: 100% !important;
        }

        /* Only affect expander in the main content */
        /* Match the expander title reliably */
        [data-testid="stExpander"] summary {
            font-size: 0;
        }

        [data-testid="stExpander"] summary span {
            font-size: 22px !important;
            font-weight: bold !important;
            color: #1a1a1a !important;
        }
    </style>
""", unsafe_allow_html=True)

##############################################

data_path = "data/press_releases_combined.csv"

global_week_path = "data/global_trends_weekly_score.csv"
global_month_path = "data/global_trends_monthly_score.csv"
global_quarter_path = "data/global_trends_quarterly_score.csv"
global_half_year_path = "data/global_trends_6months_score.csv"
global_year_path = "data/global_trends_12months_score.csv"

local_week_path = "data/local_trends_weekly_score.csv"
local_month_path = "data/local_trends_monthly_score.csv"
local_quarter_path = "data/local_trends_quarterly_score.csv"
local_half_year_path = "data/local_trends_6months_score.csv"
local_year_path = "data/local_trends_12months_score.csv"

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

press_data = load_data(data_path)

global_week = load_data(global_week_path)
global_month = load_data(global_month_path)
global_quarter = load_data(global_quarter_path)
global_half_year = load_data(global_half_year_path)
global_year = load_data(global_year_path)

local_week = load_data(local_week_path)
local_month = load_data(local_month_path)
local_quarter = load_data(local_quarter_path)
local_half_year = load_data(local_half_year_path)
local_year = load_data(local_year_path)

##############################################

# Determine the number of points based on the selected period
period_mapping = {
    "Past Week": 7,
    "Past Month": 30,
    "Past Quarter": 90,
    "Six Months": 180,
    "Up to A Year": 365
}

# Mapping from selection to dataset
date_region_map = {
    "Local": {
        "Past Week": local_week,
        "Past Month": local_month,
        "Past Quarter": local_quarter,
        "Six Months": local_half_year,
        "Up to A Year": local_year,
    },
    "Global": {
        "Past Week": global_week,
        "Past Month": global_month,
        "Past Quarter": global_quarter,
        "Six Months": global_half_year,
        "Up to A Year": global_year,
    }
}

##############################################

def filter_by_pestle(df: pd.DataFrame, selected_pestle: str) -> pd.DataFrame:
    if selected_pestle == "All":
        return df
    if "PESTLE" in df.columns:
        return df[df["PESTLE"].str.contains(selected_pestle, case=False, na=False)]
    elif "PESTLE Tag1" in df.columns and "PESTLE Tag2" in df.columns:
        return df[
            df["PESTLE Tag1"].str.contains(selected_pestle, case=False, na=False) |
            df["PESTLE Tag2"].str.contains(selected_pestle, case=False, na=False)
        ]
    else:
        st.warning("⚠️ No PESTLE columns found for filtering.")
        return df

# Custom-styled function to generate a press release expander with a bold, large title and fixed icon.

def show_press_release_from_index(index: int, df: pd.DataFrame, pr_df):
    # Get and parse the link(s)
    link_cell = str(df.loc[index, "link"]).strip()
    link_ids = [int(x) for x in link_cell.split(",") if x.strip().isdigit()]

    # Get trend title
    title = str(df.loc[index, "Trend"])
    display_title = f"📰 {title}"

    st.markdown(f"""
    <div style='font-size: 20px; font-weight: bold; margin-bottom: -10px;'>
        {display_title}
    </div>
    """, unsafe_allow_html=True)

    # Show PESTLE tags and description
    st.markdown(f"**🧩 PESTLE Tags:** {df.loc[index, 'PESTLE']}")
    st.markdown(f"**📝 Trend Description:** {df.loc[index, 'Description']}")

    # Show selected metrics as a 1-row dataframe
    metric_columns = ["DC", "SS", "AC", "AT", "RL", "SC", "EF", "IB", "Total"]
    if all(col in df.columns for col in metric_columns):
        metric_df = df.loc[[index], metric_columns]
        st.dataframe(metric_df, use_container_width=True)
    else:
        st.warning("⚠️ Some metric columns are missing from the dataset.")
        
    # Expander with linked press releases sorted by date
    with st.expander(f"🔗 Trends item {index} — linked to {len(link_ids)} press release(s)"):
        valid_rows = []
        for link_id in link_ids:
            if 0 <= link_id < len(pr_df):
                valid_rows.append(pr_df.loc[link_id])
            else:
                st.warning(f"⚠️ Invalid link index: {link_id}")

        if valid_rows:
            pr_subset = pd.DataFrame(valid_rows)

            # Ensure 'date' column is datetime type
            pr_subset["date"] = pd.to_datetime(pr_subset["date"], errors="coerce")

            # Sort by date ascending
            pr_subset = pr_subset.sort_values(by="date", ascending=False)

            for _, row in pr_subset.iterrows():
                st.markdown(f"**📌 Title:** {row['title']}")
                st.markdown(f"**📅 Date:** {row['date'].strftime('%Y-%m-%d')}  **🗺️ Location:** {row['publish_location']}")
                st.markdown(f"[🔗 Link]({row['link']})\n")
                st.markdown(f"**📄First Paragraph:** {row['first_paragraph']}\n")
                st.markdown("---")
        else:
            st.info("ℹ️ No valid press releases found for this trend.")

def show_global_trends(index: int, df: pd.DataFrame):
    # Get trend title
    title = str(df.loc[index, "trend"])
    display_title = f"📰 {title}"

    st.markdown(f"""
    <div style='font-size: 20px; font-weight: bold; margin-bottom: -10px;'>
        {display_title}
    </div>
    """, unsafe_allow_html=True)

    # Show PESTLE tags and description
    st.markdown(f"**🧩 PESTLE Tags:** {df.loc[index, 'PESTLE Tag1']}, {df.loc[index, 'PESTLE Tag2']}")
    st.markdown(f"**📝 Global Trend Summary:** {df.loc[index, 'summary']}")

    # Show selected metrics as a 1-row dataframe
    metric_columns = ["DC", "SS", "AC", "AT", "RL", "SC", "EF", "IB", "Total"]
    if all(col in df.columns for col in metric_columns):
        metric_df = df.loc[[index], metric_columns]
        st.dataframe(metric_df, use_container_width=True)
    else:
        st.warning("⚠️ Some metric columns are missing from the dataset.")
        
    # Expander with linked press releases sorted by date
    with st.expander(f"🔗 Trends linked to {df.loc[index, 'source']} press release(s)"):
        st.markdown(f"**📌 Title:** {df.loc[index, 'source title']}")
        st.markdown(f"**📅 Date:** {df.loc[index, 'source date']}  **🗺️ Location:** {df.loc[index, 'source location']}")
        st.markdown(f"[🔗 Link]({df.loc[index, 'links']})\n")
        st.markdown("---")

# Assuming selected_df is already filtered by region/date
# and contains either a 'PESTLE' column or 'PESTLE1'/'PESTLE2'

# --- Prepare time-series frequency data ---
def compute_pestle_counts(df: pd.DataFrame, selected_pestle: str) -> pd.DataFrame:
    df = df.copy()

    if "PESTLE" in df.columns:
        if selected_pestle == "All":
            # Count each unique tag across all rows
            pestle_counts = df["PESTLE"].dropna().str.split(",").explode().str.strip().value_counts()
        else:
            pestle_counts = pd.Series({
                selected_pestle: df["PESTLE"].str.contains(selected_pestle, case=False, na=False).sum()
            })

    elif "PESTLE Tag1" in df.columns and "PESTLE Tag2" in df.columns:
        tags = []
        if selected_pestle == "All":
            tags = pd.concat([df["PESTLE Tag1"], df["PESTLE Tag2"]]).dropna().str.strip()
            pestle_counts = tags.value_counts()
        else:
            count = (
                df["PESTLE Tag1"].str.contains(selected_pestle, case=False, na=False).sum() +
                df["PESTLE Tag2"].str.contains(selected_pestle, case=False, na=False).sum()
            )
            pestle_counts = pd.Series({selected_pestle: count})
    else:
        st.warning("⚠️ No PESTLE columns found.")
        return pd.DataFrame()

    return pestle_counts.reset_index().rename(columns={"index": "PESTLE", 0: "Frequency"}).set_index("PESTLE")

def compute_pestle_counts_by_source(local_df: pd.DataFrame, global_df: pd.DataFrame) -> pd.DataFrame:
    # Helper to extract and count PESTLEs
    def extract_counts(df: pd.DataFrame) -> pd.Series:
        if "PESTLE" in df.columns:
            tags = df["PESTLE"].dropna().str.split(",").explode().str.strip()
        elif "PESTLE Tag1" in df.columns and "PESTLE Tag2" in df.columns:
            tags = pd.concat([df["PESTLE Tag1"], df["PESTLE Tag2"]]).dropna().str.lower().str.strip()
        else:
            return pd.Series(dtype=int)
        return tags.value_counts()

    local_counts = extract_counts(local_df).rename("Local")
    global_counts = extract_counts(global_df).rename("Global")

    # Combine into single DataFrame
    combined = pd.concat([local_counts, global_counts], axis=1).fillna(0).astype(int)
    return combined

def compute_pestle_timeseries_multiline(df: pd.DataFrame) -> pd.DataFrame:
    STANDARD_PESTLES = ["political", "economic", "social", "technological", "legal", "environmental"]

    # Identify the correct date column
    date_column = None
    for col in ["date", "Date", "source date", "published_date"]:
        if col in df.columns:
            date_column = col
            break
    if not date_column:
        st.warning("⚠️ No date column found.")
        return pd.DataFrame()

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=[date_column])

    # Extract and normalize PESTLE tags
    if "PESTLE" in df.columns:
        pestle_series = df["PESTLE"].dropna().str.lower().str.split(",")
    elif "PESTLE Tag1" in df.columns and "PESTLE Tag2" in df.columns:
        combined = df[["PESTLE Tag1", "PESTLE Tag2"]].astype(str).agg(",".join, axis=1)
        pestle_series = combined.str.lower().str.split(",")
    else:
        st.warning("⚠️ No PESTLE columns found.")
        return pd.DataFrame()

    # Explode rows: one per PESTLE per date
    expanded_df = pd.DataFrame({
        "date": df[date_column].values.repeat(pestle_series.str.len()),
        "pestle": [tag.strip() for sublist in pestle_series.tolist() for tag in sublist]
    })

    # Filter only standard pestle tags
    expanded_df = expanded_df[expanded_df["pestle"].isin(STANDARD_PESTLES)]

    # Group by date and pestle tag
    timeseries = expanded_df.groupby(["date", "pestle"]).size().unstack(fill_value=0)

    # Ensure all 6 columns exist
    for tag in STANDARD_PESTLES:
        if tag not in timeseries.columns:
            timeseries[tag] = 0

    # Sort columns consistently
    timeseries = timeseries[STANDARD_PESTLES]
    timeseries.index.name = "Date"
    return timeseries

def compute_pestle_timeseries_from_links(df: pd.DataFrame, press_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a multi-line PESTLE frequency time series by resolving 'link' indices to press_data dates.
    Returns a DataFrame with date as index and 6 PESTLE tag columns as lines.
    """

    STANDARD_PESTLES = ["political", "economic", "social", "technological", "legal", "environmental"]
    records = []

    for idx in df.index:
        # Parse press release link IDs
        link_cell = str(df.loc[idx, "link"])
        link_ids = [int(x) for x in link_cell.split(",") if x.strip().isdigit()]

        # Determine PESTLE tags in this row
        if "PESTLE" in df.columns:
            pestles = [p.strip().lower() for p in str(df.loc[idx, "PESTLE"]).split(",") if p.strip()]
        elif "PESTLE Tag1" in df.columns and "PESTLE Tag2" in df.columns:
            pestles = [
                str(df.loc[idx, "PESTLE Tag1"]).strip().lower(),
                str(df.loc[idx, "PESTLE Tag2"]).strip().lower()
            ]
        else:
            continue

        for link_id in link_ids:
            if 0 <= link_id < len(press_df):
                date = pd.to_datetime(press_df.loc[link_id, "date"], errors="coerce")
                if pd.isna(date):
                    continue
                for tag in pestles:
                    if tag in STANDARD_PESTLES:
                        records.append({"date": date.date(), "pestle": tag})

    # Aggregate to time series
    df_expanded = pd.DataFrame(records)
    if df_expanded.empty:
        return pd.DataFrame()

    timeline = df_expanded.groupby(["date", "pestle"]).size().unstack(fill_value=0)

    # Ensure all 6 columns
    for tag in STANDARD_PESTLES:
        if tag not in timeline.columns:
            timeline[tag] = 0

    return timeline[STANDARD_PESTLES].sort_index()


##############################################

st.title("🏦🌍🚨 apoBank Trends")
st.caption("🚀 Trend mapping and impact analysis for apoBank.")

# Sidebar

with st.sidebar:

    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.title("🌿 Dashboard")
    st.write("Choose the options below")
    
    region_options = ("Local", "Global", "Local+Global")
    selected_region = st.selectbox(
        label="Choose your preferred Region:",
        options=region_options,
    )

    # Dropdown to select date period
    date_options = ("Past Week", "Past Month", "Past Quarter", "Six Months", "Up to A Year")
    selected_date = st.selectbox(
        label="Choose your preferred Date Period:",
        options=date_options,
    )

    # Dropdown to select PESTLE tags
    pestle_options = ("All", "Political", "Economic", "Social", "Technological", "Legal", "Environmental")
    selected_pestle = st.selectbox(
        label="Choose your preferred PESTLE factor:",
        options=pestle_options,
    )
    
    # Retrieve the correct DataFrame based on selection
    # Only use date_region_map for Local or Global
    if selected_region in ["Local", "Global"]:
        selected_df = date_region_map[selected_region][selected_date]
        selected_df = filter_by_pestle(selected_df,selected_pestle)
        
    # Compute trend data depending on region
    if selected_region == "Local":
        pestle_counts_df = compute_pestle_counts(selected_df, selected_pestle)

    elif selected_region == "Global":
        pestle_counts_df = compute_pestle_counts(selected_df, selected_pestle)

    elif selected_region == "Local+Global":
        local_df = filter_by_pestle(date_region_map["Local"][selected_date], selected_pestle)
        global_df = filter_by_pestle(date_region_map["Global"][selected_date], selected_pestle)
        pestle_counts_df = compute_pestle_counts_by_source(local_df, global_df)

    st.subheader("📊 PESTLE Trendline Frequency")
    if not pestle_counts_df.empty:
        st.bar_chart(pestle_counts_df)
    else:
        st.info("No data to display for the selected PESTLE tag and time period.")

    if selected_region == "Local":
        selected_df = date_region_map["Local"][selected_date]
        timeline_df = compute_pestle_timeseries_from_links(selected_df, press_data)
    
    elif selected_region == "Global":
        selected_df = date_region_map["Global"][selected_date]
        timeline_df = compute_pestle_timeseries_multiline(selected_df)
    
    elif selected_region == "Local+Global":
        local_df = date_region_map["Local"][selected_date]
        global_df = date_region_map["Global"][selected_date]
    
        local_ts = compute_pestle_timeseries_from_links(local_df, press_data)
        global_ts = compute_pestle_timeseries_multiline(global_df)
    
        timeline_df = local_ts.add(global_ts, fill_value=0).sort_index()

    st.subheader("📆 Multi-PESTLE Trends Over Time")
    if not timeline_df.empty:
        st.area_chart(timeline_df, use_container_width=True)
    else:
        st.info("No time-based trend data available.")


    st.button("Refresh Data")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main Body

# Loop and display each trend and its press releases
if selected_region == "Local":
    for idx in selected_df.index:
        show_press_release_from_index(idx, selected_df, press_data)
elif selected_region == "Global":
    sorted_df = selected_df.sort_values(by="Total", ascending=False)
    for idx in sorted_df.index:
        show_global_trends(idx, sorted_df)
elif selected_region == "Local+Global":
    # Filter individually
    local_df = filter_by_pestle(date_region_map["Local"][selected_date], selected_pestle)
    global_df = filter_by_pestle(date_region_map["Global"][selected_date], selected_pestle)

    # Build sortable list of (Total, source, index)
    combined = []

    for idx in local_df.index:
        total = local_df.at[idx, "Total"]
        combined.append((total, "local", idx))

    for idx in global_df.index:
        total = global_df.at[idx, "Total"]
        combined.append((total, "global", idx))

    # Sort by Total ascending
    combined.sort(key=lambda x: x[0],reverse=True)

    # Display using the correct function
    for total, source, idx in combined:
        if source == "local":
            show_press_release_from_index(idx, local_df, press_data)
        else:
            show_global_trends(idx, global_df)

        
