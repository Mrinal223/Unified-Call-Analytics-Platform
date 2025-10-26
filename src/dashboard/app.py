import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import json
import time
from datetime import datetime, timedelta
import os
import glob
import psycopg2

st.set_page_config(page_title="Unified Call Analytics Platform", page_icon="📞", layout="wide", initial_sidebar_state="expanded")

# --- Configuration (using relative paths from project root) ---
DB_FILE = "../../data/call_analysis.db"
POSTGRES_HOST = "localhost"
POSTGRES_DB = "call_analytics_db"
POSTGRES_USER = "user"
POSTGRES_PASSWORD = "password"
POSTGRES_PORT = "5432"

# --- Data Loading ---
@st.cache_data(ttl=10)
def load_data_from_sqlite():
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query("SELECT * FROM calls", conn)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['products_mentioned'] = df['products_mentioned'].apply(
            lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else []
        )
        df['customer_sentiment_score'] = pd.to_numeric(df['customer_sentiment_score'], errors='coerce').fillna(0.0)
        df['agent_sentiment_score'] = pd.to_numeric(df['agent_sentiment_score'], errors='coerce').fillna(0.0)
        df['resolved_chance'] = pd.to_numeric(df['resolved_chance'], errors='coerce').fillna(0.0)
        df['problem_resolved'] = df['problem_resolved'].astype(bool)
        df['product_name_display'] = df['products_mentioned'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) and x else 'N/A'
        )
        return df
    except Exception as e:
        st.error(f"Failed to load data from SQLite database: {e}")
        return pd.DataFrame()
    finally:
        if conn: conn.close()

# --- Page Layout and Logic ---
st.title("📞 Unified Call Analytics Platform")
st.markdown("Live analysis of customer support calls.")

df_raw = load_data_from_sqlite()

if df_raw.empty:
    st.warning("No data found. Please run the audio processing script to populate the database.")
else:
    st.sidebar.header("Filters")
    df_valid_timestamps = df_raw.dropna(subset=['timestamp'])
    if not df_valid_timestamps.empty:
        min_date = df_valid_timestamps['timestamp'].min().date()
        max_date = df_valid_timestamps['timestamp'].max().date()
    else:
        min_date = datetime.now().date()
        max_date = datetime.now().date()
    date_range = st.sidebar.date_input("Select Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if len(date_range) == 2:
        start_date = datetime.combine(date_range[0], datetime.min.time())
        end_date = datetime.combine(date_range[1], datetime.max.time())
        df_filtered = df_raw[(df_raw['timestamp'] >= start_date) & (df_raw['timestamp'] <= end_date)].copy()
    else: df_filtered = df_raw.copy()

    all_agents = ['All'] + sorted(df_filtered['agent_id'].dropna().unique().tolist())
    selected_agent = st.sidebar.selectbox("Filter by Agent ID", options=all_agents)
    if selected_agent != 'All': df_filtered = df_filtered[df_filtered['agent_id'] == selected_agent]

    all_exp_levels = ['All'] + sorted(df_filtered['agent_experience_level'].dropna().unique().tolist())
    selected_exp_level = st.sidebar.selectbox("Filter by Agent Experience", options=all_exp_levels)
    if selected_exp_level != 'All': df_filtered = df_filtered[df_filtered['agent_experience_level'] == selected_exp_level]

    all_resolution_statuses = ['All'] + sorted(df_filtered['resolution_status'].dropna().unique().tolist())
    selected_resolution_status = st.sidebar.selectbox("Filter by Resolution Status", options=all_resolution_statuses)
    if selected_resolution_status != 'All': df_filtered = df_filtered[df_filtered['resolution_status'] == selected_resolution_status]

    st.markdown("### Key Performance Indicators (Filtered Data)")
    if not df_filtered.empty:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Calls Analyzed", len(df_filtered))
        col2.metric("Avg. Customer Sentiment", f"{df_filtered['customer_sentiment_score'].mean():.2f}")
        col3.metric("Avg. Agent Score", f"{df_filtered['agent_sentiment_score'].mean():.2f}")
        col4.metric("Avg. Resolution Chance", f"{df_filtered['resolved_chance'].mean():.1%}")
        actual_resolution_rate = df_filtered['problem_resolved'].mean() * 100
        col5.metric("Actual Resolution Rate", f"{actual_resolution_rate:.1f}%")
    else: st.info("No data available for the selected filters.")

    st.markdown("---")

    if not df_filtered.empty:
        st.header("Visualizations")
        col_viz1, col_viz2 = st.columns(2)
        with col_viz1:
            st.subheader("Customer Sentiment Distribution")
            sentiment_counts = df_filtered['customer_sentiment'].value_counts()
            fig_sentiment = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title="Customer Sentiment Breakdown")
            st.plotly_chart(fig_sentiment, use_container_width=True)
        with col_viz2:
            st.subheader("Problem Resolution Status")
            status_counts = df_filtered['resolution_status'].value_counts()
            fig_status = px.bar(x=status_counts.index, y=status_counts.values, title="Call Resolution Status", labels={'x': 'Status', 'y': 'Number of Calls'}, color=status_counts.index, color_discrete_map={'Fully Resolved': 'green', 'Partially Resolved': 'orange', 'Escalated': 'red', 'In Progress': 'blue', 'Requires Follow-up': 'purple'})
            st.plotly_chart(fig_status, use_container_width=True)
        st.markdown("---")

        st.header("Score Distributions (KDE-like Plots)")
        col_kde1, col_kde2 = st.columns(2)
        with col_kde1:
            st.subheader("Customer Sentiment Score Distribution")
            fig_cust_kde = px.histogram(df_filtered, x="customer_sentiment_score", nbins=20, histnorm='probability density', title="Customer Sentiment Score Distribution", labels={'customer_sentiment_score': 'Score'}, color_discrete_sequence=['#FF7F0E'])
            fig_cust_kde.update_traces(marker_line_width=1, marker_line_color="white")
            st.plotly_chart(fig_cust_kde, use_container_width=True)
        with col_kde2:
            st.subheader("Agent Sentiment Score Distribution")
            fig_agent_kde = px.histogram(df_filtered, x="agent_sentiment_score", nbins=20, histnorm='probability density', title="Agent Sentiment Score Distribution", labels={'agent_sentiment_score': 'Score'}, color_discrete_sequence=['#1F77B4'])
            fig_agent_kde.update_traces(marker_line_width=1, marker_line_color="white")
            st.plotly_chart(fig_agent_kde, use_container_width=True)
        st.markdown("---")

        col_viz3, col_viz4 = st.columns(2)
        with col_viz3:
            st.subheader("Calls by Issue Complexity")
            complexity_counts = df_filtered['issue_complexity'].value_counts()
            fig_complexity = px.bar(x=complexity_counts.index, y=complexity_counts.values, title="Issue Complexity Breakdown", labels={'x': 'Complexity Level', 'y': 'Number of Calls'}, color=complexity_counts.index)
            st.plotly_chart(fig_complexity, use_container_width=True)
        with col_viz4:
            st.subheader("Call Volume by Product Category")
            category_counts = df_filtered['product_category'].value_counts()
            fig_category = px.bar(x=category_counts.index, y=category_counts.values, title="Top Product Categories by Calls", labels={'x': 'Product Category', 'y': 'Number of Calls'}, color=category_counts.index)
            st.plotly_chart(fig_category, use_container_width=True)
        st.markdown("---")

        col_viz5, col_viz6 = st.columns(2)
        with col_viz5:
            st.subheader("Top 10 Most Frequent Issues Raised")
            issue_counts = df_filtered['issue_description'].value_counts().nlargest(10)
            fig_issue = px.bar(x=issue_counts.index, y=issue_counts.values, title="Most Common Issues", labels={'x': 'Issue Description', 'y': 'Number of Calls'}, color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_issue, use_container_width=True)
        with col_viz6:
            st.subheader("Top 10 Most Mentioned Products")
            all_products = [product for sublist in df_filtered['products_mentioned'] for product in sublist]
            if all_products:
                product_counts = pd.Series(all_products).value_counts().nlargest(10)
                fig_product = px.bar(x=product_counts.index, y=product_counts.values, title="Most Mentioned Products", labels={'x': 'Product Name', 'y': 'Number of Mentions'}, color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig_product, use_container_width=True)
            else: st.info("No specific products mentioned in the filtered data.")
        st.markdown("---")

        col_viz7, col_viz8 = st.columns(2)
        with col_viz7:
            st.subheader("Customer Sentiment Trend Over Time")
            sentiment_trend = df_filtered.set_index('timestamp').resample('D')['customer_sentiment_score'].mean().reset_index()
            fig_sentiment_trend = px.line(sentiment_trend, x='timestamp', y='customer_sentiment_score', title="Daily Average Customer Sentiment", labels={'customer_sentiment_score': 'Avg Sentiment Score'}, markers=True)
            fig_sentiment_trend.update_yaxes(range=[-2.0, 2.0])
            st.plotly_chart(fig_sentiment_trend, use_container_width=True)
        with col_viz8:
            st.subheader("Agent Performance Overview")
            agent_performance = df_filtered.groupby('agent_id').agg(avg_sentiment=('customer_sentiment_score', 'mean'), avg_resolved_chance=('resolved_chance', 'mean'), total_calls=('call_id', 'count')).reset_index()
            agent_performance = agent_performance[agent_performance['total_calls'] >= 5]
            if not agent_performance.empty:
                fig_agent_sentiment = px.bar(agent_performance, x='agent_id', y='avg_sentiment', title="Avg. Customer Sentiment per Agent", labels={'avg_sentiment': 'Avg Sentiment Score'}, color='avg_sentiment', color_continuous_scale=px.colors.sequential.RdBu, range_y=[-2.0, 2.0])
                st.plotly_chart(fig_agent_sentiment, use_container_width=True)
                fig_agent_resolved = px.bar(agent_performance, x='agent_id', y='avg_resolved_chance', title="Avg. Resolution Chance per Agent", labels={'avg_resolved_chance': 'Avg Resolution Chance'}, color='avg_resolved_chance', color_continuous_scale=px.colors.sequential.Greens, range_y=[0.0, 1.0])
                st.plotly_chart(fig_agent_resolved, use_container_width=True)
            else: st.info("No agents with sufficient call volume in the filtered data for performance overview.")
        st.markdown("---")
        st.subheader("Detailed Call Log")
        st.write("Browse and search all analyzed calls based on filters.")
        st.dataframe(df_filtered[[
            'timestamp', 'filename', 'agent_id', 'customer_id', 'call_type',
            'call_summary_one_line', 'issue_description', 'product_category', 'product_name_display',
            'customer_sentiment', 'customer_sentiment_score', 'agent_sentiment', 'agent_sentiment_score',
            'resolution_status', 'problem_resolved', 'resolved_chance', 'issue_complexity', 'duration_seconds'
        ]].set_index('call_id'), use_container_width=True)
        st.subheader("Drill Down into a Specific Call")
        if not df_filtered.empty:
            selected_call_id = st.selectbox("Select a Call ID to view full details", df_filtered['call_id'].unique())
            if selected_call_id:
                call_details = df_filtered[df_filtered['call_id'] == selected_call_id].iloc[0]
                with st.expander(f"Full Transcript and Analysis for {call_details['filename']}"):
                    st.json(call_details.to_dict())
        else: st.info("No calls to drill down into with current filters.")
    else: st.warning("No data available for the selected filters to display visualizations.")
    refresh_interval = 20
    with st.sidebar:
        st.header("⚙️ Live Refresh")
        placeholder = st.empty()
        for seconds in range(refresh_interval, 0, -1):
            placeholder.metric("Next refresh in...", f"{seconds} s")
            time.sleep(1)
        st.rerun()