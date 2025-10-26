import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime
import numpy as np
import os

# Attempt to import modern LangChain and Ollama libraries
try:
    from langchain_community.agent_toolkits.sql.base import create_sql_agent
    from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
    from langchain_community.utilities import SQLDatabase
    from langchain_ollama.chat_models import ChatOllama

    LANGCHAIN_LOCAL_AVAILABLE = True
except ImportError:
    LANGCHAIN_LOCAL_AVAILABLE = False

# --- Constants ---
DB_FILE = "call_analysis.db"


# --- Data Loading ---
@st.cache_data(ttl=20)
def load_data():
    """
    Loads data from the SQLite database and prepares it for analysis.
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query("SELECT * FROM calls", conn)
        conn.close()
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Add simulated call duration if not present for new plots
        if 'call_duration_seconds' not in df.columns:
            df['call_duration_seconds'] = np.random.randint(60, 900, size=len(df))

        # Simulate agent IDs for more granular analysis
        if 'agent_id' not in df.columns:
            agents = [f"Agent_{i:03}" for i in range(10)]
            df['agent_id'] = np.random.choice(agents, size=len(df))

        return df
    except Exception as e:
        st.error(f"Failed to load data from database: {e}")
        return pd.DataFrame()


# --- Sidebar and Filtering ---
def display_sidebar(df):
    """
    Displays the sidebar with filters and a refresh button.
    Returns the filtered DataFrame.
    """
    st.sidebar.header("Filters")

    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    if df.empty:
        st.sidebar.warning("No data to filter.")
        return df

    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()

    if min_date > max_date:
        st.sidebar.warning("Invalid date range in data.")
        return pd.DataFrame()

    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if len(date_range) != 2:
        return df

    status_options = sorted(df['resolution_status'].unique())
    selected_statuses = st.sidebar.multiselect(
        "Resolution Status",
        options=status_options,
        default=status_options
    )

    category_options = sorted(df['product_category'].unique())
    selected_categories = st.sidebar.multiselect(
        "Product Category",
        options=category_options,
        default=category_options
    )

    start_date, end_date = date_range
    filtered_df = df[
        (df['timestamp'].dt.date >= start_date) &
        (df['timestamp'].dt.date <= end_date) &
        (df['resolution_status'].isin(selected_statuses)) &
        (df['product_category'].isin(selected_categories))
        ]

    return filtered_df


# --- Helper Functions ---
def get_sentiment_label(score):
    """Converts a sentiment score to a descriptive label with an emoji."""
    if score > 0.2:
        return "Positive 😊"
    elif score < -0.2:
        return "Negative 😠"
    else:
        return "Neutral 😐"


# --- UI Components ---
def display_kpis(df):
    """Displays the Key Performance Indicators with sentiment labels."""
    st.markdown("### Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Calls Analyzed", len(df))

    if not df.empty:
        avg_customer_sentiment = df['customer_sentiment_score'].mean()
        avg_agent_sentiment = df['agent_sentiment_score'].mean()

        col2.metric("Avg. Customer Sentiment", get_sentiment_label(avg_customer_sentiment))
        col3.metric("Avg. Agent Score", get_sentiment_label(avg_agent_sentiment))
        col4.metric("Avg. Resolution Chance", f"{df['resolved_chance'].mean():.1%}")
    else:
        col2.metric("Avg. Customer Sentiment", "N/A")
        col3.metric("Avg. Agent Score", "N/A")
        col4.metric("Avg. Resolution Chance", "N/A")
    st.markdown("---")


def display_overview_charts(df):
    """Displays the primary charts for the overview tab."""
    st.header("Overview Charts")
    st.markdown("---")

    # --- Row 1: Pie and Stacked Bar Charts ---
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Resolution Status Distribution")
        # Ensure 'resolution_status' column exists and is not empty
        if 'resolution_status' in df.columns and not df['resolution_status'].empty:
            status_counts = df['resolution_status'].value_counts()
            fig = px.pie(
                status_counts,
                values=status_counts.values,
                names=status_counts.index,
                title="Call Outcomes",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True, key="res_status_pie")
        else:
            st.info("No resolution status data available.")

    with col_b:
        st.subheader("Sentiment Distribution by Resolution Status")
        # Ensure necessary columns exist and are not empty
        if 'resolution_status' in df.columns and 'customer_sentiment' in df.columns and not df.empty:
            sentiment_by_status = df.groupby(['resolution_status', 'customer_sentiment']).size().reset_index(
                name='count')
            fig_stacked_sentiment = px.bar(
                sentiment_by_status,
                x="resolution_status",
                y="count",
                color="customer_sentiment",
                title="Customer Sentiment Breakdown by Resolution Status",
                labels={'resolution_status': 'Resolution Status', 'count': 'Number of Calls',
                        'customer_sentiment': 'Sentiment'},
                color_discrete_map={'Positive': 'green', 'Neutral': 'blue', 'Negative': 'red'},
                category_orders={"customer_sentiment": ["Positive", "Neutral", "Negative"]}
            )
            st.plotly_chart(fig_stacked_sentiment, use_container_width=True, key="stacked_res_sentiment_bar")
        else:
            st.info("No data available for this chart.")

    st.markdown("---")

    # --- Row 2: Trends ---
    st.subheader("Key Trends Over Time")
    col_c, col_d = st.columns(2)

    with col_c:
        st.subheader("Daily Call Volume Trend")
        # Ensure 'timestamp' column exists
        if 'timestamp' in df.columns and not df.empty:
            calls_by_day = df.set_index('timestamp').resample('D').size().reset_index(name='call_count')
            fig_calls_day = px.line(
                calls_by_day,
                x='timestamp',
                y='call_count',
                title="Daily Call Volume",
                labels={'timestamp': 'Date', 'call_count': 'Number of Calls'},
                markers=True
            )
            st.plotly_chart(fig_calls_day, use_container_width=True, key="daily_call_volume_trend")
        else:
            st.info("No timestamp data available for this trend plot.")

    with col_d:
        st.subheader("Daily Average Sentiment Trend")
        # Ensure 'timestamp' and 'customer_sentiment_score' columns exist
        if 'timestamp' in df.columns and 'customer_sentiment_score' in df.columns and not df.empty:
            daily_sentiment = df.set_index('timestamp').resample('D')['customer_sentiment_score'].mean().reset_index()
            fig_sentiment_trend = px.line(
                daily_sentiment,
                x='timestamp',
                y='customer_sentiment_score',
                title="Daily Average Customer Sentiment",
                labels={'customer_sentiment_score': 'Avg Sentiment Score'},
                markers=True
            )
            fig_sentiment_trend.update_yaxes(range=[-2.0, 2.0])
            st.plotly_chart(fig_sentiment_trend, use_container_width=True, key="daily_sentiment_trend")
        else:
            st.info("No sentiment data available for this trend plot.")

    st.markdown("---")

    # --- Row 3: Hourly and Product Charts ---
    st.subheader("Additional Overviews")
    col_e, col_f = st.columns(2)

    with col_e:
        st.subheader("Hourly Call Distribution")
        if 'timestamp' in df.columns and not df.empty:
            df['hour'] = df['timestamp'].dt.hour
            hourly_calls = df['hour'].value_counts().sort_index()
            fig_hourly = px.bar(
                hourly_calls,
                x=hourly_calls.index,
                y=hourly_calls.values,
                title="Peak Call Times",
                labels={'x': 'Hour of Day', 'y': 'Number of Calls'}
            )
            st.plotly_chart(fig_hourly, use_container_width=True, key="hourly_call_dist")
        else:
            st.info("No timestamp data available for this plot.")

    with col_f:
        st.subheader("Call Volume by Product Category")
        if 'product_category' in df.columns and not df.empty:
            category_counts = df['product_category'].value_counts()
            fig_category = px.pie(
                category_counts,
                values=category_counts.values,
                names=category_counts.index,
                title="Product Category Distribution"
            )
            st.plotly_chart(fig_category, use_container_width=True, key="product_category_pie")
        else:
            st.info("No product category data available.")


def display_deeper_analysis_charts(df):
    """Displays charts for the deeper analysis tab."""

    # --- Row 1: Score Distributions ---
    st.subheader("Sentiment Score Distribution")
    col_c, col_d = st.columns(2)
    with col_c:
        fig_cust_kde = px.histogram(df, x="customer_sentiment_score", nbins=20, histnorm='probability density',
                                    title="Customer Sentiment Score Distribution",
                                    labels={'customer_sentiment_score': 'Score'}, color_discrete_sequence=['#FF7F0E'])
        fig_cust_kde.update_traces(marker_line_width=1, marker_line_color="white")
        st.plotly_chart(fig_cust_kde, use_container_width=True, key="cust_kde")
    with col_d:
        fig_agent_kde = px.histogram(df, x="agent_sentiment_score", nbins=20, histnorm='probability density',
                                     title="Agent Sentiment Score Distribution",
                                     labels={'agent_sentiment_score': 'Score'}, color_discrete_sequence=['#1F77B4'])
        fig_agent_kde.update_traces(marker_line_width=1, marker_line_color="white")
        st.plotly_chart(fig_agent_kde, use_container_width=True, key="agent_kde")

    st.markdown("---")

    # --- Row 2: Relationship Plots ---
    st.subheader("Relationship Between Key Metrics")
    col_e, col_f = st.columns(2)

    with col_e:
        st.subheader("Call Duration vs. Customer Sentiment")
        try:
            fig_duration_sentiment = px.scatter(df, x='call_duration_seconds', y='customer_sentiment_score',
                                                title="Duration vs. Sentiment",
                                                labels={'call_duration_seconds': 'Call Duration (s)',
                                                        'customer_sentiment_score': 'Customer Sentiment'},
                                                trendline="ols", trendline_color_override="red")
        except (ModuleNotFoundError, ImportError):
            st.warning("`statsmodels` is not installed. Trendline will not be displayed.")
            fig_duration_sentiment = px.scatter(df, x='call_duration_seconds', y='customer_sentiment_score',
                                                title="Duration vs. Sentiment",
                                                labels={'call_duration_seconds': 'Call Duration (s)',
                                                        'customer_sentiment_score': 'Customer Sentiment'})
        st.plotly_chart(fig_duration_sentiment, use_container_width=True, key="duration_sentiment")

    with col_f:
        st.subheader("Core Metric Correlation Heatmap")
        # Simplified list of core numeric columns
        numeric_cols = ['customer_sentiment_score', 'agent_sentiment_score', 'resolved_chance', 'call_duration_seconds']

        numeric_df = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        corr_df = numeric_df.corr()

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns,
            y=corr_df.columns,
            colorscale='Blues',
            colorbar=dict(title='Correlation')
        ))
        fig_heatmap.update_layout(title_text='Correlation Between Core Metrics')
        st.plotly_chart(fig_heatmap, use_container_width=True, key="heatmap_core_metrics")

    st.markdown("---")

    # --- Row 3: Correlation & Impact ---
    st.subheader("Correlation & Impact Analysis")
    col_g, col_h = st.columns(2)

    with col_g:
        st.subheader("Resolution Chance vs. Issue Complexity")
        if 'issue_complexity' in df.columns and 'resolved_chance' in df.columns:
            fig_res_box = px.box(df, x='issue_complexity', y='resolved_chance',
                                 title="Resolution Chance vs. Issue Complexity",
                                 labels={'issue_complexity': 'Issue Complexity',
                                         'resolved_chance': 'Resolution Chance'},
                                 color='issue_complexity')
            st.plotly_chart(fig_res_box, use_container_width=True, key="res_complexity_box")
        else:
            st.info("`issue_complexity` or `resolved_chance` columns not found in data.")

    with col_h:
        st.subheader("Customer Sentiment vs. Agent Sentiment")
        fig_sent_corr = px.scatter(df, x='agent_sentiment_score', y='customer_sentiment_score',
                                   title="Agent Sentiment's Impact on Customer Sentiment",
                                   labels={'agent_sentiment_score': 'Agent Sentiment Score',
                                           'customer_sentiment_score': 'Customer Sentiment Score'},
                                   trendline='ols', trendline_scope='overall')
        st.plotly_chart(fig_sent_corr, use_container_width=True, key="sent_corr")


def display_product_analysis(df):
    """Displays more insightful charts for the product analysis tab."""
    st.subheader("Product-Specific Insights")

    # Treemap for Call Volume and Sentiment
    st.markdown("#### Call Volume and Sentiment by Product")
    product_summary = df.groupby('product_category').agg(
        call_count=('call_id', 'count'),
        avg_sentiment=('customer_sentiment_score', 'mean')
    ).reset_index()

    if not product_summary.empty:
        fig_treemap = px.treemap(product_summary, path=[px.Constant("All Products"), 'product_category'],
                                 values='call_count', color='avg_sentiment',
                                 title="Product Performance Overview: Volume and Sentiment",
                                 color_continuous_scale='RdYlGn',
                                 color_continuous_midpoint=0)
        fig_treemap.update_layout(margin=dict(t=50, l=25, r=25, b=25))
        st.plotly_chart(fig_treemap, use_container_width=True, key="product_treemap")
    else:
        st.info("No data available to create a treemap.")
    st.markdown("---")

    # Stacked Bar Chart: Sentiment Distribution by Product
    st.markdown("#### Customer Sentiment Distribution by Product Category")
    sentiment_by_product = df.groupby(['product_category', 'customer_sentiment']).size().reset_index(name='count')
    if not sentiment_by_product.empty:
        fig_stacked_sentiment = px.bar(
            sentiment_by_product,
            x="product_category",
            y="count",
            color="customer_sentiment",
            title="Customer Sentiment Breakdown by Product",
            labels={'product_category': 'Product Category', 'count': 'Number of Calls',
                    'customer_sentiment': 'Sentiment'},
            color_discrete_map={'Positive': 'green', 'Neutral': 'blue', 'Negative': 'red'},
            category_orders={"customer_sentiment": ["Positive", "Neutral", "Negative"]}
        )
        st.plotly_chart(fig_stacked_sentiment, use_container_width=True, key="product_sentiment_stacked")
    else:
        st.info("No data available for this plot.")
    st.markdown("---")

    # Pie Chart: Issue Complexity for a Selected Product Category
    st.markdown("#### Issue Complexity Breakdown for a Selected Category")
    category_options = sorted(df['product_category'].unique())
    selected_category_pie = st.selectbox("Select a Product Category for Issue Complexity", options=category_options)

    if selected_category_pie:
        df_category = df[df['product_category'] == selected_category_pie]
        complexity_counts = df_category['issue_complexity'].value_counts()
        if not complexity_counts.empty:
            fig_complexity_pie = px.pie(
                values=complexity_counts.values,
                names=complexity_counts.index,
                title=f"Issue Complexity for {selected_category_pie}",
                hole=0.4
            )
            st.plotly_chart(fig_complexity_pie, use_container_width=True, key="complexity_pie")
        else:
            st.info(f"No data for issue complexity in {selected_category_pie}.")

    # Resolution Rate by Product



# In src/services/dashboard/app.py

def display_agent_analysis(df):
    """
    Displays more insightful charts for the agent analysis tab, focusing on agent-specific and hourly KPIs.
    """
    st.subheader("Agent Performance Insights")

    # --- Row 1: Agent Performance Quadrant & Individual KPI Bars ---
    st.markdown("#### Agent Performance: Sentiment vs. Resolution Rate")
    agent_summary = df.groupby('agent_id').agg(
        avg_customer_sentiment=('customer_sentiment_score', 'mean'),
        call_count=('call_id', 'count')
    ).reset_index()

    resolved_counts = df[df['resolution_status'] == 'Fully Resolved'].groupby('agent_id').size().reset_index(
        name='resolved_count')

    agent_summary = pd.merge(agent_summary, resolved_counts, on='agent_id', how='left').fillna(0)
    agent_summary['resolution_rate'] = (agent_summary['resolved_count'] / agent_summary['call_count']) * 100

    avg_res_rate = agent_summary['resolution_rate'].mean()
    avg_sent = agent_summary['avg_customer_sentiment'].mean()

    col_quadrant, col_kpi_bars = st.columns(2)

    with col_quadrant:
        fig_agent_quad = px.scatter(agent_summary, x='resolution_rate', y='avg_customer_sentiment',
                                    size='call_count', color='avg_customer_sentiment',
                                    hover_name='agent_id',
                                    title="Agent Performance Quadrant",
                                    labels={'resolution_rate': 'Resolution Rate (%)',
                                            'avg_customer_sentiment': 'Average Customer Sentiment'})
        fig_agent_quad.add_vline(x=avg_res_rate, line_width=2, line_dash="dash", line_color="gray",
                                 annotation_text="Avg. Resolution")
        fig_agent_quad.add_hline(y=avg_sent, line_width=2, line_dash="dash", line_color="gray",
                                 annotation_text="Avg. Sentiment")
        st.plotly_chart(fig_agent_quad, use_container_width=True, key="agent_quad_chart")

    with col_kpi_bars:
        st.markdown("#### Agent Performance Bar Charts")
        fig_agent_sentiment_bar = px.bar(agent_summary, x='agent_id', y='avg_customer_sentiment',
                                         title="Avg. Customer Sentiment per Agent",
                                         labels={'avg_customer_sentiment': 'Avg Sentiment Score'},
                                         color='avg_customer_sentiment',
                                         color_continuous_scale=px.colors.sequential.RdBu,
                                         range_y=[-2.0, 2.0])
        st.plotly_chart(fig_agent_sentiment_bar, use_container_width=True, key="agent_sentiment_bar")

        fig_agent_resolved_bar = px.bar(agent_summary, x='agent_id', y='resolution_rate',
                                        title="Avg. Resolution Rate per Agent",
                                        labels={'resolution_rate': 'Resolution Rate (%)'},
                                        color='resolution_rate',
                                        color_continuous_scale=px.colors.sequential.Greens,
                                        range_y=[0.0, 100.0])
        st.plotly_chart(fig_agent_resolved_bar, use_container_width=True, key="agent_resolved_bar")

    st.markdown("---")
    st.subheader("Hourly KPIs and Performance Trends")

    # Dropdown for selecting an agent to analyze
    all_agents_for_trend = sorted(df['agent_id'].dropna().unique().tolist())
    selected_agent_for_trend = st.selectbox("Select an Agent to view Hourly Trends", options=all_agents_for_trend)

    if selected_agent_for_trend:
        df_agent = df[df['agent_id'] == selected_agent_for_trend].copy()

        if not df_agent.empty:
            df_agent['hour'] = df_agent['timestamp'].dt.hour
            hourly_kpis = df_agent.groupby('hour').agg(
                calls_handled=('call_id', 'count'),
                avg_sentiment=('customer_sentiment_score', 'mean'),
            ).reset_index()

            col_trend_a, col_trend_b = st.columns(2)

            with col_trend_a:
                st.markdown(f"#### Hourly Call Volume for {selected_agent_for_trend}")
                fig_hourly_vol = px.bar(
                    hourly_kpis,
                    x='hour',
                    y='calls_handled',
                    title="Calls Handled per Hour",
                    labels={'hour': 'Hour of Day', 'calls_handled': 'Calls Handled'},
                    color='calls_handled',
                    color_continuous_scale=px.colors.sequential.Blues
                )
                st.plotly_chart(fig_hourly_vol, use_container_width=True, key="agent_hourly_vol")

            with col_trend_b:
                st.markdown(f"#### Hourly Sentiment Trend for {selected_agent_for_trend}")
                fig_hourly_perf = go.Figure()
                fig_hourly_perf.add_trace(go.Scatter(
                    x=hourly_kpis['hour'], y=hourly_kpis['avg_sentiment'], mode='lines+markers', name='Avg Sentiment',
                    yaxis='y1'
                ))
                fig_hourly_perf.update_layout(
                    title="Hourly Performance Trend",
                    xaxis_title="Hour of Day",
                    yaxis=dict(title='Avg Sentiment', range=[-2.0, 2.0]),
                    hovermode="x unified"
                )
                st.plotly_chart(fig_hourly_perf, use_container_width=True, key="agent_hourly_perf_trend")

        else:
            st.info(f"No data available for agent {selected_agent_for_trend}.")

    st.markdown("---")

    # Call Volume by Experience
    st.markdown("#### Call Volume Handled by Agent Experience Level")
    calls_by_exp = df['agent_experience_level'].value_counts().reset_index()
    calls_by_exp.columns = ['agent_experience_level', 'call_count']
    fig_vol_exp = px.pie(calls_by_exp, values='call_count', names='agent_experience_level',
                         title="Share of Calls Handled by Experience Level", hole=0.4)
    st.plotly_chart(fig_vol_exp, use_container_width=True, key="call_vol_exp_pie")

    st.markdown("---")

    # Customer vs Agent Sentiment Correlation
    st.markdown("#### Customer Sentiment vs. Agent Sentiment")
    fig_sent_corr = px.scatter(df, x='agent_sentiment_score', y='customer_sentiment_score',
                               color='agent_experience_level',
                               title="Agent Sentiment's Impact on Customer Sentiment",
                               labels={'agent_sentiment_score': 'Agent Sentiment Score',
                                       'customer_sentiment_score': 'Customer Sentiment Score'},
                               trendline='ols', trendline_scope='overall')
    st.plotly_chart(fig_sent_corr, use_container_width=True, key="sent_corr_scatter")


def display_detailed_view(df):
    """Displays the detailed call log and a drill-down option."""
    st.subheader("Detailed Call Log")
    st.dataframe(df.drop(columns=['hour'], errors='ignore'), use_container_width=True)

    st.subheader("Drill Down into a Specific Call")
    if not df.empty:
        all_filenames = df['filename'].unique()
        selected_filename = st.selectbox("Select a Filename to view full details", all_filenames)
        if selected_filename:
            call_details_series = df[df['filename'] == selected_filename].iloc[0]
            call_details_dict = call_details_series.to_dict()

            for key, value in call_details_dict.items():
                if isinstance(value, pd.Timestamp):
                    call_details_dict[key] = value.isoformat()
                elif isinstance(value, np.generic):
                    call_details_dict[key] = value.item()

            with st.expander(f"Full Transcript and Analysis for {selected_filename}"):
                st.json(call_details_dict)
    else:
        st.write("No call details to display for the selected filters.")


def display_chatbot():
    """
    Displays an optimized chatbot interface for querying the database using a local LLM.
    This version uses a direct SQL chain for faster, more reliable query generation and adds
    guardrails to prevent random or incorrect answers.
    """
    st.subheader("Chat with your Data (Powered by Local LLM)")

    if not LANGCHAIN_LOCAL_AVAILABLE:
        st.error(
            "Local LLM libraries are not installed. Please run `pip install langchain-community langchain-ollama` to use the chatbot.")
        return

    # Using a cache for the database connection and LLM to avoid re-initializing
    @st.cache_resource
    def get_llm_and_db():
        try:
            db = SQLDatabase.from_uri(f"sqlite:///{DB_FILE}")
            # For even faster performance, consider smaller, specialized models.
            # For example: ollama pull sqlcoder:7b or ollama pull duckdb-nsql
            llm = ChatOllama(model="llama3.1:8b")
            return llm, db
        except Exception as e:
            st.error(f"Failed to initialize resources. Is Ollama running? Error: {e}")
            return None, None

    llm, db = get_llm_and_db()
    if not llm or not db:
        return

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about your call data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.status("The agent is thinking...", expanded=True) as status:
                try:
                    # Import necessary LangChain components
                    from langchain_core.prompts import ChatPromptTemplate
                    from langchain_core.output_parsers import StrOutputParser
                    from langchain_core.runnables import RunnablePassthrough
                    # Corrected import path for create_sql_query_chain
                    from langchain.chains import create_sql_query_chain

                    # Step 1: Create a chain to generate the SQL query
                    status.update(label="Generating SQL query...")

                    # This chain is specifically designed for creating SQL queries.
                    write_query = create_sql_query_chain(llm, db)
                    generated_text = write_query.invoke({"question": prompt})

                    # **FIX**: Clean up the generated text to extract only the SQL query
                    if "SQLQuery:" in generated_text:
                        sql_query = generated_text.split("SQLQuery:")[-1].strip()
                    else:
                        sql_query = generated_text

                    # Remove potential markdown formatting
                    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

                    status.update(label="Executing SQL query...")
                    status.info(f"**Generated SQL Query:**\n```sql\n{sql_query}\n```")

                    # Step 2: Execute the SQL query and get the result
                    try:
                        query_result = db.run(sql_query)
                        status.update(label="Analyzing results...")
                        status.info(f"**Query Result:**\n```\n{query_result}\n```")
                    except Exception as e:
                        # If the query itself is invalid, inform the user.
                        final_answer = f"I tried to run a query, but it failed with the following error: {e}. Please try rephrasing your question."
                        st.error(final_answer)
                        st.session_state.messages.append({"role": "assistant", "content": final_answer})
                        status.update(label="Query Failed", state="error", expanded=False)
                        return

                    # Guardrail: Check if the query returned any results.
                    if not query_result or query_result == "[]":
                        final_answer = "I ran a query, but it returned no results. Please try a different question."
                        st.warning(final_answer)
                        st.session_state.messages.append({"role": "assistant", "content": final_answer})
                        status.update(label="No results found", state="complete", expanded=False)
                        return

                    # Step 3: Create a chain to summarize the result into a natural language answer
                    status.update(label="Summarizing the answer...")
                    answer_prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system",
                             "You are an assistant who answers user questions based on SQL query results. "
                             "Given the user's question and the corresponding SQL result, write a clear, natural language answer. "
                             "Base your answer ONLY on the provided SQL result. Do not make up information."),
                            ("user", "Question: {question}\nSQL Result: {result}"),
                        ]
                    )

                    answer_chain = answer_prompt | llm | StrOutputParser()
                    final_answer = answer_chain.invoke({"question": prompt, "result": query_result})

                    status.update(label="Query complete!", state="complete", expanded=False)
                    st.markdown(final_answer)
                    st.session_state.messages.append({"role": "assistant", "content": final_answer})

                except Exception as e:
                    error_message = f"An unexpected error occurred: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    status.update(label="Error", state="error")


def main():
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(
        page_title="Unified Call Analytics Platform",
        page_icon="📞",
        layout="wide"
    )

    st.title("📞 Unified Call Analytics Platform")
    st.markdown("Live analysis of customer support calls. Use the sidebar to filter data.")

    df = load_data()

    if df.empty:
        st.warning("No data found. Please run the audio processing script to populate the database.")
        display_sidebar(df)
        return

    filtered_df = display_sidebar(df)

    if filtered_df.empty and not df.empty:
        st.warning("No data matches the selected filters.")

    display_kpis(filtered_df)

    # --- Tabbed Layout ---
    tab_titles = [
        "📊 Dashboard Overview",
        "📦 Product Analysis",
        "🧑‍💼 Agent Analysis",
        "🧠 Deeper Analysis",
        "📋 Detailed Call Log",
        "🤖 Chatbot"
    ]
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_titles)

    with tab1:
        if not filtered_df.empty:
            display_overview_charts(filtered_df)
        else:
            st.info("No data for the selected filters to display charts.")

    with tab2:
        if not filtered_df.empty:
            display_product_analysis(filtered_df)
        else:
            st.info("No data for the selected filters to display charts.")

    with tab3:
        if not filtered_df.empty:
            display_agent_analysis(filtered_df)
        else:
            st.info("No data for the selected filters to display charts.")

    with tab4:
        if not filtered_df.empty:
            display_deeper_analysis_charts(filtered_df)
        else:
            st.info("No data for the selected filters to display charts.")

    with tab5:
        if not filtered_df.empty:
            display_detailed_view(filtered_df)
        else:
            st.info("No data for the selected filters to display details.")

    with tab6:
        display_chatbot()


if __name__ == "__main__":
    main()
