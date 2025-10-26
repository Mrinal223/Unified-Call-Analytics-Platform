import os
import json
import sqlite3
import wave
import vosk
import requests
import tempfile
from pydub import AudioSegment
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go  # For more complex plots if needed
import json  # To parse product_name_options
from datetime import datetime, timedelta

# --- Configuration ---
DB_FILE = "../../call_analysis.db"  # Adjusted path to project root for the SQLite DB


# --- Data Loading ---
@st.cache_data(ttl=10)  # Cache data for 10 seconds to avoid constant re-loading
def load_data():
    """Loads data from the SQLite database into a pandas DataFrame."""
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query("SELECT * FROM calls", conn)

        # Convert timestamp column to datetime objects
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Convert products_mentioned (JSON string) back to a list
        # Handle cases where it might be NaN or not a valid JSON string
        df['products_mentioned'] = df['products_mentioned'].apply(
            lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else []
        )

        # Ensure numeric types are correct
        df['customer_sentiment_score'] = pd.to_numeric(df['customer_sentiment_score'], errors='coerce').fillna(0.0)
        df['agent_sentiment_score'] = pd.to_numeric(df['agent_sentiment_score'], errors='coerce').fillna(0.0)
        df['resolved_chance'] = pd.to_numeric(df['resolved_chance'], errors='coerce').fillna(0.0)
        df['problem_resolved'] = df['problem_resolved'].astype(bool)  # Ensure boolean type

        return df
    except FileNotFoundError:
        st.error(f"Database file not found at {DB_FILE}. Please ensure the audio processing script has run.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to load data from database: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()


# --- Page Configuration ---
st.set_page_config(
    page_title="Call Center Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Call Center Analytics Dashboard")
st.markdown("Live analysis of customer support calls.")

# --- Load Data ---
df_raw = load_data()

if df_raw.empty:
    st.warning("No data found. Please ensure the `local_audio_processor.py` script is running and processing files.")
else:
    # --- Sidebar Filters ---
    st.sidebar.header("Filters")

    # Date Range Filter
    min_date = df_raw['timestamp'].min().date()
    max_date = df_raw['timestamp'].max().date()

    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if len(date_range) == 2:
        start_date = datetime.combine(date_range[0], datetime.min.time())
        end_date = datetime.combine(date_range[1], datetime.max.time())
        df_filtered = df_raw[(df_raw['timestamp'] >= start_date) & (df_raw['timestamp'] <= end_date)].copy()
    else:  # If only one date is selected initially
        df_filtered = df_raw.copy()

    # Agent ID Filter
    all_agents = ['All'] + sorted(df_filtered['agent_id'].dropna().unique().tolist())
    selected_agent = st.sidebar.selectbox("Filter by Agent ID", options=all_agents)
    if selected_agent != 'All':
        df_filtered = df_filtered[df_filtered['agent_id'] == selected_agent]

    # Product Category Filter
    all_categories = ['All'] + sorted(df_filtered['product_category'].dropna().unique().tolist())
    selected_category = st.sidebar.selectbox("Filter by Product Category", options=all_categories)
    if selected_category != 'All':
        df_filtered = df_filtered[df_filtered['product_category'] == selected_category]

    # --- Top Level KPIs ---
    st.markdown("### Key Performance Indicators (Filtered Data)")
    if not df_filtered.empty:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Calls Analyzed", len(df_filtered))
        col2.metric("Avg. Customer Sentiment", f"{df_filtered['customer_sentiment_score'].mean():.2f}")
        col3.metric("Avg. Agent Score", f"{df_filtered['agent_sentiment_score'].mean():.2f}")
        col4.metric("Avg. Resolution Chance", f"{df_filtered['resolved_chance'].mean():.1%}")

        # Calculate actual resolution rate (problem_resolved)
        actual_resolution_rate = df_filtered['problem_resolved'].mean() * 100
        col5.metric("Actual Resolution Rate", f"{actual_resolution_rate:.1f}%")

    else:
        st.info("No data available for the selected filters.")

    st.markdown("---")

    # --- Visualizations ---
    if not df_filtered.empty:
        st.header("Visualizations")

        # Row 1: Sentiment & Resolution
        col_viz1, col_viz2 = st.columns(2)

        with col_viz1:
            st.subheader("Customer Sentiment Distribution")
            sentiment_counts = df_filtered['customer_sentiment'].value_counts()
            fig_sentiment = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Customer Sentiment Breakdown"
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)

        with col_viz2:
            st.subheader("Problem Resolution Status")
            status_counts = df_filtered['resolution_status'].value_counts()
            fig_status = px.bar(
                x=status_counts.index,
                y=status_counts.values,
                title="Call Resolution Status",
                labels={'x': 'Status', 'y': 'Number of Calls'},
                color=status_counts.index,  # Color by status
                color_discrete_map={'Fully Resolved': 'green', 'Partially Resolved': 'orange', 'Escalated': 'red',
                                    'In Progress': 'blue', 'Requires Follow-up': 'purple'}
            )
            st.plotly_chart(fig_status, use_container_width=True)

        st.markdown("---")

        # Row 2: Issue & Product Category
        col_viz3, col_viz4 = st.columns(2)

        with col_viz3:
            st.subheader("Calls by Issue Complexity")
            complexity_counts = df_filtered['issue_complexity'].value_counts()
            fig_complexity = px.bar(
                x=complexity_counts.index,
                y=complexity_counts.values,
                title="Issue Complexity Breakdown",
                labels={'x': 'Complexity Level', 'y': 'Number of Calls'},
                color=complexity_counts.index  # Color by complexity
            )
            st.plotly_chart(fig_complexity, use_container_width=True)

        with col_viz4:
            st.subheader("Call Volume by Product Category")
            category_counts = df_filtered['product_category'].value_counts()
            fig_category = px.bar(
                x=category_counts.index,
                y=category_counts.values,
                title="Top Product Categories by Calls",
                labels={'x': 'Product Category', 'y': 'Number of Calls'},
                color=category_counts.index  # Color by category
            )
            st.plotly_chart(fig_category, use_container_width=True)

        st.markdown("---")

        # Row 3: Top Issues & Products
        col_viz5, col_viz6 = st.columns(2)

        with col_viz5:
            st.subheader("Top 10 Most Frequent Issues Raised")
            issue_counts = df_filtered['issue_description'].value_counts().nlargest(10)
            fig_issue = px.bar(
                x=issue_counts.index,
                y=issue_counts.values,
                title="Most Common Issues",
                labels={'x': 'Issue Description', 'y': 'Number of Calls'},
                color_discrete_sequence=px.colors.qualitative.Pastel  # Different color palette
            )
            st.plotly_chart(fig_issue, use_container_width=True)

        with col_viz6:
            st.subheader("Top 10 Most Mentioned Products")
            # Flatten the list of products mentioned per call and count
            all_products = [product for sublist in df_filtered['products_mentioned'] for product in sublist]
            if all_products:
                product_counts = pd.Series(all_products).value_counts().nlargest(10)
                fig_product = px.bar(
                    x=product_counts.index,
                    y=product_counts.values,
                    title="Most Mentioned Products",
                    labels={'x': 'Product Name', 'y': 'Number of Mentions'},
                    color_discrete_sequence=px.colors.qualitative.Set2  # Another color palette
                )
                st.plotly_chart(fig_product, use_container_width=True)
            else:
                st.info("No specific products mentioned in the filtered data.")

        st.markdown("---")

        # Row 4: Sentiment Trend & Agent Performance
        col_viz7, col_viz8 = st.columns(2)

        with col_viz7:
            st.subheader("Customer Sentiment Trend Over Time")
            # Resample data to daily average sentiment
            sentiment_trend = df_filtered.set_index('timestamp').resample('D')[
                'customer_sentiment_score'].mean().reset_index()
            fig_sentiment_trend = px.line(
                sentiment_trend,
                x='timestamp',
                y='customer_sentiment_score',
                title="Daily Average Customer Sentiment",
                labels={'customer_sentiment_score': 'Avg Sentiment Score'},
                markers=True  # Show markers on line
            )
            fig_sentiment_trend.update_yaxes(range=[-2.0, 2.0])  # Ensure consistent y-axis for sentiment
            st.plotly_chart(fig_sentiment_trend, use_container_width=True)

        with col_viz8:
            st.subheader("Agent Performance Overview")
            agent_performance = df_filtered.groupby('agent_id').agg(
                avg_sentiment=('customer_sentiment_score', 'mean'),
                avg_resolved_chance=('resolved_chance', 'mean'),
                total_calls=('call_id', 'count')
            ).reset_index()

            # Filter out agents with very few calls for meaningful averages
            agent_performance = agent_performance[
                agent_performance['total_calls'] >= 5]  # Only show agents with >=5 calls

            if not agent_performance.empty:
                # Plot Average Sentiment per Agent
                fig_agent_sentiment = px.bar(
                    agent_performance,
                    x='agent_id',
                    y='avg_sentiment',
                    title="Avg. Customer Sentiment per Agent",
                    labels={'avg_sentiment': 'Avg Sentiment Score'},
                    color='avg_sentiment',  # Color based on sentiment score
                    color_continuous_scale=px.colors.sequential.RdBu,  # Red-Blue scale
                    range_y=[-2.0, 2.0]
                )
                st.plotly_chart(fig_agent_sentiment, use_container_width=True)

                # Plot Average Resolution Chance per Agent
                fig_agent_resolved = px.bar(
                    agent_performance,
                    x='agent_id',
                    y='avg_resolved_chance',
                    title="Avg. Resolution Chance per Agent",
                    labels={'avg_resolved_chance': 'Avg Resolution Chance'},
                    color='avg_resolved_chance',  # Color based on resolution chance
                    color_continuous_scale=px.colors.sequential.Greens,  # Green scale
                    range_y=[0.0, 1.0]
                )
                st.plotly_chart(fig_agent_resolved, use_container_width=True)
            else:
                st.info("No agents with sufficient call volume in the filtered data for performance overview.")

        st.markdown("---")

        # --- Detailed View ---
        st.subheader("Detailed Call Log")
        st.write("Browse and search all analyzed calls based on filters.")

        # Display a subset of columns for readability
        st.dataframe(df_filtered[[
            'timestamp', 'filename', 'agent_id', 'customer_id', 'call_type',
            'call_summary_one_line', 'issue_description', 'product_category', 'product_name_display',
            'customer_sentiment', 'customer_sentiment_score', 'agent_sentiment', 'agent_sentiment_score',
            'resolution_status', 'problem_resolved', 'resolved_chance', 'issue_complexity', 'duration_seconds'
        ]].set_index('call_id'))

        # Expander to see full details of a selected call
        st.subheader("Drill Down into a Specific Call")
        # Ensure selected_call_id is from the filtered DataFrame
        if not df_filtered.empty:
            selected_call_id = st.selectbox("Select a Call ID to view full details", df_filtered['call_id'].unique())
            if selected_call_id:
                call_details = df_filtered[df_filtered['call_id'] == selected_call_id].iloc[0]
                with st.expander(f"Full Transcript and Analysis for {call_details['filename']}"):
                    # Convert pandas Series to dictionary for json display
                    st.json(call_details.to_dict())
        else:
            st.info("No calls to drill down into with current filters.")
    else:
        st.warning("No data available for the selected filters to display visualizations.")
load_dotenv()

# Configuration
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "../../vosk_model/vosk-model-small-en-us-0.15")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
AUDIO_FILES_DIR = "../../audio_files_source"
DB_FILE = "call_analysis.db"

def initialize_vosk():
    """Initialize Vosk model."""
    if not os.path.exists(VOSK_MODEL_PATH):
        print(f"❌ Vosk model not found at {VOSK_MODEL_PATH}")
        return None

    vosk.SetLogLevel(-1)
    model = vosk.Model(VOSK_MODEL_PATH)
    print(f"✅ Vosk model loaded from {VOSK_MODEL_PATH}")
    return model

def convert_audio_for_vosk(audio_path):
    """Convert audio file to WAV format suitable for Vosk (16kHz, mono)."""
    try:
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_frame_rate(16000)

        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio.export(temp_wav.name, format="wav")
        return temp_wav.name
    except Exception as e:
        print(f"❌ Error converting audio: {str(e)}")
        return None

def transcribe_audio(vosk_model, audio_path):
    """Transcribe audio file to text using Vosk."""
    wav_path = None
    try:
        print(f"  🔄 Converting audio format...")
        wav_path = convert_audio_for_vosk(audio_path)
        if not wav_path:
            return None

        print(f"  🔄 Opening audio file...")
        wf = wave.open(wav_path, 'rb')

        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            print("  ❌ Audio format incompatible")
            wf.close()
            return None

        print(f"  🔄 Starting transcription...")
        rec = vosk.KaldiRecognizer(vosk_model, wf.getframerate())
        transcript_parts = []

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                if result.get('text'):
                    transcript_parts.append(result['text'])

        final_result = json.loads(rec.FinalResult())
        if final_result.get('text'):
            transcript_parts.append(final_result['text'])

        wf.close()
        full_transcript = ' '.join(transcript_parts).strip()

        return full_transcript if full_transcript else "No speech detected"

    except Exception as e:
        print(f"  ❌ Error transcribing: {str(e)}")
        return None
    finally:
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)

def analyze_transcript_with_ollama_new(transcript):
    """Analyze transcript using Ollama with improved prompt and parameters."""

    prompt = f"""
        You are an expert customer service quality analyst. Analyze the following customer service transcript and extract detailed insights.
        
        TRANSCRIPT TO ANALYZE:
        "{transcript}"
        
        Provide your analysis in the following EXACT JSON format. Do not include any extra text or markdown.
        
        {{
            "call_summary_one_line": "A one-sentence summary of the entire call from start to finish.",
            "issue_description": "clear 3-5 word summary of the main problem",
            "product_name_options": ["list of specific products mentioned"],
            "product_category": "primary category",
            "customer_sentiment": "customer's emotional state",
            "agent_sentiment": "agent's emotional state and professionalism",
            "customer_sentiment_score": "numerical_score as a string",
            "agent_sentiment_score": "numerical_score as a string",
            "agent_experience_level": "assessment of agent's skill level",
            "issue_complexity": "complexity assessment of the issue",
            "resolved_chance": "probability_score as a string",
            "resolution_status": "current status assessment"
        }}
        
        DETAILED ANALYSIS GUIDELINES:
        - **call_summary_one_line**: Concisely summarize the entire interaction in a single sentence, covering the customer's issue, the agent's key actions, and the final outcome.
        - **issue_description**: Summarize the core problem in 3-5 words (e.g., "Account login problems").
        - **product_name_options**: List specific products. If none, use ["General Service Inquiry"].
        - **product_category**: Choose from: "Electronics", "Software/Apps", "Financial Services", "Retail/Shopping", "Telecommunications", "Other".
        - **customer_sentiment**: Describe emotion (e.g., "Moderately Upset but Cooperative", "Satisfied and Appreciative").
        - **agent_sentiment**: Assess professionalism (e.g., "Professional and Empathetic", "Helpful but Rushed").
        - **customer_sentiment_score**: Rate from -2.0 to +2.0 as a string.
        - **agent_sentiment_score**: Rate from -2.0 to +2.0 as a string.
        - **agent_experience_level**: Assess expertise (e.g., "Intermediate Agent", "Experienced Agent").
        - **issue_complexity**: Rate difficulty (e.g., "Simple/Routine", "Moderate", "Complex").
        - **resolved_chance**: Probability from 0.0 to 1.0 as a string.
        - **resolution_status**: Current state (e.g., "Fully Resolved", "Partially Resolved", "Escalated").
        
        IMPORTANT: Respond with ONLY the JSON object.
    """

    try:
        print(f"  🔄 Sending to Ollama for analysis...")
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_predict": 1024
                }
            },
            timeout=90
        )

        print(f"  📡 Response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            analysis_text = result.get('response', '').strip()
            print(f"  📝 Got response from Ollama ({len(analysis_text)} chars)")

            try:
                analysis = json.loads(analysis_text)
                print(f"  ✅ Successfully parsed JSON from Ollama")

                required_fields = {
                    'call_summary_one_line': 'Summary not generated.',
                    'issue_description': 'General Inquiry',
                    'product_name_options': ['General Service'],
                    'product_category': 'Other',
                    'customer_sentiment': 'Neutral',
                    'agent_sentiment': 'Professional',
                    'customer_sentiment_score': "0.0",
                    'agent_sentiment_score': "0.0",
                    'agent_experience_level': 'Intermediate Agent',
                    'issue_complexity': 'Moderate',
                    'resolved_chance': "0.5",
                    'resolution_status': 'In Progress'
                }

                for field, default_value in required_fields.items():
                    if field not in analysis:
                        analysis[field] = default_value
                        print(f"  ⚠️ Added missing field '{field}' with default value")

                try:
                    analysis['resolved_chance'] = float(analysis['resolved_chance'])
                    analysis['customer_sentiment_score'] = float(analysis['customer_sentiment_score'])
                    analysis['agent_sentiment_score'] = float(analysis['agent_sentiment_score'])
                except (ValueError, TypeError):
                    print("  ⚠️ Could not convert score strings to numbers.")

                return analysis

            except json.JSONDecodeError as e:
                print(f"  ❌ Failed to parse JSON: {e}")
                print(f"  📝 Raw response: {analysis_text}")
                return None
        else:
            print(f"  ❌ Ollama API failed with status {response.status_code}")
            print(f"  📄 Response: {response.text[:200]}...")
            return None

    except requests.exceptions.RequestException as e:
        print(f"  ❌ Error calling Ollama: {str(e)}")
        return None

def display_analysis_results(result):
    """Display analysis results in a user-friendly format."""
    if not result or 'analysis' not in result:
        return

    analysis = result['analysis']
    print(f"\n📊 ANALYSIS RESULTS FOR: {result['filename']}")
    print("=" * 60)

    print(f"🎯 Issue: {analysis.get('issue_description', 'N/A')}")
    print(f"📦 Products: {', '.join(analysis.get('product_name_options', []))}")
    print(f"🏷️  Category: {analysis.get('product_category', 'N/A')}")

    print(f"\n👤 CUSTOMER ANALYSIS:")
    print(f"   Sentiment: {analysis.get('customer_sentiment', 'N/A')}")
    print(f"   Score: {analysis.get('customer_sentiment_score', 0)} (-2.0 to +2.0)")

    print(f"\n🎧 AGENT ANALYSIS:")
    print(f"   Performance: {analysis.get('agent_sentiment', 'N/A')}")
    print(f"   Score: {analysis.get('agent_sentiment_score', 0)} (-2.0 to +2.0)")
    print(f"   Experience Level: {analysis.get('agent_experience_level', 'N/A')}")

    print(f"\n🔧 ISSUE ANALYSIS:")
    print(f"   Complexity: {analysis.get('issue_complexity', 'N/A')}")
    print(f"   Resolution Chance: {analysis.get('resolved_chance', 0):.1%}")
    print(f"   Current Status: {analysis.get('resolution_status', 'N/A')}")

    print("-" * 60)

def create_call_summary(filename, transcript, analysis):
    """Create call summary like main processor."""
    call_data = {
        'call_id': f'CALL_{filename.replace(".", "_")}_{int(datetime.now().timestamp())}',
        'timestamp': datetime.now().isoformat(),
        'filename': filename,
        'transcript': transcript,
        'analysis': analysis,
        'processing_status': 'completed'
    }
    return call_data

def process_audio_file(vosk_model, audio_path):
    """Process single audio file: Audio → Transcribe → Analyze - EXACT same flow as main processor."""
    filename = os.path.basename(audio_path)
    print(f"\n🎵 Processing: {filename}")

    # Step 1: Transcribe Audio
    print(f"📝 Step 1: Transcribing audio...")
    transcript = transcribe_audio(vosk_model, audio_path)

    if not transcript:
        print(f"❌ Transcription failed for {filename}")
        return None

    print(f"✅ Transcript: '{transcript}'")

    # Step 2: Analyze with Ollama
    print(f"🤖 Step 2: Analyzing with Ollama...")
    analysis = analyze_transcript_with_ollama_new(transcript)
    print(analysis)
    if not analysis:
        print(f"❌ Analysis failed for {filename}")
        return None

    print(f"✅ Analysis completed!")
    # print(f"   Issue: {analysis.get('issue_description')}")
    # print(f"   Products: {analysis.get('product_name_options')}")
    # print(f"   Category: {analysis.get('product_category')}")
    # print(f"   Sentiment: {analysis.get('sentiment_range')}")
    # print(f"   Resolved: {analysis.get('resolved_chance')}")

    # Step 3: Create call summary
    call_summary = create_call_summary(filename, transcript, analysis)

    return call_summary


def setup_database():
    """Creates the SQLite database and the calls table if they don't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Create table with flattened columns from the JSON analysis
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        call_id TEXT UNIQUE,
        timestamp TEXT,
        filename TEXT,
        processing_status TEXT,
        transcript TEXT,
        call_summary_one_line TEXT,
        issue_description TEXT,
        products_mentioned TEXT,
        product_category TEXT,
        customer_sentiment TEXT,
        agent_sentiment TEXT,
        customer_sentiment_score REAL,
        agent_sentiment_score REAL,
        agent_experience_level TEXT,
        issue_complexity TEXT,
        resolved_chance REAL,
        resolution_status TEXT
    )
    ''')

    conn.commit()
    conn.close()
    print(f"✅ Database '{DB_FILE}' is ready.")


def save_result_to_db(result):
    """Saves a single call analysis result to the SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    analysis = result.get('analysis', {})

    try:
        cursor.execute('''
        INSERT INTO calls (
            call_id, timestamp, filename, processing_status, transcript,
            call_summary_one_line, issue_description, products_mentioned, product_category,
            customer_sentiment, agent_sentiment, customer_sentiment_score, agent_sentiment_score,
            agent_experience_level, issue_complexity, resolved_chance, resolution_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.get('call_id'),
            result.get('timestamp'),
            result.get('filename'),
            result.get('processing_status'),
            result.get('transcript'),
            analysis.get('call_summary_one_line'),
            analysis.get('issue_description'),
            json.dumps(analysis.get('product_name_options', [])),  # Store list as JSON string
            analysis.get('product_category'),
            analysis.get('customer_sentiment'),
            analysis.get('agent_sentiment'),
            analysis.get('customer_sentiment_score'),
            analysis.get('agent_sentiment_score'),
            analysis.get('agent_experience_level'),
            analysis.get('issue_complexity'),
            analysis.get('resolved_chance'),
            analysis.get('resolution_status')
        ))
        conn.commit()
        print(f"  💾 Saved {result['filename']} to database.")
    except sqlite3.IntegrityError:
        print(f"  ⚠️ {result['filename']} with call_id {result.get('call_id')} already exists in DB. Skipping.")
    except Exception as e:
        print(f"  ❌ Error saving to DB: {e}")
    finally:
        conn.close()


def main():
    """Main test function."""
    print("🚀 Testing Audio Processing Pipeline")
    print("=" * 50)

    # Check Ollama connection
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code != 200:
            print(f"❌ Cannot connect to Ollama at {OLLAMA_BASE_URL}")
            return
        print(f"✅ Ollama connected at {OLLAMA_BASE_URL}")
    except:
        print(f"❌ Ollama not running. Start with: ollama serve")
        return

    # Initialize Vosk
    vosk_model = initialize_vosk()
    if not vosk_model:
        return

    # SETUP DATABASE
    setup_database()

    # Check audio files
    if not os.path.exists(AUDIO_FILES_DIR):
        print(f"❌ Audio directory not found: {AUDIO_FILES_DIR}")
        return

    audio_files = [f for f in os.listdir(AUDIO_FILES_DIR)
                   if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac'))]

    if not audio_files:
        print(f"❌ No audio files found in {AUDIO_FILES_DIR}")
        return

    print(f"📁 Found {len(audio_files)} audio files")

    # Process each audio file
    results = []
    for audio_file in audio_files[900:1000]:  # Process first 3 files
        audio_path = os.path.join(AUDIO_FILES_DIR, audio_file)
        result = process_audio_file(vosk_model, audio_path)
        if result:
            results.append(result)
            save_result_to_db(result)



    # Save results
    if results:
        with open("audio_test_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n🎉 Successfully processed {len(results)} audio files!")
        print(f"💾 Results saved to audio_test_results.json")

        # Show summary
        print(f"\n📊 Summary:")
        for result in results:
            print(f"   {result['filename']}: {result['analysis']['issue_description']}")

    else:
        print(f"\n❌ No files processed successfully")


if __name__ == "__main__":
    main()