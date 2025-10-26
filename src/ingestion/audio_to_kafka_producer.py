import os
import json
import sqlite3
import wave
import vosk
import requests
import tempfile
import random
from pydub import AudioSegment
from datetime import datetime
from dotenv import load_dotenv
from kafka import KafkaProducer
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

load_dotenv()

# Configuration
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "../../data/vosk_model/vosk-model-small-en-us-0.15")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
AUDIO_FILES_DIR = os.getenv("AUDIO_FILES_DIR", "../../audio_files_source")
DB_FILE = os.getenv("DB_FILE", "../../data/call_analysis.db")
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "raw-calls")

_vosk_model_instance = None
_vosk_model_lock = threading.Lock()
_kafka_producer_instance = None
_kafka_producer_lock = threading.Lock()

def initialize_vosk():
    global _vosk_model_instance
    with _vosk_model_lock:
        if _vosk_model_instance is None:
            if not os.path.exists(VOSK_MODEL_PATH):
                print(f"❌ Vosk model not found at {VOSK_MODEL_PATH}.")
                return None
            vosk.SetLogLevel(-1)
            try:
                _vosk_model_instance = vosk.Model(VOSK_MODEL_PATH)
                print(f"✅ Vosk model loaded.")
            except Exception as e:
                print(f"❌ Error loading Vosk model: {e}")
                return None
        return _vosk_model_instance

def get_kafka_producer():
    global _kafka_producer_instance
    with _kafka_producer_lock:
        if _kafka_producer_instance is None:
            try:
                _kafka_producer_instance = KafkaProducer(
                    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                    value_serializer=lambda x: json.dumps(x).encode('utf-8')
                )
                print(f"✅ Kafka producer initialized.")
            except Exception as e:
                print(f"❌ Error initializing Kafka producer: {e}")
                return None
        return _kafka_producer_instance

def convert_audio_for_vosk(audio_path):
    try:
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio.export(temp_wav.name, format="wav")
        return temp_wav.name
    except Exception as e:
        print(f"❌ Error converting audio '{audio_path}': {str(e)}")
        return None

def transcribe_audio(vosk_model, audio_path):
    wav_path = None
    try:
        print(f"  🔄 Converting audio format for Vosk...")
        wav_path = convert_audio_for_vosk(audio_path)
        if not wav_path: return None
        with wave.open(wav_path, 'rb') as wf:
            rec = vosk.KaldiRecognizer(vosk_model, wf.getframerate())
            transcript_parts = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0: break
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    if result.get('text'): transcript_parts.append(result['text'])
            final_result = json.loads(rec.FinalResult())
            if final_result.get('text'): transcript_parts.append(final_result['text'])
            full_transcript = ' '.join(transcript_parts).strip()
            return full_transcript if full_transcript else "No speech detected"
    except Exception as e:
        print(f"  ❌ Error during Vosk transcription: {str(e)}")
        return None
    finally:
        if wav_path and os.path.exists(wav_path): os.unlink(wav_path)

def analyze_transcript_with_ollama(transcript):
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
            "customer_sentiment_score": 0.0,
            "agent_sentiment_score": 0.0,
            "agent_experience_level": "assessment of agent's skill level",
            "issue_complexity": "complexity assessment of the issue",
            "resolved_chance": 0.5,
            "resolution_status": "current status assessment",
            "agent_id": "AGENT_XX",
            "customer_id": "CUST_YY",
            "call_type": "support"
        }}
        DETAILED ANALYSIS GUIDELINES:
        - **call_summary_one_line**: Concisely summarize the entire interaction.
        - **issue_description**: Summarize the core problem in 3-5 words.
        - **product_name_options**: List specific products. If none, use ["General Service Inquiry"].
        - **product_category**: Choose from: "Electronics", "Software/Apps", "Financial Services", "Retail/Shopping", "Telecommunications", "Other".
        - **customer_sentiment**: Describe emotion.
        - **agent_sentiment**: Assess professionalism.
        - **customer_sentiment_score**: Rate from -2.0 to +2.0 as a float.
        - **agent_sentiment_score**: Rate from -2.0 to +2.0 as a float.
        - **agent_experience_level**: Assess expertise.
        - **issue_complexity**: Rate difficulty.
        - **resolved_chance**: Probability from 0.0 to 1.0 as a float.
        - **resolution_status**: Current state.
        - **agent_id**: Infer an agent ID (e.g., "AGENT_1" to "AGENT_15").
        - **customer_id**: Infer a customer ID (e.g., "CUST_101" to "CUST_999").
        - **call_type**: Infer the call type from the conversation: "complaint", "support", or "praise".
        IMPORTANT: Respond with ONLY the JSON object.
    """
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json", "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 1024}},
            timeout=120
        )
        if response.status_code == 200:
            analysis_text = response.json().get('response', '').strip()
            if analysis_text.startswith("```json") and analysis_text.endswith("```"):
                analysis_text = analysis_text[7:-3].strip()
            analysis = json.loads(analysis_text)
            required_fields = {
                'call_summary_one_line': 'Summary not generated.', 'issue_description': 'General Inquiry',
                'product_name_options': ["General Service"], 'product_category': 'Other',
                'customer_sentiment': 'Neutral', 'agent_sentiment': 'Professional',
                'customer_sentiment_score': 0.0, 'agent_sentiment_score': 0.0,
                'agent_experience_level': 'Intermediate Agent', 'issue_complexity': 'Moderate',
                'resolved_chance': 0.5, 'resolution_status': 'In Progress',
                'agent_id': f"AGENT_{random.randint(1,15)}", 'customer_id': f"CUST_{random.randint(100,999)}",
                'call_type': 'support'
            }
            for field, default_value in required_fields.items():
                if field not in analysis or analysis[field] is None: analysis[field] = default_value
                if field in ['customer_sentiment_score', 'agent_sentiment_score', 'resolved_chance']:
                    try: analysis[field] = float(analysis[field])
                    except (ValueError, TypeError): analysis[field] = default_value
                if field == 'product_name_options' and not isinstance(analysis[field], list):
                     analysis[field] = [str(analysis[field])]
            if analysis.get('resolution_status', '').lower() in ['fully resolved', 'resolved', 'complete']:
                analysis['problem_resolved'] = True
            else:
                analysis['problem_resolved'] = False
            return analysis
        else:
            print(f"❌ Ollama API call failed with status {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Network error calling Ollama: {str(e)}")
        return None

def setup_database():
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
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
            resolution_status TEXT,
            problem_resolved BOOLEAN,
            agent_id TEXT,
            customer_id TEXT,
            call_type TEXT
        )
        ''')
        conn.commit()
        print(f"✅ Local SQLite database '{DB_FILE}' is ready.")
    except Exception as e:
        print(f"❌ Error setting up local SQLite database: {e}")
    finally:
        if conn: conn.close()

def save_result_to_db(result):
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        analysis = result.get('analysis', {})
        products_json = json.dumps(analysis.get('product_name_options', []))
        cursor.execute('''
        INSERT INTO calls (
            call_id, timestamp, filename, processing_status, transcript,
            call_summary_one_line, issue_description, products_mentioned, product_category,
            customer_sentiment, agent_sentiment, customer_sentiment_score, agent_sentiment_score,
            agent_experience_level, issue_complexity, resolved_chance, resolution_status, problem_resolved,
            agent_id, customer_id, call_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.get('call_id'), result.get('timestamp'), result.get('filename'), 'completed',
            result.get('transcript'), analysis.get('call_summary_one_line'),
            analysis.get('issue_description'), products_json, analysis.get('product_category'),
            analysis.get('customer_sentiment'), analysis.get('agent_sentiment'),
            analysis.get('customer_sentiment_score'), analysis.get('agent_sentiment_score'),
            analysis.get('agent_experience_level'), analysis.get('issue_complexity'),
            analysis.get('resolved_chance'), analysis.get('resolution_status'),
            analysis.get('problem_resolved'), analysis.get('agent_id'),
            analysis.get('customer_id'), analysis.get('call_type')
        ))
        conn.commit()
        print(f"  💾 Saved {result['filename']} to local SQLite DB.")
    except sqlite3.IntegrityError:
        print(f"  ⚠️ {result['filename']} with call_id {result.get('call_id')} already exists in local DB. Skipping.")
    except Exception as e:
        print(f"  ❌ Error saving {result.get('filename')} to local DB: {e}")
    finally:
        if conn: conn.close()

def process_audio_file_and_send_to_kafka(vosk_model, audio_path, kafka_producer):
    filename = os.path.basename(audio_path)
    print(f"\n--- Processing: {filename} ---")
    transcript = transcribe_audio(vosk_model, audio_path)
    if not transcript: return {'status': 'failed', 'filename': filename, 'error': 'Transcription failed'}
    analysis = analyze_transcript_with_ollama(transcript)
    if not analysis: return {'status': 'failed', 'filename': filename, 'error': 'Analysis failed'}
    call_summary = {
        'call_id': f'CALL_{filename.replace(".", "_")}_{int(datetime.now().timestamp())}_{random.randint(1,999)}',
        'timestamp': datetime.now().isoformat(), 'filename': filename,
        'agent_id': analysis.get('agent_id'), 'customer_id': analysis.get('customer_id'),
        'call_type': analysis.get('call_type', 'support'), 'transcript': transcript,
        'duration_seconds': random.randint(60, 300), 'sentiment': analysis.get('customer_sentiment'),
        'polarity_score': analysis.get('customer_sentiment_score'), 'problem_resolved': analysis.get('problem_resolved'),
        'issue_description': analysis.get('issue_description'), 'product_name': json.dumps(analysis.get('product_name_options')),
        'product_category': analysis.get('product_category'), 'call_summary_one_line': analysis.get('call_summary_one_line'),
        'agent_sentiment': analysis.get('agent_sentiment'), 'agent_sentiment_score': analysis.get('agent_sentiment_score'),
        'agent_experience_level': analysis.get('agent_experience_level'), 'issue_complexity': analysis.get('issue_complexity'),
        'resolved_chance': analysis.get('resolved_chance'), 'resolution_status': analysis.get('resolution_status')
    }
    try:
        kafka_producer.send(KAFKA_TOPIC, call_summary)
        print(f"✅ Sent {filename} summary to Kafka topic '{KAFKA_TOPIC}'.")
        return {'status': 'success', 'call_id': call_summary['call_id'], 'filename': filename, 'transcript': transcript, 'analysis': analysis}
    except Exception as e:
        print(f"❌ Error sending {filename} to Kafka: {e}")
        return {'status': 'failed', 'filename': filename, 'error': str(e)}

def main_processor():
    print("🚀 Starting Local Audio Processing Pipeline to Kafka")
    print("=" * 60)
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code != 200:
            print(f"❌ Cannot connect to Ollama at {OLLAMA_BASE_URL}. Status: {response.status_code}")
            return
        print(f"✅ Ollama connected at {OLLAMA_BASE_URL}")
        models = [m['name'] for m in response.json().get('models', [])]
        if OLLAMA_MODEL not in models:
            print(f"❌ Ollama model '{OLLAMA_MODEL}' not found.")
            return
        print(f"✅ Ollama model '{OLLAMA_MODEL}' is available.")
    except Exception as e:
        print(f"❌ An unexpected error occurred while connecting to Ollama: {e}")
        return
    vosk_model = initialize_vosk()
    if not vosk_model: return
    setup_database()
    try:
        kafka_producer = get_kafka_producer()
        if not kafka_producer: return
    except Exception as e:
        print(f"❌ Error initializing Kafka producer: {e}")
        return
    if not os.path.exists(AUDIO_FILES_DIR):
        print(f"❌ Audio directory not found: {AUDIO_FILES_DIR}")
        return
    all_audio_files = [f for f in os.listdir(AUDIO_FILES_DIR) if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac'))]
    if not all_audio_files:
        print(f"❌ No audio files found in {AUDIO_FILES_DIR}.")
        return
    num_files_to_process = -1
    while num_files_to_process < 0:
        try:
            user_input = input(f"Enter number of files to process (1-{len(all_audio_files)}, or 'all'): ").strip().lower()
            if user_input == 'all': num_files_to_process = len(all_audio_files)
            else:
                num_files_to_process = int(user_input)
                if not (1 <= num_files_to_process <= len(all_audio_files)):
                    print(f"Please enter a number between 1 and {len(all_audio_files)}, or 'all'.")
                    num_files_to_process = -1
        except ValueError: print("Invalid input.")
    audio_files_to_process = all_audio_files[:num_files_to_process]
    random.shuffle(audio_files_to_process)
    print(f"⚡ Processing {len(audio_files_to_process)} selected audio file(s) in parallel and sending to Kafka...")
    processed_count = 0
    failed_count = 0
    MAX_WORKERS = os.cpu_count() * 2
    if MAX_WORKERS > 8: MAX_WORKERS = 8
    print(f"Using {MAX_WORKERS} parallel workers.")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for audio_file in audio_files_to_process:
            audio_path = os.path.join(AUDIO_FILES_DIR, audio_file)
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute("SELECT call_id FROM calls WHERE filename = ?", (audio_file,))
            if cursor.fetchone():
                print(f"\n--- Skipping: {audio_file} (already processed) ---")
                conn.close()
                continue
            conn.close()
            futures.append(executor.submit(
                process_audio_file_and_send_to_kafka, vosk_model, audio_path, kafka_producer
            ))
        for i, future in enumerate(as_completed(futures)):
            result_obj = future.result()
            if result_obj and result_obj['status'] == 'success':
                processed_count += 1
                save_result_to_db(result_obj)
            else:
                failed_count += 1
            print(f"Progress: {i+1}/{len(futures)} processed. Success: {processed_count}, Failed: {failed_count}.")
    print(f"\nSummary of run:")
    print(f"🎉 Successfully processed and sent {processed_count} files to Kafka.")
    print(f"💔 Failed to process {failed_count} files.")
    print("\nLocal Audio Processing Pipeline finished.")

if __name__ == "__main__": main_processor()