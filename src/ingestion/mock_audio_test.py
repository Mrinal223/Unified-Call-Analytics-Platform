# simple_audio_test.py
import os
import json
import wave
import vosk
import requests
import tempfile
from pydub import AudioSegment
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Configuration
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "../../vosk_model/vosk-model-small-en-us-0.15")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
AUDIO_FILES_DIR = "../../audio_files_source_test"


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


def analyze_transcript_with_ollama(transcript):
    """Analyze transcript using Ollama - EXACT same method as main processor."""
    prompt = f"""
You are an expert customer service analyst. Analyze the following customer service transcript and extract key information.

Transcript: "{transcript}"

Please analyze this transcript and provide the response in the following exact JSON format:
{{
    "transcript_base": "exact transcript text",
    "issue_description": "brief description of the main issue or topic",
    "product_name_options": ["list", "of", "mentioned", "products"],
    "product_category": "general category of products discussed",
    "sentiment_range": [min_sentiment_score, max_sentiment_score],
    "resolved_chance": probability_between_0_and_1
}}

Guidelines for analysis:
- transcript_base: Use the exact transcript provided
- issue_description: Summarize the main issue in 2-4 words
- product_name_options: Extract any specific product names mentioned, if none then ["General Product"]
- product_category: Categorize as Electronics, Apparel, Furniture, Services, or Other
- sentiment_range: Provide range as [min, max] where -1.0=very negative, 0=neutral, 1.0=very positive
- resolved_chance: Estimate probability (0.0-1.0) that the issue was or will be resolved

Respond only with the JSON object, no additional text.
"""

    try:
        print(f"  🔄 Sending to Ollama for analysis...")
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            },
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            analysis_text = result.get('response', '').strip()

            print(f"  📝 Got response from Ollama ({len(analysis_text)} chars)")

            # Clean the response to extract JSON
            if '```json' in analysis_text:
                analysis_text = analysis_text.split('```json')[1].split('```')[0]
            elif '```' in analysis_text:
                analysis_text = analysis_text.split('```')[1].split('```')[0]

            analysis = json.loads(analysis_text)
            print(f"  ✅ Successfully parsed JSON from Ollama")
            return analysis

        else:
            print(f"  ❌ Ollama API failed with status {response.status_code}")
            return None

    except json.JSONDecodeError as e:
        print(f"  ❌ Failed to parse JSON: {e}")
        print(f"  📝 Raw response: {analysis_text}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Error calling Ollama: {str(e)}")
        return None


def analyze_transcript_with_ollama_new(transcript):
    """Analyze transcript using Ollama with improved prompt and parameters."""
    prompt = f"""
        You are an expert customer service quality analyst with years of experience in call center operations. Analyze the following customer service transcript and extract detailed insights.
        
        TRANSCRIPT TO ANALYZE:
        "{transcript}"
        
        Provide your analysis in the following EXACT JSON format (no additional text):
        
        {{
            "transcript_base": "exact transcript text as provided",
            "issue_description": "clear 3-5 word summary of the main problem",
            "product_name_options": ["list of specific products mentioned"],
            "product_category": "primary category",
            "customer_sentiment": "customer's emotional state",
            "agent_sentiment": "agent's emotional state and professionalism",
            "customer_sentiment_score": numerical_score,
            "agent_sentiment_score": numerical_score,
            "agent_experience_level": "assessment of agent's skill level",
            "issue_complexity": "complexity assessment of the issue",
            "resolved_chance": probability_score,
            "resolution_status": "current status assessment"
        }}
        
        DETAILED ANALYSIS GUIDELINES:
        
        **issue_description**: Summarize the core problem in 3-5 words (e.g., "Account login problems", "Defective product return", "Billing dispute resolution")
        
        **product_name_options**: List specific products/services mentioned. If none, use ["General Service Inquiry"]
        
        **product_category**: Choose from: "Electronics", "Software/Apps", "Financial Services", "Retail/Shopping", "Telecommunications", "Healthcare", "Travel/Hospitality", "Utilities", "Other"
        
        **customer_sentiment**: Describe customer's emotional state using phrases like:
        - "Very Frustrated and Angry" 
        - "Moderately Upset but Cooperative"
        - "Neutral and Matter-of-fact"
        - "Satisfied and Appreciative"
        - "Confused but Patient"
        
        **agent_sentiment**: Assess agent's professionalism and approach:
        - "Professional and Empathetic"
        - "Helpful but Rushed" 
        - "Patient and Knowledgeable"
        - "Defensive or Dismissive"
        - "Friendly and Solution-focused"
        
        **customer_sentiment_score**: Rate from -2.0 to +2.0 (but as a string) where:
        - -2.0 = Extremely angry/hostile
        - -1.0 = Frustrated/upset
        - 0.0 = Neutral/indifferent  
        - +1.0 = Satisfied/pleased
        - +2.0 = Very happy/grateful
        
        **agent_sentiment_score**: Rate agent's performance from -2.0 to +2.0 (but as string)  where:
        - -2.0 = Unprofessional/rude
        - -1.0 = Poor service/unhelpful
        - 0.0 = Basic/adequate service
        - +1.0 = Good service quality
        - +2.0 = Exceptional service
        
        **agent_experience_level**: Assess agent's expertise:
        - "Trainee/New Agent" - Basic responses, uncertain, needs supervision
        - "Intermediate Agent" - Handles routine issues well, some hesitation on complex matters
        - "Experienced Agent" - Confident, knowledgeable, handles most issues smoothly  
        - "Senior/Expert Agent" - Highly skilled, handles complex issues, mentors others
        - "Specialist/Supervisor" - Deep expertise, handles escalations
        
        **issue_complexity**: Rate the difficulty level:
        - "Simple/Routine" - Standard FAQ, quick resolution
        - "Moderate" - Requires some research or multiple steps
        - "Complex" - Technical issue, policy exceptions, multiple departments
        - "Highly Complex" - Rare issue, requires specialist knowledge or escalation
        
        **resolved_chance**: Probability from 0.0 to 1.0 (but as string):
        - 0.0-0.2 = Very unlikely to resolve (major systemic issues)
        - 0.3-0.4 = Low chance (complex problems, multiple attempts needed)
        - 0.5-0.6 = Moderate chance (standard issues with some complications)
        - 0.7-0.8 = High chance (routine issues, agent capable)
        - 0.9-1.0 = Very likely resolved (simple issues, expert handling)
        
        **resolution_status**: Current state assessment:
        - "Fully Resolved" - Issue completely solved, customer satisfied
        - "Partially Resolved" - Some progress made, follow-up needed
        - "In Progress" - Actively being worked on
        - "Escalated" - Transferred to higher level/specialist
        - "Unresolved" - No solution found yet
        - "Pending Customer Action" - Waiting for customer response/action
        
        IMPORTANT: Respond with ONLY the JSON object. No explanations, no additional text, no markdown formatting.
    """

    try:
        print(f"  🔄 Sending to Ollama for analysis...")
        print(f"  📍 URL: {OLLAMA_BASE_URL}/api/generate")
        print(f"  🤖 Model: {OLLAMA_MODEL}")

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,  # Slightly higher for more nuanced responses
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "num_predict": 500  # Ensure enough tokens for complete response
                }
            },
            timeout=90  # Longer timeout for complex analysis
        )

        print(f"  📡 Response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            analysis_text = result.get('response', '').strip()

            print(f"  analysis: {analysis_text} \n 📝 Got response from Ollama ({len(analysis_text)} chars)")

            # Clean the response to extract JSON
            analysis_text = analysis_text.strip()

            # Remove any markdown formatting
            if '```json' in analysis_text:
                analysis_text = analysis_text.split('```json')[1].split('```')[0].strip()
            elif '```' in analysis_text:
                analysis_text = analysis_text.split('```')[1].split('```')[0].strip()

            # Remove any leading/trailing text that isn't JSON
            if analysis_text.startswith('{') and analysis_text.endswith('}'):
                pass  # Good, already clean JSON
            else:
                # Try to find JSON object within the text
                start = analysis_text.find('{')
                end = analysis_text.rfind('}') + 1
                if start != -1 and end > start:
                    analysis_text = analysis_text[start:end]

            try:
                analysis = json.loads(analysis_text)
                print(f"  ✅ Successfully parsed JSON from Ollama")

                # Validate required fields and add defaults if missing
                required_fields = {
                    'transcript_base': transcript,
                    'issue_description': 'General Inquiry',
                    'product_name_options': ['General Service'],
                    'product_category': 'Other',
                    'customer_sentiment': 'Neutral',
                    'agent_sentiment': 'Professional',
                    'customer_sentiment_score': 0.0,
                    'agent_sentiment_score': 0.0,
                    'agent_experience_level': 'Intermediate Agent',
                    'issue_complexity': 'Moderate',
                    'resolved_chance': 0.5,
                    'resolution_status': 'In Progress'
                }

                for field, default_value in required_fields.items():
                    if field not in analysis:
                        analysis[field] = default_value
                        print(f"  ⚠️ Added missing field '{field}' with default value")

                return analysis

            except json.JSONDecodeError as e:
                print(f"  ❌ Failed to parse JSON: {e}")
                print(f"  📝 Cleaned response: {analysis_text}...")
                return None

        elif response.status_code == 404:
            print(f"  ❌ Model '{OLLAMA_MODEL}' not found (404)")
            print(f"  💡 Try: ollama pull {OLLAMA_MODEL}")
            print(f"  📋 Check available models: ollama list")
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
    for audio_file in audio_files[:3]:  # Process first 3 files
        audio_path = os.path.join(AUDIO_FILES_DIR, audio_file)
        result = process_audio_file(vosk_model, audio_path)
        if result:
            results.append(result)

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