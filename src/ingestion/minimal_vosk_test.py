# minimal_vosk_test.py
import os
import json
import wave
import vosk


def test_vosk_setup():
    """Test if Vosk is properly set up."""

    model_path = "../../vosk_model/vosk-model-small-en-us-0.15"

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Vosk model not found at {model_path}")
        print("Please download it with:")
        print("wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")
        print("unzip vosk-model-small-en-us-0.15.zip")
        return False

    try:
        # Test model loading
        print("Loading Vosk model...")
        vosk.SetLogLevel(-1)  # Suppress logs
        model = vosk.Model(model_path)
        print("✅ Vosk model loaded successfully!")

        # Test recognizer creation
        rec = vosk.KaldiRecognizer(model, 16000)
        print("✅ Vosk recognizer created successfully!")

        return True

    except Exception as e:
        print(f"❌ Error testing Vosk: {str(e)}")
        return False


def test_audio_file_exists():
    """Check if test audio files exist."""
    test_dir = "../../audio_files_source_test"

    if not os.path.exists(test_dir):
        print(f"❌ Test directory {test_dir} doesn't exist")
        return False

    audio_files = [f for f in os.listdir(test_dir) if f.endswith('.wav')]

    if not audio_files:
        print(f"❌ No audio files found in {test_dir}")
        print("Run test_audio_generator.py first or add real audio files")
        return False

    print(f"✅ Found {len(audio_files)} audio files: {audio_files}")
    return True


if __name__ == "__main__":
    print("=== Vosk Setup Test ===")
    vosk_ok = test_vosk_setup()

    print("\n=== Audio Files Test ===")
    audio_ok = test_audio_file_exists()

    if vosk_ok and audio_ok:
        print("\n✅ All tests passed! Ready to run the main script.")
    else:
        print("\n❌ Some tests failed. Fix the issues above first.")