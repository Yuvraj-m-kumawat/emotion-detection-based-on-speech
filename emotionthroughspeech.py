import speech_recognition as sr
from transformers import pipeline

# Load the emotion detection model globally (so it's not reloaded each time)
emotion_classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion"
)

def recognize_speech_from_mic():
    """Capture speech from microphone and convert to text."""
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Please speak something...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.RequestError:
        print("Could not request results from Google Speech Recognition service.")
        return None
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None

def detect_emotion_from_text(text):
    """Detect emotion from a given text input."""
    if not text:
        return None
    
    try:
        emotions = emotion_classifier(text)
        return emotions
    except Exception as e:
        print(f"Error detecting emotion: {e}")
        return None

# Example usage
if __name__ == "__main__":
    spoken_text = recognize_speech_from_mic()
    if spoken_text:
        emotions = detect_emotion_from_text(spoken_text)
        print("Detected emotions:", emotions)
