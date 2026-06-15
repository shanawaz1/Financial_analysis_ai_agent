from transformers import pipeline

summarizer = pipeline("summarization", model="knkarthick/MEETING_SUMMARY")
print(summarizer("The quick brown fox jumps over the lazy dog."))

asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=-1)
print("Whisper model loaded successfully")
