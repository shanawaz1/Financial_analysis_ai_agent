from transformers import pipeline
import gradio as gr

asr = pipeline("automatic-speech-recognition", "facebook/wav2vec2-base-960h")

classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

def transcribe(audio):
    text = asr(audio)["text"]
    return text

def speech_to_text(speech):
    text = asr(speech)["text"]
    return text

def text_to_sentiment(text):
    sentiment = classifier(text)[0]["label"]
    return sentiment 
    
demo = gr.Blocks()

with demo:

    audio_file = gr.inputs.Audio(source="microphone", type="filepath")
    b1 = gr.Button("Recognize Speech") 
    text = gr.Textbox()
    b1.click(speech_to_text, inputs=audio_file, outputs=text)
    
    b2 = gr.Button("Classify Sentiment")
    label = gr.Label()
    b2.click(text_to_sentiment, inputs=text, outputs=label)
    
demo.launch(share=True)