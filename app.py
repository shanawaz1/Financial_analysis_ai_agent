import streamlit as st
import datetime
from transformers import pipeline
import gradio as gr


asr = pipeline("automatic-speech-recognition", "facebook/wav2vec2-base-960h")

def transcribe(audio):
    text = asr(audio)["text"]
    return text

classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion")

def speech_to_text(speech):
    text = asr(speech)["text"]
    return text

def text_to_sentiment(text):
    sentiment = classifier(text)[0]["label"]
    return sentiment 
    
demo = gr.Blocks()

with demo:
    #audio_file = gr.Audio(type="filepath")
    audio_file = gr.inputs.Audio(source="microphone", type="filepath")
    text = gr.Textbox()
    label = gr.Label()
    saved = gr.Textbox()
    savedAll = gr.Textbox()
    
    b1 = gr.Button("Recognize Speech")
    b2 = gr.Button("Classify Sentiment")

    b1.click(speech_to_text, inputs=audio_file, outputs=text)
    b2.click(text_to_sentiment, inputs=text, outputs=label)
    
demo.launch(share=True)