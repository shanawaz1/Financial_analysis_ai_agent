from transformers import pipeline
import gradio as gr

asr = pipeline("automatic-speech-recognition", "facebook/wav2vec2-base-960h")
summarizer = pipeline("summarization", model="knkarthick/MEETING_SUMMARY")
classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

def transcribe(audio):
    text = asr(audio)["text"]
    return text

def speech_to_text(speech):
    text = asr(speech)["text"]
    return text

def summarize_text(text):
    stext = summarizer(text)
    return stext

def text_to_sentiment(text):
    sentiment = classifier(text)[0]["label"]
    return sentiment 
    
demo = gr.Blocks()

with demo:

    audio_file = gr.inputs.Audio(source="microphone", type="filepath")
    b1 = gr.Button("Recognize Speech") 
    text = gr.Textbox()
    b1.click(speech_to_text, inputs=audio_file, outputs=text)
    
    b2 = gr.Button("Summarize Text")
    stext = gr.Textbox()
    b2.click(summarize_text, inputs=text, outputs=stext)
    
    b3 = gr.Button("Classify Sentiment")
    label = gr.Label()
    b3.click(text_to_sentiment, inputs=stext, outputs=label)
    
demo.launch(share=True)