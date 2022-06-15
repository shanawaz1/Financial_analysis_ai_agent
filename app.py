from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr
import os
auth_token = os.environ.get("HF_Token")

asr = pipeline("automatic-speech-recognition", "facebook/wav2vec2-base-960h")
summarizer = pipeline("summarization", model="knkarthick/MEETING_SUMMARY")
tokenizer = AutoTokenizer.from_pretrained("demo-org/auditor_review_model",use_auth_token=auth_token)
audit_model = AutoModelForSequenceClassification.from_pretrained("demo-org/auditor_review_model",use_auth_token=auth_token)
nlp = pipeline("text-classification", model=audit_model, tokenizer=tokenizer)

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
    sentiment = nlp(text)[0]["label"]
    return sentiment 
    
def ner(text):
    api = gr.Interface.load("dslim/bert-base-NER", src='models')
    spans = api(text)
    #replaced_spans = [(key, None) if value=='No Disease' else (key, value) for (key, value) in spans]
    return spans    
    
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
    
    b4 = gr.Button("Extract Companies & Segments")
    replaced_spans = gr.HighlightedText()
    b4.click(ner, inputs=text, outputs=replaced_spans)
    
demo.launch(share=True)