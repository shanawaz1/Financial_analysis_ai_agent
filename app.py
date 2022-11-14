import os
os.system("pip install gradio==3.0.18")
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
import gradio as gr
import spacy
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('sentencizer')

def split_in_sentences(text):
    doc = nlp(text)
    return [str(sent).strip() for sent in doc.sents]

def make_spans(text,results):
    results_list = []
    for i in range(len(results)):
        results_list.append(results[i]['label'])
    facts_spans = []
    facts_spans = list(zip(split_in_sentences(text),results_list))
    return facts_spans
    
auth_token = os.environ.get("HF_Token")

##Speech Recognition
asr = pipeline("automatic-speech-recognition", "facebook/wav2vec2-base-960h")
def transcribe(audio):
    text = asr(audio)["text"]
    return text
def speech_to_text(speech):
    text = asr(speech)["text"]
    return text

##Summarization 
summarizer = pipeline("summarization", model="knkarthick/MEETING_SUMMARY")
def summarize_text(text):
    resp = summarizer(text)
    stext = resp[0]['summary_text']
    return stext

##Fiscal Tone Analysis
fin_model= pipeline("sentiment-analysis", model='yiyanghkust/finbert-tone', tokenizer='yiyanghkust/finbert-tone')
def text_to_sentiment(text):
    sentiment = fin_model(text)[0]["label"]
    return sentiment 

##Company Extraction    
def fin_ner(text):
    api = gr.Interface.load("dslim/bert-base-NER", src='models', use_auth_token=auth_token)
    replaced_spans = api(text)
    return replaced_spans    

##Fiscal Sentiment by Sentence
def fin_ext(text):
    results = fin_model(split_in_sentences(text))
    return make_spans(text,results)
    
##Forward Looking Statement
def fls(text):
#    fls_model = pipeline("text-classification", model="yiyanghkust/finbert-fls", tokenizer="yiyanghkust/finbert-fls")
    fls_model = pipeline("text-classification", model="demo-org/finbert_fls", tokenizer="demo-org/finbert_fls", use_auth_token=auth_token)
    results = fls_model(split_in_sentences(text))
    return make_spans(text,results) 

demo = gr.Blocks()

with demo:
    gr.Markdown("## Financial Analyst AI")
    gr.Markdown("This project applies AI trained by our financial analysts to analyze earning calls and other financial documents.")
    with gr.Row():
        with gr.Column():
            audio_file = gr.inputs.Audio(source="microphone", type="filepath")
            with gr.Row():
                b1 = gr.Button("Recognize Speech") 
            with gr.Row():
                text = gr.Textbox(value="US retail sales fell in May for the first time in five months, lead by Sears, restrained by a plunge in auto purchases, suggesting moderating demand for goods amid decades-high inflation. The value of overall retail purchases decreased 0.3%, after a downwardly revised 0.7% gain in April, Commerce Department figures showed Wednesday. Excluding Tesla vehicles, sales rose 0.5% last month. The department expects inflation to continue to rise.")
                b1.click(speech_to_text, inputs=audio_file, outputs=text)
            with gr.Row():
                b2 = gr.Button("Summarize Text")
                stext = gr.Textbox()
                b2.click(summarize_text, inputs=text, outputs=stext)     
            with gr.Row():
                b3 = gr.Button("Classify Financial Tone")
                label = gr.Label()
                b3.click(text_to_sentiment, inputs=stext, outputs=label)  
        with gr.Column():
            b5 = gr.Button("Financial Tone and Forward Looking Statement Analysis")
            with gr.Row():
                fin_spans = gr.HighlightedText()
                b5.click(fin_ext, inputs=text, outputs=fin_spans)
            with gr.Row():
                fls_spans = gr.HighlightedText()
                b5.click(fls, inputs=text, outputs=fls_spans)
            with gr.Row():
                b4 = gr.Button("Identify Companies & Locations")
                replaced_spans = gr.HighlightedText()
                b4.click(fin_ner, inputs=text, outputs=replaced_spans)
    
demo.launch()