from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr
import os
import spacy
nlp = spacy.load('en_core_web_sm')

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
    stext = summarizer(text)
    return stext

##Fiscal Sentiment
tokenizer = AutoTokenizer.from_pretrained("demo-org/auditor_review_model",use_auth_token=auth_token)
audit_model = AutoModelForSequenceClassification.from_pretrained("demo-org/auditor_review_model",use_auth_token=auth_token)
nlp = pipeline("text-classification", model=audit_model, tokenizer=tokenizer)
def text_to_sentiment(text):
    sentiment = nlp(text)[0]["label"]
    return sentiment 

##Company Extraction    
def ner(text):
    api = gr.Interface.load("dslim/bert-base-NER", src='models')
    spans = api(text)
    #replaced_spans = [(key, None) if value=='No Disease' else (key, value) for (key, value) in spans]
    return spans    

##Fiscal Sentiment by Sentence
def fin_ext(text):
    doc = nlp(text)
    doc_sents = [sent for sent in doc.sents]
    sents_list = []
    for sent in doc.sents:
        sents_list.append(sent.text)
    results_list = []
    for i in range(len(results)):
        results_list.append(results[i]['label'])
    fin_spans = []
    fin_spans = list(zip(sents_list,results_list))
    return fin_spans    

demo = gr.Blocks()

with demo:

    audio_file = gr.inputs.Audio(source="microphone", type="filepath")
    b1 = gr.Button("Recognize Speech") 
    text = gr.Textbox()
    b1.click(speech_to_text, inputs=audio_file, outputs=text)
    
    b2 = gr.Button("Summarize Text")
    stext = gr.Textbox()
    b2.click(summarize_text, inputs=text, outputs=stext)
    
    b3 = gr.Button("Classify Overall Financial Sentiment")
    label = gr.Label()
    b3.click(text_to_sentiment, inputs=stext, outputs=label)
    
    b4 = gr.Button("Extract Companies & Segments")
    replaced_spans = gr.HighlightedText()
    b4.click(ner, inputs=text, outputs=replaced_spans)
    
    b5 = gr.Button("Extract Financial Sentiment")
    replaced_spans = gr.HighlightedText()
    b5.click(fin_ext, inputs=text, outputs=fin_spans)
    
demo.launch(share=True)