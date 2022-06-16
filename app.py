from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
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
    resp = summarizer(text)
    stext = resp[0]['summary_text']
    return stext

##Fiscal Sentiment
#fin_model = pipeline("text-classification", model="demo-org/auditor_review_model", \
#    tokenizer="demo-org/auditor_review_model",use_auth_token=auth_token)
fin_model = pipeline("text-classification")
def text_to_sentiment(text):
    sentiment = fin_model(text)[0]["label"]
    return sentiment 

##Company Extraction    
def fin_ner(text):
    print ("ner")
    #ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", tokenizer="dslim/bert-base-NER")
    api = gr.Interface.load("dslim/bert-base-NER", src='models')
    replaced_spans = api(text)
    print (replaced_spans)
    print ("spans2")
    #replaced_spans = [(key, None) if value=='No Disease' else (key, value) for (key, value) in spans]
    return replaced_spans    

##Fiscal Sentiment by Sentence
def fin_ext(text):
    print ("sent")
    doc = nlp(text)
    doc_sents = [sent for sent in doc.sents]
    sents_list = []
    for sent in doc.sents:
        sents_list.append(sent.text)
    results = fin_model(sents_list)
    print (results)
    results_list = []
    for i in range(len(results)):
        results_list.append(results[i]['label'])
    fin_spans = []
    fin_spans = list(zip(sents_list,results_list))
    print (fin_spans)
    return fin_spans    

demo = gr.Blocks()

demo = gr.Blocks()

with demo:
    with gr.Row():
        with gr.Column():
            audio_file = gr.inputs.Audio(source="microphone", type="filepath")
            with gr.Row():
                b1 = gr.Button("Recognize Speech") 
            with gr.Row():
                text = gr.Textbox(value="US retail sales fell in May for the first time in five months, restrained by a plunge in auto purchases and other big-ticket items, suggesting moderating demand for goods amid decades-high inflation. The value of overall retail purchases decreased 0.3%, after a downwardly revised 0.7% gain in April, Commerce Department figures showed Wednesday. Excluding vehicles, sales rose 0.5% last month. The figures aren’t adjusted for inflation.")
                b1.click(speech_to_text, inputs=audio_file, outputs=text)
            with gr.Row():
                b2 = gr.Button("Summarize Text")
                stext = gr.Textbox()
                b2.click(summarize_text, inputs=text, outputs=stext)       
        with gr.Column():
            with gr.Row():
                b3 = gr.Button("Classify Overall Financial Sentiment")
                label = gr.Label()
                b3.click(text_to_sentiment, inputs=stext, outputs=label)
            with gr.Row():
                b4 = gr.Button("Extract Companies & Segments")
                replaced_spans = gr.HighlightedText()
                b4.click(fin_ner, inputs=text, outputs=replaced_spans)
            with gr.Row():
                b5 = gr.Button("Extract Financial Sentiment")
                fin_spans = gr.HighlightedText()
                b5.click(fin_ext, inputs=text, outputs=fin_spans)
    
demo.launch()