import os
import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import spacy

# Install required packages
os.system("pip install gradio==3.0.18")
os.system("pip install librosa")
os.system("pip install soundfile")

# Load spaCy model

nlp = spacy.load("en_core_web_lg")  # Use large model for better accuracy
nlp.add_pipe('sentencizer')

def split_in_sentences(text):
    doc = nlp(text)
    return [str(sent).strip() for sent in doc.sents]

def make_spans(text, results):
    results_list = [results[i]['label'] for i in range(len(results))]
    facts_spans = list(zip(split_in_sentences(text), results_list))
    return facts_spans

# Speech Recognition
# asr = pipeline("automatic-speech-recognition", "facebook/wav2vec2-base-960h")
# Use a pipeline as a high-level helper

from transformers import pipeline

# Initialize pipeline with chunking parameters
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    chunk_length_s=30,  # Split audio into 30s chunks
    stride_length_s=(5, 5),  # 5s overlap at chunk boundaries
    device=0  # Use GPU if available
)

def speech_to_text(audio_path):
    if audio_path is None:
        return "Please upload an audio file first."
    try:
        # Process with timestamps and chunking
        result = asr(audio_path, return_timestamps=True)
        
        # Combine text from all chunks
        full_text = " ".join([chunk['text'] for chunk in result['chunks']])
        
        return full_text
    except Exception as e:
        return f"Error processing audio: {str(e)}"

# Summarization
summarizer = pipeline("summarization", model="knkarthick/MEETING_SUMMARY")

def summarize_text(text):
    resp = summarizer(text)
    stext = resp[0]['summary_text']
    return stext

# Fiscal Tone Analysis
fin_model = pipeline("sentiment-analysis", 
                    model='yiyanghkust/finbert-tone', 
                    tokenizer='yiyanghkust/finbert-tone')

def text_to_sentiment(text):
    sentiment = fin_model(text)[0]["label"]
    return sentiment

# Company Extraction
def fin_ner(text):
    doc = nlp(text)
    entities = []
    
    for ent in doc.ents:
        if ent.label_ == "ORG" and len(ent.text) > 2:
            entities.append((ent.start_char, ent.end_char, "COMPANY"))
        elif ent.label_ in ("GPE", "LOC"):
            entities.append((ent.start_char, ent.end_char, "LOCATION"))
    
    # Convert to Gradio's HighlightedText format
    output = []
    last_end = 0
    for start, end, label in sorted(entities, key=lambda x: x[0]):
        # Add text before entity
        if start > last_end:
            output.append((text[last_end:start], None))
        # Add entity
        output.append((text[start:end], label))
        last_end = end
    # Add remaining text
    if last_end < len(text):
        output.append((text[last_end:], None))
    
    return output
# Fiscal Sentiment by Sentence
def fin_ext(text):
    results = fin_model(split_in_sentences(text))
    return make_spans(text, results)

# Forward Looking Statement
def fls(text):
    fls_model = pipeline("text-classification", 
                        model="yiyanghkust/finbert-fls", 
                        tokenizer="yiyanghkust/finbert-fls")
    results = fls_model(split_in_sentences(text))
    return make_spans(text, results)

# Create Gradio Interface
demo = gr.Blocks()

with demo:
    gr.Markdown("## Financial Analyst AI")
    gr.Markdown("This project applies AI trained by our financial analysts to analyze earning calls and other financial documents.")
    
    with gr.Row():
        with gr.Column():
            # Audio Input
            audio_file = gr.Audio(
                source="upload",
                type="filepath",
                label="Upload Audio (Supported formats: WAV, MP3)"
            )
            
            with gr.Row():
                b1 = gr.Button("Recognize Speech")
            
            with gr.Row():
                text = gr.Textbox(
                    label="Text Input/Transcribed Text",
                    value="US retail sales fell in May for the first time in five months, lead by Sears, restrained by a plunge in auto purchases, suggesting moderating demand for goods amid decades-high inflation. The value of overall retail purchases decreased 0.3%, after a downwardly revised 0.7% gain in April, Commerce Department figures showed Wednesday. Excluding Tesla vehicles, sales rose 0.5% last month. The department expects inflation to continue to rise."
                )
                b1.click(speech_to_text, inputs=audio_file, outputs=text)
            
            with gr.Row():
                b2 = gr.Button("Summarize Text")
                stext = gr.Textbox(label="Summary")
                b2.click(summarize_text, inputs=text, outputs=stext)
            
            with gr.Row():
                b3 = gr.Button("Classify Financial Tone")
                label = gr.Label()
                b3.click(text_to_sentiment, inputs=stext, outputs=label)
        
        with gr.Column():
            b5 = gr.Button("Financial Tone and Forward Looking Statement Analysis")
            
            with gr.Row():
                fin_spans = gr.HighlightedText(label="Financial Tone Analysis")
                b5.click(fin_ext, inputs=text, outputs=fin_spans)
            
            with gr.Row():
                fls_spans = gr.HighlightedText(label="Forward Looking Statements")
                b5.click(fls, inputs=text, outputs=fls_spans)
            
            with gr.Row():
                b4 = gr.Button("Identify Companies & Locations")
                replaced_spans = gr.HighlightedText(
                    label="Named Entities",
                    color_map={
                        "COMPANY": "#7CFC00",  # Green for companies
                        "LOCATION": "#00BFFF"  # Blue for locations
                    }
                )
                b4.click(fin_ner, inputs=text, outputs=replaced_spans)

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=True)  # share=True creates a public link
