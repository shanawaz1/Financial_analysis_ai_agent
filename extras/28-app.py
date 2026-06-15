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
custom_css = """

<style>
body, html {
    height: 100%;
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    background-color: #808000; /* Ice blue */
    color: #000000; /* Darker text for contrast */
}

.gradio-container {
    background-color: rgba(255, 255, 255, 0.8);  /* Light overlay for readability */
    padding: 30px;
    border-radius: 15px;
    max-width: 900px;
    margin: auto;
    box-shadow: 0 0 15px rgba(0,0,0,0.2);
}


#logo {
    position: absolute;
    top: 20px;
    right: 20px;
    width: 120px;
    height: auto;
    z-index: 1000;
}

button {
    background-color: #009a4d;
    border: none;
    color: black;
    padding: 12px 20px;
    text-align: center;
    font-size: 16px;
    border-radius: 10px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}

button:hover {
    background-color: #8fbc8f;
}

/* Style selected/active tab */
.tab-nav button[aria-selected="true"] {
    background-color: #8fbc8f !important;
    color: black !important;
    font-weight: bold;
}
#logo {
    position: absolute;
    top: 5px;
    right: 20px;
    width: 80px;       /* Adjust this value to make it smaller */
    height: auto;
    z-index: 1000;
}

h1, h2, h3, h4 {
    color: #000000;
}

input, textarea {
    border-radius: 8px;
    border: 1px solid #ccc;
    padding: 10px;
    font-size: 16px;
    width: 100%;
    box-sizing: border-box;
    color: #000000;
    background-color: #000000;
}
</style>
"""

# Create Gradio Interface
demo = gr.Blocks()

with demo:
    gr.HTML(custom_css)  # Apply custom CSS
    gr.Image("https://lh3.googleusercontent.com/pw/AP1GczMzUegdq6DGcyV69iCQ9SqCFfTiS6t8MwMxQNdJP2pwtXEX4KLpMJhIGjVgdoHs9vezs8eA-MWs7GBZsRqzrIVdH0fIrcMHiWiKl3L3O0yt7mShvQ=w2400", elem_id="logo")
    gr.Markdown("## Financial Analyst AI")
    gr.Markdown("This project applies AI trained to analyze earning calls and financial documents.")
    
    with gr.Tabs():
        with gr.TabItem("Speech Recognition"):
            audio_file = gr.Audio(source="upload", type="filepath", label="Upload Audio (WAV, MP3)")
            text = gr.Textbox(label="Transcribed Text", lines=5, placeholder="Speech transcript output goes here...")
            b1 = gr.Button("Recognize Speech")
            b1.click(speech_to_text, inputs=audio_file, outputs=text)
        
        with gr.TabItem("Summarization & Tone"):
            text_input = gr.Textbox(label="Input Text", lines=8, placeholder="Paste or type your text here...")
            summary_output = gr.Textbox(label="Summary", lines=4)
            b2 = gr.Button("Summarize")
            b3 = gr.Button("Classify Financial Tone")
            tone_label = gr.Label()
            
            b2.click(summarize_text, inputs=text_input, outputs=summary_output)
            b3.click(text_to_sentiment, inputs=summary_output, outputs=tone_label)
        
        with gr.TabItem("In-depth Analysis"):
            fin_spans = gr.HighlightedText(label="Financial Tone Analysis")
            fls_spans = gr.HighlightedText(label="Forward Looking Statements")
            entities_spans = gr.HighlightedText(label="Named Entities", 
                                                color_map={"COMPANY": "#7CFC00", "LOCATION": "#00BFFF"})
            
            analyze_button = gr.Button("Analyze")
            analyze_button.click(fin_ext, inputs=text_input, outputs=fin_spans)
            analyze_button.click(fls, inputs=text_input, outputs=fls_spans)
            analyze_button.click(fin_ner, inputs=text_input, outputs=entities_spans)
# Launch the interface
if __name__ == "__main__":
    demo.launch(share=True)  # share=True creates a public link
