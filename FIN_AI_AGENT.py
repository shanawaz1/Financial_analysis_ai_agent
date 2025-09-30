import os
import gradio as gr
import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import spacy
# Install SHAP if not already installed
os.system("pip install shap")

import shap
import numpy as np

# Install required packages
os.system("pip install gradio==3.0.18")
os.system("pip install librosa")
os.system("pip install soundfile")

# Load spaCy model
transformers.logging.set_verbosity_debug()
nlp = spacy.load("en_core_web_lg")  # Use large model for better accuracy
nlp.add_pipe('sentencizer')
text_masker = shap.maskers.Text(" ")

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
    device=-1  # Use GPU if available
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

# Labels for FinBERT
finbert_labels = ["positive", "negative", "neutral"]

def finbert_predict(texts):
    clean_texts = []
    for t in texts:
        if isinstance(t, (list, np.ndarray)):   # SHAP may pass arrays/tokens
            clean_texts.append(" ".join(map(str, t)))
        else:
            clean_texts.append(str(t))

    outputs = fin_model(clean_texts)  # Batch input
    prob_vectors = []
    for out in outputs:
        probs = [0.0, 0.0, 0.0]
        label = out['label'].lower()
        if label == "positive":
            probs[0] = out['score']
        elif label == "negative":
            probs[1] = out['score']
        elif label == "neutral":
            probs[2] = out['score']
        total = sum(probs)
        if total > 0:
            probs = [p/total for p in probs]
        prob_vectors.append(probs)
    return np.array(prob_vectors)


# SHAP Explainer for FinBERT

finbert_explainer = shap.Explainer(finbert_predict, text_masker)
def explain_finbert(text):
    shap_values = finbert_explainer([text])
    html_plot = shap.plots.text(shap_values[0], display=False)

    # Extract raw probabilities
    probs = finbert_predict([text])[0]
    labels = ["Positive", "Negative", "Neutral"]

    # Styled probability table
    prob_table = """
    <div style='text-align:center;'>
    <table style='margin:auto; border-collapse: collapse; width:60%;'>
    <tr><th style='text-align:right;'>Sentiment</th><th style='text-align:right;'>Probability</th></tr>
    """
    for label, score in zip(labels, probs):
        prob_table += f"<tr><td style='text-align:right;'>{label}</td><td style='text-align:right;'>{score:.2f}</td></tr>"
    prob_table += "</table></div>"

    # Combine both views
    combined_html = f"<h3>Model Explanation</h3>{html_plot}<br><h3>Prediction Confidence</h3>{prob_table}"
    return combined_html
def keyword_boost(text, probs):
    keywords = ["expect", "forecast", "will", "project", "anticipated", "next quarter","follow"]
    if any(kw in text.lower() for kw in keywords) and probs[0] < 0.5:
        return [min(probs[0] + 0.3, 1.0), max(probs[1] - 0.3, 0.0)]
    return probs
# Define prediction function for FLS model
# FLS prediction wrapper
fls_labels = ["Forward-Looking", "Not Forward-Looking"]
# Load FLS pipeline once globally
fls_model = pipeline("text-classification", 
                     model="yiyanghkust/finbert-fls", 
                     tokenizer="yiyanghkust/finbert-fls")

def fls_predict_scalar_boosted(texts):
    boosted_scores = []
    for text in texts:
        raw_probs = fls_predict([text])[0]
        boosted = keyword_boost(text, raw_probs)
        boosted_scores.append(boosted[0])  # Forward-Looking score
    return np.array(boosted_scores)


fls_boosted_explainer = shap.Explainer(fls_predict_scalar_boosted, text_masker)

def explain_fls_waterfall_plot(text):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io, base64
    import numpy as np

    # Get SHAP values for scalar output
    shap_values = fls_boosted_explainer([text])

    # Get prediction class and confidence
    raw_probs = fls_predict([text])[0]
    boosted_probs = keyword_boost(text, raw_probs)
    pred_class = np.argmax(boosted_probs)
    pred_label = fls_labels[pred_class]

    # Generate waterfall plot
    fig = plt.figure(figsize=(10, 5))
    shap.plots.waterfall(shap_values[0], show=False)

    # Convert to base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)

    # Styled probability table
    prob_table = """
    <div style='text-align:center;'>
    <table style='margin:auto; border-collapse: collapse; width:50%;'>
    <tr><th style='text-align:right;'>Class</th><th style='text-align:right;'>Probability</th></tr>
    """
    for label, score in zip(fls_labels, boosted_probs):
        prob_table += f"<tr><td style='text-align:right;'>{label}</td><td style='text-align:right;'>{score:.2f}</td></tr>"
    prob_table += "</table></div>"

    return f"""
    <h3>FLS SHAP Waterfall Plot (Predicted: {pred_label})</h3>
    <div style='text-align:center;'>
        <img src="data:image/png;base64,{img_str}" width="800"/>
    </div>
    <h3>Prediction Confidence</h3>{prob_table}
    """

def fls_predict(texts):
    clean_texts = []
    for t in texts:
        if isinstance(t, (list, np.ndarray)):   # SHAP may pass arrays/tokens
            clean_texts.append(" ".join(map(str, t)))
        else:
            clean_texts.append(str(t))

    outputs = fls_model(clean_texts)

    prob_vectors = []
    for out in outputs:
        if out['label'].lower() == "forward-looking":
            prob_vectors.append([out['score'], 1 - out['score']])
        else:
            prob_vectors.append([1 - out['score'], out['score']])
    return np.array(prob_vectors)
# SHAP explainer for FLS
fls_explainer = shap.Explainer(fls_predict, shap.maskers.Text(" "))

def global_shap_summary(text_list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io
    import base64
    import numpy as np

    clean_texts = [str(t) for t in text_list if isinstance(t, str) and len(t.strip()) > 0]
    if not clean_texts:
        return "No valid input texts provided."

    shap_values = finbert_explainer(clean_texts)

    token_scores = {}
    for sv in shap_values:
        tokens = sv.data
        values = sv.values
        for token, val in zip(tokens, values):
            score = float(np.sum(np.abs(val))) if isinstance(val, np.ndarray) else abs(val)
            token_scores[token] = token_scores.get(token, 0.0) + score

    sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)[:15]
    tokens, scores = zip(*sorted_tokens)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(tokens[::-1], scores[::-1], color="#009a4d")
    ax.set_title("Top 15 Influential Tokens Across Financial Texts", fontsize=14)
    ax.set_xlabel("Aggregate SHAP Value (Absolute)", fontsize=12)
    ax.tick_params(axis='y', labelsize=11)

    for bar, score in zip(bars, scores[::-1]):
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2, f"{score:.2f}", va='center', fontsize=10)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)

    return f'<img src="data:image/png;base64,{img_str}" width="700"/>'

def global_fls_shap_summary(text_list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io
    import base64
    import numpy as np

    clean_texts = [str(t) for t in text_list if isinstance(t, str) and len(t.strip()) > 0]
    if not clean_texts:
        return "No valid input texts provided."

    shap_values = fls_explainer(clean_texts)

    # Aggregate SHAP values across all texts
    token_scores = {}
    for sv in shap_values:
        tokens = sv.data
        values = sv.values
        for token, val in zip(tokens, values):
            score = float(np.sum(np.abs(val))) if isinstance(val, np.ndarray) else abs(val)
            token_scores[token] = token_scores.get(token, 0.0) + score

    # Sort and select top tokens
    sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)[:15]
    tokens, scores = zip(*sorted_tokens)

    # Plot using matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(tokens[::-1], scores[::-1], color="#0077b6")
    ax.set_title("Top 15 Influential Tokens for Forward-Looking Detection", fontsize=14)
    ax.set_xlabel("Aggregate SHAP Value (Absolute)", fontsize=12)
    ax.tick_params(axis='y', labelsize=11)

    for bar, score in zip(bars, scores[::-1]):
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2, f"{score:.2f}", va='center', fontsize=10)

    plt.tight_layout()

    # Convert to base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)

    return f'<img src="data:image/png;base64,{img_str}" width="700"/>'

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
        
        with gr.TabItem("FLS Decision Plot"):
            fls_text_input = gr.Textbox(
                label="Enter a financial statement",
                lines=6,
                placeholder="Paste a forward-looking statement here..."
            )
            fls_decision_html = gr.HTML(value="<i>Decision plot will appear here...</i>")
            fls_decision_button = gr.Button("Explain FLS Decision")

            fls_decision_button.click(
                explain_fls_waterfall_plot,
                inputs=fls_text_input,
                outputs=fls_decision_html
            )

        with gr.TabItem("Global SHAP Summary"):
            multi_text_input = gr.Textbox(
                label="Paste multiple financial texts (separate by newline)",
                lines=10,
                placeholder="Text 1\nText 2\nText 3..."
            )
            global_shap_html = gr.HTML(label="Global SHAP Feature Importance")
            global_button = gr.Button("Generate Global Summary")
            global_button.click(
                lambda txt: global_shap_summary(txt.split("\n")),
                inputs=multi_text_input,
                outputs=global_shap_html
            )
        
        # Launch the interface
if __name__ == "__main__":
    demo.launch(share=True)  # share=True creates a public link
