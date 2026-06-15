import gradio as gr

from core.analysis import fin_ext, fin_ner, fls, speech_to_text, summarize_text, text_to_sentiment
from core.config import NER_LABEL_COLORS
from core.shap_utils import (
    explain_finbert,
    explain_fls_waterfall_plot,
    global_fls_shap_summary,
    global_shap_summary,
)
from ui.css import CUSTOM_CSS


def build_app():
    demo = gr.Blocks()
    with demo:
        gr.HTML(CUSTOM_CSS)
        gr.Image(
            "https://lh3.googleusercontent.com/pw/AP1GczMzUegdq6DGcyV69iCQ9SqCFfTiS6t8MwMxQNdJP2pwtXEX4KLpMJhIGjVgdoHs9vezs8eA-MWs7GBZsRqzrIVdH0fIrcMHiWiKl3L3O0yt7mShvQ=w2400",
            elem_id="logo",
        )
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
                entities_spans = gr.HighlightedText(
                    label="Named Entities",
                    color_map=NER_LABEL_COLORS,
                )

                analyze_button = gr.Button("Analyze")
                analyze_button.click(fin_ext, inputs=text_input, outputs=fin_spans)
                analyze_button.click(fls, inputs=text_input, outputs=fls_spans)
                analyze_button.click(fin_ner, inputs=text_input, outputs=entities_spans)

            with gr.TabItem("FLS Decision Plot"):
                fls_text_input = gr.Textbox(
                    label="Enter a financial statement",
                    lines=6,
                    placeholder="Paste a forward-looking statement here...",
                )
                fls_decision_html = gr.HTML(value="<i>Decision plot will appear here...</i>")
                fls_decision_button = gr.Button("Explain FLS Decision")

                fls_decision_button.click(
                    explain_fls_waterfall_plot,
                    inputs=fls_text_input,
                    outputs=fls_decision_html,
                )

            with gr.TabItem("Global SHAP Summary"):
                multi_text_input = gr.Textbox(
                    label="Paste multiple financial texts (separate by newline)",
                    lines=10,
                    placeholder="Text 1\nText 2\nText 3...",
                )
                global_shap_html = gr.HTML(label="Global SHAP Feature Importance")
                global_button = gr.Button("Generate Global Summary")
                global_button.click(
                    lambda txt: global_shap_summary(txt.split("\n")),
                    inputs=multi_text_input,
                    outputs=global_shap_html,
                )

            with gr.TabItem("Global FLS SHAP Summary"):
                multi_fls_input = gr.Textbox(
                    label="Paste multiple financial texts (separate by newline)",
                    lines=10,
                    placeholder="Text 1\nText 2\nText 3...",
                )
                global_fls_html = gr.HTML(label="Global FLS SHAP Feature Importance")
                global_fls_button = gr.Button("Generate Global FLS Summary")
                global_fls_button.click(
                    lambda txt: global_fls_shap_summary(txt.split("\n")),
                    inputs=multi_fls_input,
                    outputs=global_fls_html,
                )

    return demo
