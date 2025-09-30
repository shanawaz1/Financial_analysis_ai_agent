Below is a representation of the app.
![image](https://github.com/user-attachments/assets/97f64b94-f47b-4151-b352-57cd9540df73)


# Financial Analyst AI Pipeline  

An AI-powered pipeline designed to assist financial professionals by automatically processing financial text and audio (e.g., earnings calls, financial reports) to provide **sentiment analysis**, **forward-looking statement (FLS) detection**, **named entity recognition (NER)**, and **interpretable insights** via an interactive user interface.  

---

## 🚀 Features  
- **Automatic Speech Recognition (ASR):**  
  Converts audio from earnings calls into text using **Whisper**.  

- **Sentiment Analysis:**  
  Detects positive, negative, or neutral sentiment using **FinBERT-tone**, optimized for financial text.  

- **Forward-Looking Statement Detection:**  
  Identifies statements about future expectations using **FinBERT-FLS** with keyword boosting.  

- **Named Entity Recognition (NER):**  
  Extracts entities like company names, monetary amounts, and dates using **spaCy**.  

- **Interpretability:**  
  Explanations generated with **SHAP**, giving users transparency into model decisions.  

- **User Interface:**  
  A **Gradio-based UI** to make the pipeline accessible to non-technical users.  

---

## 🏗️ Architecture  
**Audio/Text Input → ASR (Whisper) → NLP Modules (Sentiment, FLS, NER) → SHAP Explanations → Gradio UI Output**  

![Pipeline Flow](pipeline_flow.png)  

---

## 📊 Example Use Case  
- Upload an **earnings call audio**.  
- The system transcribes speech, classifies sentiment of statements, detects forward-looking guidance, extracts company and financial terms, and provides **explainable AI visualizations**.  
- Results are displayed in a **simple interactive dashboard** for analysts.  

---

## ⚙️ Installation  

### Requirements  
- Python 3.9+  
- Recommended: GPU for faster inference  

### Setup  
```bash
# Clone the repository
git clone https://github.com/your-username/financial-ai-pipeline.git
cd financial-ai-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies  
- [transformers](https://huggingface.co/transformers/)  
- [torch](https://pytorch.org/)  
- [whisper](https://github.com/openai/whisper)  
- [spacy](https://spacy.io/)  
- [shap](https://shap.readthedocs.io/en/latest/)  
- [gradio](https://www.gradio.app/)  

---

## ▶️ Usage  

### Run the Pipeline  
```bash
python FIN_AI_AGENT.py
```

### Access the Interface  
Open the Gradio link (http://127.0.0.1:7860/) in your browser.  

---

## 📂 Project Structure  
```
financial-ai-pipeline/
│── app.py                  # Main Gradio app
│── pipeline.py             # Core pipeline logic
│── models/                 # Pretrained model configs
│── data/                   # Sample financial texts/audio
│── utils/                  # Helper functions
│── requirements.txt        # Dependencies
│── README.md               # Project documentation
```

---

## 🧪 Evaluation  
- **Sentiment (FinBERT-tone):** 94.2% on Financial PhraseBank, 89.7% on FiQA  
- **FLS (FinBERT-FLS + keyword boosting):** 90.9% F1-score  
- **ASR (Whisper-large-v3):** Avg. WER ~9.8%  
- **NER (spaCy):** 88% F1-score  

---


## 🤝 Contributing  
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to improve.  

---

## 📜 License  
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.  

---

## 🙌 Acknowledgments  
- OpenAI **Whisper**  
- Hugging Face **FinBERT-tone** and **FinBERT-FLS**  
- **spaCy** for NER  
- **SHAP** for interpretability  
- **Gradio** for UI  

