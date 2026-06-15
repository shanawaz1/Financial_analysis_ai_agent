import spacy
import gradio as gr

# Load a better model for company recognition
nlp = spacy.load("en_core_web_lg")  # Use large model for better accuracy

def identify_entities(text):
    doc = nlp(text)
    entities = {
        "Companies": [],
        "Locations": [],
        "Other": []
    }
    
    for ent in doc.ents:
        # Filter for organizations with more than 1 character
        if ent.label_ == "ORG" and len(ent.text) > 2:
            entities["Companies"].append(ent.text)
        elif ent.label_ == "GPE" or ent.label_ == "LOC":
            entities["Locations"].append(ent.text)
        else:
            entities["Other"].append(f"{ent.text} ({ent.label_})")
    
    # Format output for Gradio
    output = []
    if entities["Companies"]:
        output.append("**Companies:**\n" + "\n".join(entities["Companies"]))
    if entities["Locations"]:
        output.append("**Locations:**\n" + "\n".join(entities["Locations"]))
    
    return "\n\n".join(output) if output else "No entities found"

# Create Gradio interface
iface = gr.Interface(
    fn=identify_entities,
    inputs=gr.Textbox(label="Input Text", placeholder="Enter text with companies and locations..."),
    outputs=gr.Textbox(label="Identified Entities"),
    title="Company & Location Extractor",
    description="Identify organizations and locations in text using spaCy NER"
)

iface.launch()