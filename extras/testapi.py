from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

def fin_ner(text):
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    entities = ner_pipeline(text)
    # Extract only the names of the companies
    company_names = []
    for entity in entities:
        if entity['entity_group'] == 'ORG':
            company_names.append(entity['word'])

    return company_names
    
text=("US retail sales fell in May for the first time in five months, lead by Sears, restrained by a plunge in auto purchases, suggesting moderating demand for goods amid decades-high inflation. The value of overall retail purchases decreased 0.3%, after a downwardly revised 0.7% gain in April, Commerce Department figures showed Wednesday. Excluding Tesla vehicles, sales rose 0.5% last month. The department expects inflation to continue to rise.")
                
results = fin_ner(text)
print("Detected Entities:")
print(results)