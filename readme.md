
# To Build a RAG Pipeline to Read PDF via Streamlit App

# Requirement to run App

1. Install all the dependencies mentioned in requirements.txt file
2. Set up you API Key
    You'll need to configure the OPENAPI_KEY to use it. In the RAG_PDF_Reader_Pipeline.ipynb and app.py file, find the line that looks like this:

    os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

    Replace the OPENAI_API_KEY with your API key

3. Replace the path of your PDF file, find the line that looks like this:

    pdf_path = "path/Sample.pdf"
   
    Replace path/Sample.pdf with your pdf file path

4. Launch the app

    streamlit run app.py

    The application should now be up and running on your local server. 

### Interpretation of Evaluation Scores

1. Cosine similarity: a measure of the cosine of the angle between two non-zero vectors, interpreted as a value between -1 and 1 where 1 indicates identical orientation, 0 indicates orthogonality (no similarity), and -1 indicates opposite orientation.

2. METEOR score: a metric for evaluating the quality of machine-generated text by comparing it to human-written reference texts, interpreting higher scores as indicating closer similarity to the reference. 

3. ROUGE Scores:
ROUGE-1: Measures the overlap of unigrams (single words).
ROUGE-2: Measures the overlap of bigrams (two-word sequences).
ROUGE-L: Measures the longest common subsequence.
Score range: 0 (no overlap) to 1 (perfect overlap).


