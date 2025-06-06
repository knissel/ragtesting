`python -m venv venv `<br />
`venv\Scripts\activate` <br />
`pip install unstructured[local-inference] pymupdf Pillow langchain-google-genai`
>  - For unstructured with OCR, you might need Tesseract installed on your system:
> -On Ubuntu: sudo apt-get install tesseract-ocr
>- On macOS: brew install tesseract
>- For Windows, download installer from Tesseract GitHub.
>- You might also need poppler for PDF processing with unstructured
>- On Ubuntu: sudo apt-get install poppler-utils
>- On macOS: brew install poppler 

`pip install streamlit google-generativeai langchain-google-genai langchain-community faiss-cpu langchain pypdf python-dotenv fitz unstructured[local-inference] Pillow markdownify`

`pip install streamlit google-generativeai langchain-google-genai langchain-community faiss-cpu langchain pypdf python-dotenv PyMuPDF "unstructured[local-inference]" Pillow markdownify`

`pip install langchain-community ollama`

.env -> GOOGLE_API_KEY=""