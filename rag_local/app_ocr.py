# app.py
import os
import streamlit as st
import traceback
from dotenv import load_dotenv
from pypdf import PdfReader
import fitz  # PyMuPDF
from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import clean
import base64
from io import BytesIO
from PIL import Image
import re
import pandas as pd

# Langchain and Ollama Imports
from langchain_core.messages import HumanMessage
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain

# <-- NEW: Imports for the local Nanonets OCR model
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
# --------------------------------------------------------------------

# -------------- GLOBAL VARIABLES & SESSION STATE SETUP -------------
load_dotenv()

# RAG and LLM setup
model_path = os.getenv("OLLAMA_MODEL_PATH", "gemma3:27b")
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
OLLAMA_EMBEDDING_MODEL = "bge-m3"

# Session state initialization
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "ocr_model" not in st.session_state: # <-- NEW
    st.session_state.ocr_model = None
if "ocr_processor" not in st.session_state: # <-- NEW
    st.session_state.ocr_processor = None

# Initialize main LLM
try:
    llm = ChatOllama(model=model_path, temperature=0.0)
    st.session_state.llm_initialized = True
except Exception as e:
    st.error(f"Could not initialize main LLM via Ollama: {e}")
    llm = None
    st.session_state.llm_initialized = False
# --------------------------------------------------------------------


# ---------------------- LOCAL OCR FUNCTIONS (NEW) ----------------------
@st.cache_resource
def load_ocr_model():
    """
    Loads and caches the Nanonets OCR model and processor.
    This function runs only once.
    """
    st.write("DEBUG: Loading Nanonets OCR model for the first time... ⏳")
    try:
        model_path = "nanonets/Nanonets-OCR-s"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation="sdpa"
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(model_path)
        st.write("DEBUG: Nanonets OCR model loaded successfully.")
        return model, processor
    except Exception as e:
        st.error(f"Error loading Nanonets OCR model: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None

def perform_nanonets_ocr(pil_image: Image.Image, ocr_model, ocr_processor) -> str:
    """
    Performs OCR on a given PIL image using the loaded Nanonets model.
    """
    if not ocr_model or not ocr_processor:
        return "OCR model not available."
    try:
        # Prepare the image and prompt for the OCR model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Extract all text from this document."}
                ]
            }
        ]
        prompt = ocr_processor.apply_chat_template(messages, add_generation_prompt=True)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        inputs = ocr_processor(text=prompt, images=[pil_image], return_tensors="pt").to("cuda", dtype=dtype)

        # Generate text
        generated_ids = ocr_model.generate(**inputs, max_length=4096)
        generated_text = ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Post-process the result to get only the final answer
        # The model output is often a JSON-like structure, we parse the final answer
        try:
            # Find the start of the final answer part
            answer_start = generated_text.rfind("final_answer")
            if answer_start != -1:
                # A bit of parsing to clean up the output
                text_part = generated_text[answer_start:]
                text_part = text_part.replace("\\n", "\n").replace('\\"', '"')
                # Find the text between the first and last quote
                first_quote = text_part.find('"') + 1
                last_quote = text_part.rfind('"')
                if first_quote < last_quote:
                    return text_part[first_quote:last_quote]
            return generated_text # Fallback to raw text if parsing fails
        except Exception:
            return generated_text # Fallback

    except Exception as e:
        st.error(f"Error during OCR processing: {e}")
        return f"Error during OCR: {e}"
# ---------------------- END OCR FUNCTIONS ----------------------


# ---------------------- PROMPTS (UNCHANGED) ----------------------
MAP_PROMPT = PromptTemplate(
    template="""
Use the following DOCUMENT EXCERPT to answer the QUESTION as accurately as possible.
The excerpt may contain text extracted directly from images via OCR.
If the excerpt does not contain the answer, respond exactly with "Information not found in this chunk."

Document excerpt:
{context}

Question:
{question}

Intermediate Answer:""",
    input_variables=["context", "question"]
)

COMBINE_PROMPT = PromptTemplate(
    template="""
You have been given multiple INTERMEDIATE ANSWERS from different excerpts, some of which may include direct text from OCR'd images.
Combine them into a final, coherent answer to the QUESTION below.
If none of the intermediate answers contain the requested information, respond exactly with "The answer is not available in the provided documents."

IMPORTANT: When presenting mathematical equations, formulas, or expressions,
please use standard LaTeX notation.
For example:
- For a fraction, use `\\frac{{numerator}}{{denominator}}`.
- For an integral, use `\\int_{{a}}^{{b}} f(x) dx`.
- For sums, use `\\sum_{{i=1}}^{{n}} x_i`.
- Enclose inline LaTeX with single dollar signs (e.g., `$E=mc^2$`).
- Enclose display-style equations (equations on their own line) with double dollar signs (e.g., `$$P(A|B) = \\frac{{P(B|A)P(A)}}{{P(B)}}$$`).

Intermediate answers:
{summaries}

Question:
{question}

Final Answer (formatted with LaTeX for equations):""",
    input_variables=["summaries", "question"]
)
# ---------------------------------------------------------------------

# ---------------------- EMBEDDING & VECTOR STORE (UNCHANGED) ----------------------
@st.cache_resource
def load_ollama_embeddings():
    st.write(f"DEBUG: Attempting to load Ollama embedding model: {OLLAMA_EMBEDDING_MODEL}")
    try:
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
        _ = embeddings.embed_query("Test query to initialize and check the model.")
        st.write("DEBUG: Ollama embeddings loaded successfully.")
        return embeddings
    except Exception as e:
        st.error(f"Error loading Ollama embeddings model '{OLLAMA_EMBEDDING_MODEL}': {e}")
        return None

@st.cache_data
def get_vector_store(text_chunks):
    if not text_chunks:
        return None
    try:
        embeddings = load_ollama_embeddings()
        if embeddings is None:
            return None
        return FAISS.from_texts(text_chunks, embedding=embeddings)
    except Exception as e:
        st.error(f"Error creating FAISS store: {e}")
        return None
# ----------------------------------------------------------------------


# ---------------------- PDF TEXT & IMAGE EXTRACTION (MODIFIED) ---------------------------
def get_text_chunks(full_text: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_text(full_text)

def get_documents_from_pdfs(pdf_files):
    all_text = []
    # <-- MODIFIED: Get the OCR model and processor from session state
    ocr_model = st.session_state.get("ocr_model")
    ocr_processor = st.session_state.get("ocr_processor")

    for pdf_file in pdf_files:
        st.write(f"DEBUG: Processing PDF: {pdf_file.name}")
        try:
            pdf_file.seek(0)
            raw_bytes = pdf_file.read()
            pdf_file.seek(0)

            # Use Unstructured for text extraction
            elements = partition_pdf(file=pdf_file, strategy="hi_res", infer_table_structure=True)
            texts = []
            for el in elements:
                if "Table" in str(type(el)):
                    html = getattr(el.metadata, "text_as_html", None)
                    if html: texts.append(f"\n[TABLE HTML START]\n{html}\n[TABLE HTML END]\n")
                    else: texts.append(f"\n[TABLE START]\n{el.text}\n[TABLE END]\n")
                else:
                    cleaned = clean(el.text, bullets=True, extra_whitespace=True)
                    if cleaned.strip(): texts.append(cleaned)
            
            if texts:
                all_text.append("\n\n".join(texts))
                st.write(f"DEBUG: Extracted text from {pdf_file.name} using Unstructured.")
            else:
                raise ValueError("Unstructured found no primary text")

        except Exception as e_unstructured:
            # Fallback to PyPDF if Unstructured fails
            st.write(f"DEBUG: Unstructured failed: {e_unstructured}. Falling back to PyPDF.")
            try:
                pdf_file.seek(0)
                reader = PdfReader(pdf_file)
                page_texts = [page.extract_text() or "" for page in reader.pages]
                if any(page_texts):
                    all_text.append("\n".join(page_texts))
            except Exception as py_err:
                st.error(f"Failed to extract text from {pdf_file.name} with all methods: {py_err}")

        # ------ MODIFIED: IMAGE PROCESSING WITH LOCAL OCR ------
        try:
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
            ocr_texts = []
            if ocr_model and ocr_processor:
                img_counter = 1
                for page_idx in range(len(doc)):
                    for img_info in doc.get_page_images(page_idx):
                        xref = img_info[0]
                        base_img = doc.extract_image(xref)
                        img_bytes = base_img["image"]
                        
                        st.write(f"DEBUG: Performing OCR on image {img_counter} on page {page_idx+1}...")
                        pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
                        
                        # Call our new OCR function
                        extracted_text = perform_nanonets_ocr(pil_img, ocr_model, ocr_processor)
                        
                        ocr_texts.append(
                            f"\n[OCR TEXT FROM IMAGE {img_counter} ON PAGE {page_idx+1} START]\n{extracted_text}\n[OCR TEXT FROM IMAGE {img_counter} END]\n"
                        )
                        img_counter += 1
                if ocr_texts:
                   all_text.append("\n".join(ocr_texts))
                st.write(f"DEBUG: Performed OCR on {img_counter-1} image(s) for {pdf_file.name}.")
            else:
                st.warning("Local OCR model not available. Skipping image OCR.")
            doc.close()
        except Exception as e:
            st.error(f"Error during image OCR for {pdf_file.name}: {e}")
        # ------ END IMAGE PROCESSING ------

    return "\n\n--- DOCUMENT BOUNDARY ---\n\n".join(all_text)
# ---------------------------------------------------------------------

# ---------------------- USER QUERY & QA CHAIN (UNCHANGED) ----------------
def user_input(user_question, vector_store):
    if not vector_store or not llm:
        st.warning("Please process documents first and ensure the LLM is available.")
        return

    try:
        docs = vector_store.similarity_search(user_question, k=3)
        if not docs:
            st.info("No relevant documents found for your question.")
            return

        map_llm_chain = LLMChain(llm=llm, prompt=MAP_PROMPT)
        combine_llm_chain = LLMChain(llm=llm, prompt=COMBINE_PROMPT)
        stuff_combine_documents_chain = StuffDocumentsChain(
            llm_chain=combine_llm_chain, document_variable_name="summaries")
        chain = MapReduceDocumentsChain(
            llm_chain=map_llm_chain,
            reduce_documents_chain=stuff_combine_documents_chain,
            document_variable_name="context",
            input_key="input_documents",
            output_key="output_text")

        with st.spinner("Querying the documents... 🤔 Please wait, this can take a moment."):
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        raw_answer = response.get("output_text", "")
        if raw_answer and raw_answer.lower().strip() not in ["information not found in this chunk.", "the answer is not available in the provided documents."]:
            st.subheader("Answer:")
            st.markdown(raw_answer, unsafe_allow_html=True)
        else:
            st.info("The answer is not available in the provided documents.")
            with st.expander("Show Retrieved Context Chunks"):
                for i, doc_ret in enumerate(docs):
                    st.caption(doc_ret.page_content[:500] + "...")

    except Exception as e:
        st.error(f"Error during question processing: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
# --------------------------------------------------------------------


# ---------------------- STREAMLIT LAYOUT (MODIFIED) ------------------------------
st.set_page_config(page_title="📄 Offline RAG with Local OCR", layout="wide")
st.title("🤖 RAG with Ollama, FAISS, and Local Nanonets OCR")

with st.sidebar:
    st.subheader("Upload and Process PDFs")
    pdf_docs = st.file_uploader("Upload PDF files here", type=["pdf"], accept_multiple_files=True)

    # <-- MODIFIED: Initialize the OCR model on button click
    if st.session_state.ocr_model is None:
        if st.button("Initialize Local OCR Model"):
             st.session_state.ocr_model, st.session_state.ocr_processor = load_ocr_model()
             st.rerun() # Rerun to update the state display
    else:
        st.success("✅ Local OCR Model is Initialized.")

    # Modified button logic to ensure OCR model is loaded first
    process_button_disabled = not pdf_docs or not st.session_state.ocr_model
    if st.button("Process Documents 🚀", disabled=process_button_disabled):
        with st.spinner("Extracting text, running OCR, and building vector store... ⏳"):
            full_text = get_documents_from_pdfs(pdf_docs)
            if full_text.strip():
                chunks = get_text_chunks(full_text)
                if chunks:
                    st.session_state.vector_store = get_vector_store(chunks)
                    st.success(f"Processed {len(pdf_docs)} document(s). Ready for questions.")
                else: st.warning("No text chunks generated.")
            else: st.error("No text extracted from PDFs.")

if st.session_state.vector_store:
    st.info("Ready to answer questions based on processed documents.")
else:
    st.info("Upload PDF(s), initialize the OCR model, and process documents to start querying.")

st.subheader("Ask a Question 🤔")
user_question_disabled = not st.session_state.vector_store
user_question = st.text_input("Type your question here:", key="user_question_input", disabled=user_question_disabled)

if user_question:
    user_input(user_question, st.session_state.vector_store)
# --------------------------------------------------------------------