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

# Import HumanMessage
from langchain_core.messages import HumanMessage

# ---------------------- IMPORT FOR EMBEDDINGS (CHANGED) ----------------------
# from sentence_transformers import SentenceTransformer             # <-- REMOVED
# from langchain_huggingface import HuggingFaceEmbeddings           # <-- REMOVED
from langchain_community.embeddings import OllamaEmbeddings     # <-- ADDED
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

# For manual chain construction
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
# --------------------------------------------------------------------
# --------------------------------------------------------------------

# -------------- GLOBAL VARIABLES & SESSION STATE SETUP -------------
load_dotenv()

model_path = os.getenv("OLLAMA_MODEL_PATH", "gemma3:27b")

try:
    llm = ChatOllama(model=model_path, temperature=0.0)
    st.session_state.llm_initialized = True
except Exception as e:
    st.error(f"Could not initialize local Gemma 3 via Ollama: {e}")
    llm = None
    st.session_state.llm_initialized = False

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
# LOCAL_EMBEDDING_PATH = r"C:\Users\kniss\rag\models\all-MiniLM-L6-v2" # <-- REMOVED
OLLAMA_EMBEDDING_MODEL = "bge-m3" # <-- ADDED: Name of the embedding model in Ollama

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "vision_model" not in st.session_state:
    st.session_state.vision_model = None
# --------------------------------------------------------------------

# ---------------------- VISION FUNCTIONS (UNCHANGED) ----------------------
# (Your vision functions remain the same, so they are omitted here for brevity)
def pil_image_to_base64(pil_img: Image.Image) -> str:
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")

def get_vision_model():
    try:
        model = ChatOllama(model=model_path, temperature=0.3)
        return model
    except Exception as e:
        st.warning(f"Could not initialize local Gemma 3 vision model: {e}")
        return None

def get_image_description(image_bytes: bytes, vision_model_instance):
    if not vision_model_instance:
        return "Image captioning disabled (no local vision model available)."
    try:
        pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
        raw_b64 = pil_image_to_base64(pil_img)
        data_uri = f"data:image/jpeg;base64,{raw_b64}"
        image_block = {"type": "image_url", "image_url": data_uri}
        text_block = {
            "type": "text",
            "text": (
                "Describe this image, focusing on any data, graphs, or formulas visible. "
                "If it's a graph, describe its type, axes, and trend. "
                "If it's a table, extract its content in a structured way. "
                "If it's a formula, represent it in text or LaTeX if possible. "
                "If it's a diagram, explain its components and relationships."
            ),
        }
        human_message = HumanMessage(content=[text_block, image_block])
        response_msg = vision_model_instance.predict_messages([human_message])
        return response_msg.content
    except Exception as e:
        st.error(f"Error generating image description: {e}")
        return "Error generating image description."
# ---------------------- END VISION FUNCTIONS ----------------------


# ---------------------- PROMPTS (UNCHANGED) ----------------------
MAP_PROMPT = PromptTemplate(
    template="""
Use the following DOCUMENT EXCERPT to answer the QUESTION as accurately as possible.
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
You have been given multiple INTERMEDIATE ANSWERS from different excerpts, some of which may include descriptions of images.
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

# ---------------------- EMBEDDING LOADING & CACHING (CHANGED) ----------------------
@st.cache_resource
def load_ollama_embeddings(): # <-- CHANGED: Renamed function for clarity
    """
    Load the bge-m3 embedding model from Ollama.
    """
    st.write(f"DEBUG: Attempting to load Ollama embedding model: {OLLAMA_EMBEDDING_MODEL}")
    try:
        # This is the main change: using OllamaEmbeddings
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
        # You can do a quick test embedding to ensure it's working
        _ = embeddings.embed_query("Test query to initialize and check the model.")
        st.write("DEBUG: Ollama embeddings loaded successfully.")
        return embeddings
    except Exception as e:
        st.error(f"Error loading Ollama embeddings model '{OLLAMA_EMBEDDING_MODEL}': {e}")
        st.error("Please ensure Ollama is running and you have pulled the model with 'ollama pull bge-m3'")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

@st.cache_data
def get_vector_store(text_chunks):
    """
    Build a FAISS index from text_chunks using Ollama bge-m3 embeddings.
    """
    if not text_chunks:
        st.warning("DEBUG: No text chunks provided to get_vector_store.")
        return None
    try:
        # <-- CHANGED: Call the new function
        embeddings = load_ollama_embeddings()
        if embeddings is None:
            st.error("DEBUG: Embeddings are None, cannot create vector store.")
            return None
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating FAISS store: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None
# ----------------------------------------------------------------------

#
# --- The rest of your code (PDF extraction, user input, Streamlit layout) ---
# --- remains exactly the same. It is included below without changes.      ---
#

# ---------------------- PDF TEXT EXTRACTION ---------------------------
def get_text_chunks(full_text: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_text(full_text)

def get_documents_from_pdfs(pdf_files):
    all_text = []
    vision_model_instance = st.session_state.get("vision_model")
    if vision_model_instance is None:
        st.session_state.vision_model = get_vision_model()
        vision_model_instance = st.session_state.vision_model

    for pdf_file in pdf_files:
        st.write(f"DEBUG: Processing PDF: {pdf_file.name}")
        try:
            pdf_file.seek(0)
            raw_bytes = pdf_file.read()
            pdf_file.seek(0)

            elements = partition_pdf(
                file=pdf_file, strategy="hi_res", infer_table_structure=True,
                extract_images_in_pdf=False
            )
            texts = []
            for el in elements:
                if "Table" in str(type(el)):
                    html = getattr(el.metadata, "text_as_html", None)
                    if html:
                        texts.append(f"\n[TABLE HTML START]\n{html}\n[TABLE HTML END]\n")
                    else:
                        texts.append(f"\n[TABLE START]\n{el.text}\n[TABLE END]\n")
                else:
                    cleaned = clean(el.text, bullets=True, extra_whitespace=True)
                    if cleaned.strip():
                        texts.append(cleaned)
            if texts:
                all_text.append("\n\n".join(texts))
                st.write(f"DEBUG: Extracted text from {pdf_file.name} using Unstructured.")
            else:
                st.warning(f"DEBUG: Unstructured found no primary text in {pdf_file.name}, attempting fallback.")
                raise ValueError("Unstructured found no primary text")
        except Exception as e_unstructured:
            st.write(f"DEBUG: Unstructured failed for {pdf_file.name}: {e_unstructured}. Falling back to PyPDF.")
            try:
                pdf_file.seek(0)
                reader = PdfReader(pdf_file)
                page_texts = [page.extract_text() or "" for page in reader.pages]
                fallback_text = "\n".join(page_texts)
                if fallback_text.strip():
                    all_text.append(fallback_text)
                    st.write(f"DEBUG: Extracted text from {pdf_file.name} using PyPDF fallback.")
                else:
                    st.warning(f"No text extracted from {pdf_file.name} via PyPDF fallback.")
            except Exception as py_err:
                st.error(f"Failed to extract PDF text from {pdf_file.name} with PyPDF: {py_err}")

        # ------ IMAGE PROCESSING (INTEGRATED) ------
        try:
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
            image_descriptions = []
            if vision_model_instance:
                img_counter = 1
                for page_idx in range(len(doc)):
                    for img_info in doc.get_page_images(page_idx):
                        xref = img_info[0]
                        base_img = doc.extract_image(xref)
                        img_bytes = base_img["image"]

                        st.write(f"DEBUG: Describing image {img_counter} on page {page_idx+1} of {pdf_file.name}...")
                        desc = get_image_description(img_bytes, vision_model_instance)
                        image_descriptions.append(
                            f"\n[IMAGE {img_counter} ON PAGE {page_idx+1} DESCRIPTION START]\n{desc}\n[IMAGE {img_counter} DESCRIPTION END]\n"
                        )
                        img_counter += 1
                if image_descriptions:
                   all_text.append("\n".join(image_descriptions))

                st.write(f"DEBUG: Described {img_counter-1} image(s) for {pdf_file.name}.")
            else:
                st.warning("Vision model not available. Skipping image descriptions.")
            doc.close()
        except Exception as e:
            st.error(f"Error extracting/describing images from {pdf_file.name}: {e}")
            st.error(f"Traceback: {traceback.format_exc()}")
        # ------ END IMAGE PROCESSING ------

    return "\n\n--- DOCUMENT BOUNDARY ---\n\n".join(all_text)
# ---------------------------------------------------------------------

# ---------------------- USER QUERY & QA CHAIN --------------------------
def user_input(user_question, vector_store):
    if not vector_store:
        st.warning("Please upload and process documents first (vector_store is None).")
        return

    if llm is None or not st.session_state.get("llm_initialized", False):
        st.error("LLM (Gemma 3) is not initialized. Cannot proceed with query.")
        return

    try:
        docs = vector_store.similarity_search(user_question, k=3)
        if not docs:
            st.info("No relevant documents found for your question.")
            return

        map_llm_chain = LLMChain(llm=llm, prompt=MAP_PROMPT)
        combine_llm_chain = LLMChain(llm=llm, prompt=COMBINE_PROMPT)
        stuff_combine_documents_chain = StuffDocumentsChain(
            llm_chain=combine_llm_chain,
            document_variable_name="summaries",
        )
        chain = MapReduceDocumentsChain(
            llm_chain=map_llm_chain,
            reduce_documents_chain=stuff_combine_documents_chain,
            document_variable_name="context",
            input_key="input_documents",
            output_key="output_text",
        )

        input_data = {"input_documents": docs, "question": user_question}

        raw_answer = None
        with st.spinner("Querying the documents... 🤔 Please wait, this can take a moment."):
            response = chain(input_data, return_only_outputs=True)

        if response and isinstance(response, dict) and "output_text" in response:
            raw_answer = response["output_text"]
        elif response and isinstance(response, str):
            raw_answer = response
        else:
            st.warning("The LLM did not return an answer in the expected dictionary format or the 'output_text' key was missing.")

        if raw_answer and raw_answer.strip() and raw_answer.lower() not in [
            "information not found in this chunk.",
            "the answer is not available in the provided documents."
        ]:
            st.subheader("Answer:")
            st.markdown(raw_answer, unsafe_allow_html=True)
        elif raw_answer and raw_answer.lower() in [
            "information not found in this chunk.",
            "the answer is not available in the provided documents."
        ]:
            st.info(raw_answer)
        else:
            st.info("No specific answer was generated by the LLM, or the answer was empty.")

            with st.expander("Show Retrieved Context Chunks"):
                for i, doc_ret in enumerate(docs):
                    snippet = doc_ret.page_content
                    display_snippet = snippet[:500] + "..." if len(snippet) > 500 else snippet
                    st.markdown(f"**Chunk {i+1}:**")
                    st.caption(display_snippet)

    except Exception as e:
        st.error(f"Error during question processing: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        if llm is None or not st.session_state.get("llm_initialized", False):
            st.error("The LLM (Gemma 3 via Ollama) could not be initialized or had an issue. Please check its setup and Ollama server logs.")
        else:
            st.error("An unexpected error occurred. Ensure Ollama and Gemma 3 are running and the model is available. Check Ollama server logs.")
# --------------------------------------------------------------------

# ---------------------- STREAMLIT LAYOUT ------------------------------
st.set_page_config(page_title="📄 Offline RAG with MapReduce", layout="wide")
# <-- CHANGED: Updated title to reflect the new model
st.title("🤖 Offline RAG with Ollama bge-m3 & MapReduce Chain")

with st.sidebar:
    st.subheader("Upload and Process PDFs")
    pdf_docs = st.file_uploader("Upload PDF files here", type=["pdf"], accept_multiple_files=True)

    if st.session_state.vision_model is None:
        if st.button("Initialize Vision Model"):
             st.session_state.vision_model = get_vision_model()
             st.rerun()

    if st.button("Process Documents 🚀", disabled=not pdf_docs):
        if pdf_docs:
            with st.spinner("Extracting text and building vector store... ⏳"):
                full_text = get_documents_from_pdfs(pdf_docs)
                if full_text.strip():
                    chunks = get_text_chunks(full_text)
                    if chunks:
                        st.write(f"DEBUG: Generated {len(chunks)} text chunks for vector store.")
                        vs = get_vector_store(chunks)
                        if vs:
                            st.session_state.vector_store = vs
                            st.success(f"Processed {len(pdf_docs)} document(s). You may now ask questions.")
                        else:
                            st.error("Failed to build vector store. Check logs for embedding/FAISS errors.")
                    else:
                        st.warning("No text chunks could be generated from the PDFs.")
                else:
                    st.error("No text extracted from uploaded PDFs.")
        else:
            st.warning("Please upload at least one PDF.")

if st.session_state.vector_store:
    st.info("Ready to answer questions based on processed documents.")
    if not st.session_state.get("llm_initialized", False):
        st.error("⚠️ LLM (Gemma 3) is not initialized. Querying will not work.")
else:
    st.info("Upload and process PDF(s) to start querying.")

st.subheader("Ask a Question 🤔")
user_question_disabled = (
    not st.session_state.vector_store or
    not st.session_state.get("llm_initialized", False)
)
user_question = st.text_input(
    "Type your question here:",
    key="user_question_input",
    disabled=user_question_disabled
)

if user_question:
    user_input(user_question, st.session_state.vector_store)
# --------------------------------------------------------------------