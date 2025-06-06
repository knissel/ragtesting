import streamlit as st
import io                                 # Required for BytesIO usage :contentReference[oaicite:7]{index=7}
from io import BytesIO                   # Alias for convenience :contentReference[oaicite:8]{index=8}
import os
import traceback
import base64                             # Required to encode images in Base64 :contentReference[oaicite:9]{index=9}
import logging
import re
import pandas as pd

from dotenv import load_dotenv
from PIL import Image

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama  # Uses local Gemma3 models via Ollama :contentReference[oaicite:10]{index=10}
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

from pypdf import PdfReader
import fitz
from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import clean

# --- Streamlit & Environment Setup ---
st.set_page_config(page_title="📄 RAG with Local Gemma", layout="wide")
load_dotenv()  # Not needed for vision key, but kept for other secrets

# Configure logging to send debug messages to the console instead of the UI :contentReference[oaicite:11]{index=11}
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def pil_image_to_base64(pil_img: Image.Image) -> str:
    """
    Convert a PIL Image to a Base64-encoded JPEG string (no data URI prefix).
    """
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")  # You can change format if needed :contentReference[oaicite:12]{index=12}
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")   # Encode raw bytes to Base64 :contentReference[oaicite:13]{index=13}

def clean_response_text(raw_text: str) -> str:
    """
    Join letters split by newlines and collapse multiple blank lines into single paragraph breaks.
    """
    # Join characters that were split by newlines (e.g., "T\nh\ni\ns" → "This")
    cleaned = re.sub(r"([A-Za-z])\n(?=[A-Za-z])", r"\1", raw_text) 
    # Collapse more than one blank line into exactly two newlines
    cleaned = re.sub(r"\n{2,}", "\n\n", cleaned)
    return cleaned.strip()

def normalize_currency_spans(text: str) -> str:
    """
    Fix spacing inside numbers and commas so that "5 , 7 4 7" becomes "5,747" or "5 7 4 7" → "5747".
    """
    # Collapse misplaced commas: "5 , 7" → "5,7"
    text = re.sub(r"(\d)\s*,\s*(\d)", r"\1,\2", text)
    # Remove spaces between digits: "5 7 4 7" → "5747"
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    return text

def display_cleaned_answer(raw_answer: str):
    """
    Clean the raw answer from the LLM and render it inside a <pre> block for better readability.
    """
    step1 = clean_response_text(raw_answer)
    final_answer = normalize_currency_spans(step1)
    st.markdown("### Answer")  # Subheading for emphasis
    # Use a <pre> block to preserve spacing and line breaks, with monospace font
    st.markdown(f"<pre style='font-size: 1rem'>{final_answer}</pre>", unsafe_allow_html=True) 

def display_context_chunks(docs):
    """
    Show retrieved context chunks in an expander. Convert any HTML tables to DataFrames.
    """
    with st.expander("Show Retrieved Context Chunks"):
        for i, doc_ret in enumerate(docs):
            snippet = doc_ret.page_content
            # If it looks like HTML, try to parse tables
            if "<table" in snippet.lower():
                try:
                    df_list = pd.read_html(snippet)  # Parse HTML tables into DataFrames :contentReference[oaicite:16]{index=16}
                    st.markdown(f"**Chunk {i+1} (Table):**")
                    for df in df_list:
                        st.dataframe(df)
                except ValueError:
                    # Not a clean HTML table, show a truncated preview
                    st.markdown(f"**Chunk {i+1}:**")
                    st.caption(snippet[:500] + "..." if len(snippet) > 500 else snippet)
            else:
                st.markdown(f"**Chunk {i+1}:**")
                st.caption(snippet[:500] + "..." if len(snippet) > 500 else snippet)

def get_vision_model():
    """
    Initialize a local Gemma 3 multimodal model via Ollama.  Requires 'gemma3:27b' (or other Gemma 3 variant) be pulled locally.
    """
    try:
        model = ChatOllama(model="gemma3:27b", temperature=0.3)  # Use local Gemma 3 27B via Ollama :contentReference[oaicite:17]{index=17}
        return model
    except Exception as e:
        st.warning(f"Could not initialize local Gemma 3 vision model: {e}")
        return None

def get_image_description(image_bytes: bytes, vision_model_instance):
    """
    Describe an image using Gemma 3 via Ollama. Wrap the Base64 string in a data URI and use 'type': 'image_url'.
    """
    if not vision_model_instance:
        return "Image captioning disabled (no local vision model available)."
    try:
        # Load the image into PIL and convert to RGB
        pil_img = Image.open(BytesIO(image_bytes)).convert("RGB") 
        raw_b64 = pil_image_to_base64(pil_img)              # Convert to Base64 string :contentReference[oaicite:19]{index=19}
        data_uri = f"data:image/jpeg;base64,{raw_b64}"      # Prefix with data URI scheme :contentReference[oaicite:20]{index=20}

        # Construct an 'image_url' block (required by ChatOllama) instead of 'image'/'data'
        image_block = {
            "type": "image_url",    # Must be exactly 'image_url' for ChatOllama :contentReference[oaicite:21]{index=21}
            "image_url": data_uri   # Pass the data URI string here :contentReference[oaicite:22]{index=22}
        }
        # Create a text block with instructions for describing the image
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

        # Combine text and image blocks into a single HumanMessage
        human_message = HumanMessage(content=[text_block, image_block]) 

        # Use predict_messages to get a BaseMessage instead of nested LLMResult 
        response_msg = vision_model_instance.predict_messages([human_message])
        return response_msg.content  # The textual description 

    except Exception as e:
        st.error(f"Error generating image description: {e}")
        return "Error generating image description."

def get_document_elements(pdf_docs, vision_model_instance):
    """
    Extract text, tables, and image descriptions from a list of uploaded PDFs.
    Returns a single concatenated string with all document content.
    """
    all_docs_content = []
    for pdf_file in pdf_docs:
        st.write(f"Processing Document: {pdf_file.name}...")
        pdf_file.seek(0)
        raw_bytes = pdf_file.read()
        pdf_file.seek(0)

        current_doc_texts = []
        try:
            elements = partition_pdf(
                file=pdf_file,
                strategy="hi_res",
                infer_table_structure=True,
                extract_images_in_pdf=False
            )
            unstructured_content = []
            for el in elements:
                element_text = ""
                if "Table" in str(type(el)):
                    table_html = getattr(el.metadata, "text_as_html", None)
                    if table_html:
                        element_text = f"\n[TABLE HTML START]\n{table_html}\n[TABLE HTML END]\n"
                    else:
                        element_text = f"\n[TABLE START]\n{el.text}\n[TABLE END]\n"
                else:
                    cleaned = clean(el.text, bullets=True, extra_whitespace=True)
                    if cleaned.strip():
                        element_text = cleaned
                if element_text:
                    unstructured_content.append(element_text)

            if unstructured_content:
                current_doc_texts.append("\n\n".join(unstructured_content))
                st.write(f"  - Extracted text/tables for {pdf_file.name}.")
            else:
                st.write(f"  - No primary text from 'unstructured'. Falling back to PyPDF.")
                pdf_file.seek(0)
                reader = PdfReader(pdf_file)
                text_fallback = "".join(page.extract_text() or "" for page in reader.pages)
                if text_fallback.strip():
                    current_doc_texts.append(text_fallback)
                    st.write(f"  - PyPDF fallback text extracted for {pdf_file.name}.")
                else:
                    st.warning(f"  - No text from PyPDF fallback for {pdf_file.name}.")
        except Exception as e:
            msg = str(e).lower()
            if "tesseract" in msg:
                st.warning(
                    f"  - 'unstructured' needs Tesseract OCR for {pdf_file.name}. "
                    "Install Tesseract and add to PATH. Falling back to PyPDF."
                )
            else:
                st.warning(f"  - 'unstructured' failed for {pdf_file.name}: {e}. Falling back to PyPDF.")
            pdf_file.seek(0)
            try:
                reader = PdfReader(pdf_file)
                text_fallback = "".join(page.extract_text() or "" for page in reader.pages)
                if text_fallback.strip():
                    current_doc_texts.append(text_fallback)
                    st.write(f"  - PyPDF fallback text extracted for {pdf_file.name}.")
                else:
                    st.warning(f"  - No text from PyPDF fallback either for {pdf_file.name}.")
            except Exception as pypdf_e:
                st.error(f"  - PyPDF fallback also failed for {pdf_file.name}: {pypdf_e}")

        # Process images for vision if available
        try:
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
            doc_images_desc = []
            if vision_model_instance:
                img_counter = 1
                for page_idx in range(len(doc)):
                    for img_info in doc.get_page_images(page_idx):
                        xref = img_info[0]
                        base_img = doc.extract_image(xref)
                        img_bytes = base_img["image"]

                        st.write(f"    - Describing image {img_counter} on page {page_idx+1} of {pdf_file.name}...")
                        desc = get_image_description(img_bytes, vision_model_instance)
                        doc_images_desc.append(
                            f"\n[IMAGE {img_counter} ON PAGE {page_idx+1} DESCRIPTION START]\n{desc}\n[IMAGE {img_counter} DESCRIPTION END]\n"
                        )
                        img_counter += 1

                if doc_images_desc:
                    current_doc_texts.append("\n".join(doc_images_desc))
                    st.write(f"  - Described {img_counter-1} image(s) for {pdf_file.name}.")
            else:
                st.write(f"  - Skipping image description (Gemma 3 not available) for {pdf_file.name}.")
            doc.close()
        except Exception as e:
            st.warning(f"  - Image extraction/description failed for {pdf_file.name}: {e}")
            st.warning(f"  - Traceback: {traceback.format_exc()}")

        if current_doc_texts:
            all_docs_content.append("\n\n".join(current_doc_texts))

    return "\n\n--- NEW DOCUMENT BOUNDARY ---\n\n".join(all_docs_content)

def get_text_chunks(text):
    """
    Split long text into manageable chunks (≈2000 characters each with 200-character overlap).
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """
    Create a FAISS vector store from text chunks using OllamaEmbeddings (nomic-embed-text).
    """
    if not text_chunks:
        st.warning("No text chunks found to process.")
        return None
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")  # Requires Ollama server with nomic-embed-text :contentReference[oaicite:26]{index=26}
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating FAISS store: {e}")
        return None

def get_conversational_chain():
    """
    Build a QA chain using Gemma 3 via ChatOllama.
    """
    prompt_template_str = """
    You are a helpful AI assistant. Answer the question in detail using the provided context.
    Context may contain text, image descriptions ([IMAGE ... DESCRIPTION]), and tables ([TABLE START]...).
    Use all available information. If not in context, say: "The answer is not available in the provided documents."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    llm = ChatOllama(model="gemma3:27b", temperature=0.3)  # Use local Gemma 3 :contentReference[oaicite:27]{index=27}
    prompt = PromptTemplate(template=prompt_template_str, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, vector_store):
    """
    Perform similarity search, build the QA chain, query Gemma 3, and display both the answer and context.
    """
    if not vector_store:
        st.warning("Please upload and process documents first.")
        return
    try:
        logger.debug("Performing similarity search...")  # Sent to console :contentReference[oaicite:28]{index=28}
        docs = vector_store.similarity_search(user_question, k=3)
        logger.debug(f"Found {len(docs)} relevant chunks.")  # Console only :contentReference[oaicite:29]{index=29}

        if not docs:
            st.info("No relevant documents found for your question.")
            return

        logger.debug("Building conversational chain with Gemma 3...")
        chain = get_conversational_chain()
        logger.debug("Chain ready.")

        input_data = {"input_documents": docs, "question": user_question}
        char_count = sum(len(doc.page_content) for doc in docs)
        logger.debug(f"Context char count ≈ {char_count}")

        if char_count > 100000:  # Soft warning for large context 
            st.warning(f"DEBUG: Context sizable ({char_count} chars). Response time may be slower.")

        logger.debug("Calling local Gemma 3 for answer...")
        response = chain(input_data, return_only_outputs=True)  # Returns dict with "output_text"
        logger.debug("Answer received.")

        raw_answer = response["output_text"]
        display_cleaned_answer(raw_answer)  # Use our cleaning & rendering logic

        # Show retrieved chunks (with table parsing)
        display_context_chunks(docs)

    except Exception as e:
        st.error(f"Error during question processing: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        st.error("Ensure Ollama and Gemma 3 are running and the model is available.")

# --- Streamlit Layout ---
st.header("🤖 RAG with Local Gemma ⚡️")
st.caption("Upload PDFs, ask questions, and get answers using a local LLM. Image analysis uses local Gemma 3 via Ollama.")

# Initialize or load the vision model
if "vision_model" not in st.session_state:
    st.session_state.vision_model = get_vision_model()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

with st.sidebar:
    st.subheader("Your Documents 📚")
    pdf_docs = st.file_uploader("Upload PDF Files here", accept_multiple_files=True, type="pdf")

    if st.button("Process Documents 🚀", disabled=not pdf_docs):
        if pdf_docs:
            with st.spinner("Processing documents... ⏳"):
                try:
                    raw_elements = get_document_elements(pdf_docs, st.session_state.vision_model)
                    if not raw_elements.strip():
                        st.error("No text or elements extracted. Check PDF(s).")
                    else:
                        chunks = get_text_chunks(raw_elements)
                        if chunks:
                            vs = get_vector_store(chunks)
                            if vs:
                                st.session_state.vector_store = vs
                                st.session_state.processed_files = [doc.name for doc in pdf_docs]
                                st.success(f"Processed: {', '.join(st.session_state.processed_files)}")
                            else:
                                st.error("Failed to build vector store. Check Ollama embeddings.")
                        else:
                            st.warning("No text chunks generated.")
                except Exception as e:
                    st.error(f"Error processing documents: {e}")
                    st.error(f"Traceback: {traceback.format_exc()}")
                    st.session_state.vector_store = None
        else:
            st.warning("Please upload at least one PDF.")

if st.session_state.processed_files:
    st.info(f"Ready to answer questions based on: {', '.join(st.session_state.processed_files)}")
else:
    st.info("Upload and process documents to start asking questions.")

st.subheader("Ask a Question 🤔")
user_question = st.text_input(
    "Type your question about the documents:", key="user_question_input",
    disabled=not st.session_state.vector_store
)

if user_question and st.session_state.vector_store:
    with st.spinner("Asking local Gemma... 🧠"):
        user_input(user_question, st.session_state.vector_store)
elif user_question and not st.session_state.vector_store:
    st.warning("Please process documents before asking a question.")

st.markdown("---")
st.caption("Powered by Local Gemma 3 (via Ollama) & LangChain. Enhanced PDF parsing by Unstructured & PyMuPDF.")
