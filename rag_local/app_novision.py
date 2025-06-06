# app.py

import os
import streamlit as st
import traceback
from dotenv import load_dotenv
from pypdf import PdfReader
import fitz  # PyMuPDF
from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import clean

# ---------------------- IMPORT FOR EMBEDDINGS ----------------------
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
# load_qa_chain is not used in this version due to manual construction
from langchain_community.chat_models import ChatOllama

# For manual chain construction
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
# --------------------------------------------------------------------

# -------------- GLOBAL VARIABLES & SESSION STATE SETUP -------------
load_dotenv()

try:
    llm = ChatOllama(model="gemma3:27b", temperature=0.0)
    st.session_state.llm_initialized = True  # Keep track if LLM is good
except Exception as e:
    st.error(f"Could not initialize local Gemma 3 via Ollama: {e}")
    llm = None
    st.session_state.llm_initialized = False

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
LOCAL_EMBEDDING_PATH = r"C:\Users\kniss\rag\models\all-MiniLM-L6-v2"  # Make sure this path is correct

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
# --------------------------------------------------------------------

# ---------------------- PROMPTS FOR MAP-REDUCE CHAIN ----------------------
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

# NOTE: Every single {…} inside the “For example:” section below has been doubled to {{…}}
#       so that PromptTemplate does not treat them as missing variables.
COMBINE_PROMPT = PromptTemplate(
    template="""
You have been given multiple INTERMEDIATE ANSWERS from different excerpts.
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

# ---------------------- EMBEDDING LOADING & CACHING ----------------------
@st.cache_resource
def load_local_embeddings():
    """
    Load the SentenceTransformer model once and wrap it in HuggingFaceEmbeddings.
    """
    # st.write("DEBUG: Attempting to load SentenceTransformer model...")
    try:
        _ = SentenceTransformer(LOCAL_EMBEDDING_PATH, local_files_only=True)
        # st.write("DEBUG: SentenceTransformer model loaded for check.")
        embeddings = HuggingFaceEmbeddings(
            model_name=LOCAL_EMBEDDING_PATH,
            model_kwargs={"local_files_only": True}
        )
        # st.write("DEBUG: HuggingFaceEmbeddings wrapped successfully.")
        return embeddings
    except Exception as e:
        st.error(f"Error loading local embeddings from {LOCAL_EMBEDDING_PATH}: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None


@st.cache_data
def get_vector_store(text_chunks):
    """
    Build a FAISS index from text_chunks using pre-loaded local HuggingFaceEmbeddings.
    """
    if not text_chunks:
        st.warning("DEBUG: No text chunks provided to get_vector_store.")
        return None
    try:
        embeddings = load_local_embeddings()
        if embeddings is None:
            st.error("DEBUG: Embeddings are None, cannot create vector store.")
            return None
        # st.write(f"DEBUG: Creating FAISS vector store from {len(text_chunks)} chunks.")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        # st.write("DEBUG: FAISS vector store created.")
        return vector_store
    except Exception as e:
        st.error(f"Error creating FAISS store: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None
# ----------------------------------------------------------------------

# ---------------------- PDF TEXT EXTRACTION ---------------------------
def get_text_chunks(full_text: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return text_splitter.split_text(full_text)

def get_documents_from_pdfs(pdf_files):
    all_text = []
    for pdf_file in pdf_files:
        st.write(f"DEBUG: Processing PDF: {pdf_file.name}")
        try:
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
                        texts.append(html)
                    else:
                        texts.append(el.text)
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
    return "\n\n--- DOCUMENT BOUNDARY ---\n\n".join(all_text)
# ---------------------------------------------------------------------

# ---------------------- USER QUERY & QA CHAIN --------------------------
def user_input(user_question, vector_store):
    """
    Perform similarity search, build the QA chain, query, and display results.
    """
    # st.write("DEBUG: Entered user_input function.")  # DEBUG

    if not vector_store:
        st.warning("Please upload and process documents first (vector_store is None).")
        return

    if llm is None or not st.session_state.get("llm_initialized", False):
        st.error("LLM (Gemma 3) is not initialized. Cannot proceed with query.")
        return

    try:
        # st.write("DEBUG: vector_store and LLM exist, proceeding with similarity search.")  # DEBUG
        docs = vector_store.similarity_search(user_question, k=3)
        # st.write(f"DEBUG: Found {len(docs)} documents from similarity search.")  # DEBUG
        if docs:
            # for i, d in enumerate(docs):
                # st.write(f"DEBUG: Doc {i} content (first 100 chars): {d.page_content[:100]}")  # DEBUG

            if not docs:
                st.info("No relevant documents found for your question.")
                return

        # st.write("DEBUG: Proceeding to build MapReduce chain.")  # DEBUG
        # ---- Manual Chain Construction ----
        map_llm_chain = LLMChain(llm=llm, prompt=MAP_PROMPT)
        combine_llm_chain = LLMChain(llm=llm, prompt=COMBINE_PROMPT)
        stuff_combine_documents_chain = StuffDocumentsChain(
            llm_chain=combine_llm_chain,
            document_variable_name="summaries",  # Matches {summaries} in COMBINE_PROMPT
        )
        chain = MapReduceDocumentsChain(
            llm_chain=map_llm_chain,  # For the map step
            reduce_documents_chain=stuff_combine_documents_chain,  # For the reduce step
            document_variable_name="context",  # Matches {context} in MAP_PROMPT
            input_key="input_documents",
            output_key="output_text",
        )
        # st.write(f"DEBUG: MapReduceDocumentsChain created.")  # DEBUG

        input_data = {"input_documents": docs, "question": user_question}
        # st.write(f"DEBUG: Input data for chain: question='{input_data['question']}', num_docs={len(input_data['input_documents'])}")  # DEBUG

        raw_answer = None
        with st.spinner("Querying the documents... 🤔 Please wait, this can take a moment."):
            response = chain(input_data, return_only_outputs=True)

        # st.write(f"DEBUG: Raw response from chain: {response}")  # DEBUG

        if response and isinstance(response, dict) and "output_text" in response:
            raw_answer = response["output_text"]
        elif response and isinstance(response, str):  # Fallback if chain directly returns string
            raw_answer = response
        else:
            st.warning("The LLM did not return an answer in the expected dictionary format or the 'output_text' key was missing.")
            # st.write("DEBUG: Full response object from chain:", response)

        if raw_answer and raw_answer.strip() and raw_answer.lower() not in [
            "information not found in this chunk.",
            "the answer is not available in the provided documents."
        ]:
            st.subheader("Answer:")
            # Use st.markdown to render text which might include LaTeX
            st.markdown(raw_answer, unsafe_allow_html=True)
        elif raw_answer and raw_answer.lower() in [
            "information not found in this chunk.",
            "the answer is not available in the provided documents."
        ]:
            st.info(raw_answer)
        else:
            st.info("No specific answer was generated by the LLM, or the answer was empty.")
            if not raw_answer:
                # st.write("DEBUG: raw_answer variable is None or empty after attempting to extract.")

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
st.title("🤖 Offline RAG with Local all-MiniLM-L6-v2 & MapReduce Chain")

with st.sidebar:
    st.subheader("Upload and Process PDFs")
    pdf_docs = st.file_uploader("Upload PDF files here", type=["pdf"], accept_multiple_files=True)

    if st.button("Process Documents 🚀", disabled=not pdf_docs):
        if pdf_docs:
            with st.spinner("Extracting text and building vector store... ⏳"):
                full_text = get_documents_from_pdfs(pdf_docs)
                if full_text.strip():
                    chunks = get_text_chunks(full_text)
                    if chunks:
                        st.write(f"DEBUG: Generated {len(chunks)} text chunks for vector store.")  # DEBUG
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
    # st.write("DEBUG: User question submitted:", user_question)  # DEBUG
    user_input(user_question, st.session_state.vector_store)
# -------------------------------------------------------------------- Ask a Question 🤔
