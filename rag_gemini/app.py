import streamlit as st
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from pypdf import PdfReader # Still needed as a fallback
import os
from dotenv import load_dotenv

# New imports for advanced PDF processing
import fitz  # PyMuPDF
from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import clean
from PIL import Image
import io
import traceback # For detailed error logging

try:
    import markdownify # For converting HTML tables to Markdown
except ImportError:
    markdownify = None
    # This warning will appear once at the start if markdownify is not installed
    # Consider moving this specific warning inside the function where it's used,
    # or ensure users know to install it via requirements.txt

# --- Streamlit App UI ---
# THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="📄 Advanced RAG with Gemini", layout="wide")

# Load environment variables FIRST
load_dotenv()

# --- Configuration (Define API_KEY) ---
_api_key_from_secrets = None
try:
    _api_key_from_secrets = st.secrets.GOOGLE_API_KEY
except AttributeError:
    pass
except Exception as e:
    st.warning(f"An unexpected error occurred while trying to access Streamlit secrets: {e}")
    pass

API_KEY = _api_key_from_secrets or os.getenv("GOOGLE_API_KEY")

# --- Helper Functions ---

def get_vision_model():
    try:
        model = genai.GenerativeModel(
            'gemini-2.5-flash-preview-05-20',
            generation_config={"response_mime_type": "text/plain"}
        )
        return model
    except Exception as e:
        st.warning(f"Could not initialize a vision model (e.g., 'gemini-1.5-flash-latest'). Image captioning might be limited. Error: {e}")
        st.info("Ensure your API key has permissions for this model and billing is enabled for your Google Cloud project.")
        return None

def get_image_description(image_bytes, vision_model_instance):
    if not vision_model_instance:
        return "Image captioning disabled (vision model not available)."
    try:
        img = Image.open(io.BytesIO(image_bytes))
        prompt_text = """Describe this image, focusing on any data, graphs, or formulas visible.
If it's a graph, describe its type, axes, and general trend.
If it's a table, try to extract its content in a structured way.
If it's a formula, try to represent it in text or LaTeX if possible.
If it's a diagram, explain its components and relationships."""
        response = vision_model_instance.generate_content([prompt_text, img])
        return response.text
    except Exception as e:
        st.error(f"Could not get image description: {e}")
        return "Error generating image description."


def get_document_elements(pdf_docs, vision_model_instance):
    all_docs_content = []

    for pdf_file_like_object in pdf_docs:
        st.write(f"Processing Document: {pdf_file_like_object.name}...")
        pdf_file_like_object.seek(0)
        file_content = pdf_file_like_object.read()
        pdf_file_like_object.seek(0)

        current_doc_texts = []

        try:
            elements = partition_pdf(
                file=pdf_file_like_object,
                strategy="hi_res", # This strategy often relies on Tesseract
                infer_table_structure=True,
                extract_images_in_pdf=False
            )
            unstructured_content = []
            for el in elements:
                element_text_representation = ""
                if "Table" in str(type(el)):
                    table_html = getattr(el.metadata, 'text_as_html', None)
                    if table_html and markdownify:
                        try:
                            table_md = markdownify.markdownify(table_html)
                            element_text_representation = f"\n[TABLE START]\n{table_md}\n[TABLE END]\n"
                        except Exception as md_err:
                            st.warning(f"Markdownify error on table for {pdf_file_like_object.name}: {md_err}. Falling back to text.")
                            element_text_representation = f"\n[TABLE START]\n{el.text}\n[TABLE END]\n"
                    elif table_html:
                         element_text_representation = f"\n[TABLE HTML START]\n{table_html}\n[TABLE HTML END]\n"
                    else:
                        element_text_representation = f"\n[TABLE START]\n{el.text}\n[TABLE END]\n"
                else:
                    cleaned_text = clean(el.text, bullets=True, extra_whitespace=True)
                    if cleaned_text.strip():
                        element_text_representation = cleaned_text
                
                if element_text_representation:
                    unstructured_content.append(element_text_representation)
            
            if unstructured_content:
                current_doc_texts.append("\n\n".join(unstructured_content))
                st.write(f"  - Extracted text and tables using 'unstructured' for {pdf_file_like_object.name}.")
            else:
                st.write(f"  - 'unstructured' found no primary text content for {pdf_file_like_object.name}. Attempting PyPDF fallback for text.")
                pdf_file_like_object.seek(0)
                pdf_reader = PdfReader(pdf_file_like_object)
                text_pypdf = "".join(page.extract_text() or "" for page in pdf_reader.pages)
                if text_pypdf.strip():
                    current_doc_texts.append(text_pypdf)
                    st.write(f"  - Extracted text using PyPDF fallback for {pdf_file_like_object.name}.")
                else:
                    st.warning(f"  - No text extracted by PyPDF fallback for {pdf_file_like_object.name} either.")

        except Exception as e:
            # Enhanced error messaging for Tesseract issue
            error_message_lower = str(e).lower()
            if "tesseract" in error_message_lower and ("not installed" in error_message_lower or "not in your path" in error_message_lower):
                st.warning(
                    f"  - 'unstructured' PDF processing for {pdf_file_like_object.name} encountered an issue with Tesseract OCR: '{e}'. "
                    "Tesseract is needed by 'unstructured' for optimal processing of image-based or scanned PDFs. "
                    "Please ensure Tesseract OCR is correctly installed and its installation directory is in your system's PATH. "
                    "You may need to restart your terminal/IDE after installation/PATH update. "
                    "Falling back to PyPDF for text extraction."
                )
            else:
                st.warning(f"  - 'unstructured' processing failed for {pdf_file_like_object.name}: {e}. Falling back to PyPDF for text.")
            
            # Fallback logic
            pdf_file_like_object.seek(0)
            try:
                pdf_reader = PdfReader(pdf_file_like_object)
                text_pypdf = "".join(page.extract_text() or "" for page in pdf_reader.pages)
                if text_pypdf.strip():
                    current_doc_texts.append(text_pypdf)
                    st.write(f"  - Successfully extracted text using PyPDF fallback for {pdf_file_like_object.name}.")
                else:
                    st.warning(f"  - PyPDF fallback also failed to extract text from {pdf_file_like_object.name}.")
            except Exception as pypdf_e:
                st.error(f"  - PyPDF fallback also failed for {pdf_file_like_object.name}: {pypdf_e}")


        # 2. Use PyMuPDF to extract images and then get descriptions
        try:
            doc_pymupdf = fitz.open(stream=file_content, filetype="pdf")
            img_desc_counter = 1
            doc_image_descriptions = []
            for page_num in range(len(doc_pymupdf)):
                for img_index, img_info in enumerate(doc_pymupdf.get_page_images(page_num)):
                    xref = img_info[0]
                    base_image = doc_pymupdf.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    st.write(f"    - Describing image {img_desc_counter} on page {page_num+1} of {pdf_file_like_object.name}...")
                    description = get_image_description(image_bytes, vision_model_instance)
                    doc_image_descriptions.append(f"\n[IMAGE {img_desc_counter} ON PAGE {page_num+1} DESCRIPTION START]\n{description}\n[IMAGE {img_desc_counter} DESCRIPTION END]\n")
                    img_desc_counter += 1
            
            if doc_image_descriptions:
                current_doc_texts.append("\n".join(doc_image_descriptions))
                st.write(f"  - Extracted and described {img_desc_counter-1} image(s) from {pdf_file_like_object.name}.")
            doc_pymupdf.close()
        except Exception as e:
            st.warning(f"  - Image extraction/description failed for {pdf_file_like_object.name}: {e}")
            st.warning(f"  - Traceback: {traceback.format_exc()}") # Added for more detail on image errors

        if current_doc_texts:
            all_docs_content.append("\n\n".join(current_doc_texts))

    return "\n\n--- NEW DOCUMENT BOUNDARY ---\n\n".join(all_docs_content)


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, current_api_key):
    if not text_chunks:
        st.warning("No text chunks found to process.")
        return None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=current_api_key)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        st.error("This might be due to an invalid API key or network issues with Google API.")
        st.error("Ensure your `models/text-embedding-004` model is enabled for your API key in Google AI Studio.")
        return None

def get_conversational_chain(current_api_key):
    prompt_template_str = """
    You are a helpful AI assistant. Answer the question as detailed as possible from the provided context.
    The context may contain regular text, descriptions of images (denoted by [IMAGE ... DESCRIPTION START]...[IMAGE ... DESCRIPTION END]),
    and data from tables (denoted by [TABLE START]...[TABLE END] or [TABLE HTML START]...[TABLE HTML END], often in Markdown or HTML format).
    Use all available information to answer the question.
    If the answer is not in the provided context, just say, "The answer is not available in the provided documents."
    Do not provide a wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    # Consider 'gemini-pro' or 'gemini-1.5-pro-latest' if 'gemini-2.5-flash-preview-05-20' is problematic
    # For now, keeping the user's specified model.
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", # Changed to a commonly available model, adjust if needed
                                 temperature=0.3,
                                 google_api_key=current_api_key)
    prompt = PromptTemplate(template=prompt_template_str, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, vector_store, current_api_key):
    if not vector_store:
        st.warning("Please upload and process documents first.")
        return

    try:
        st.write("DEBUG: Attempting similarity search...")
        # Consider reducing k for testing if context size is an issue
        docs = vector_store.similarity_search(user_question, k=3) # Reduced k to 3 for testing
        st.write(f"DEBUG: Similarity search completed. Found {len(docs)} documents.")

        if not docs:
            st.info("No relevant documents found for your question in the provided PDF.")
            return

        # st.write("DEBUG: Retrieved document snippets (first 100 chars each):")
        # for i, doc_ret in enumerate(docs):
        #     st.caption(f"DEBUG: Doc {i+1}: {doc_ret.page_content[:100]}...")

        st.write("DEBUG: Getting conversational chain...")
        chain = get_conversational_chain(current_api_key)
        st.write("DEBUG: Conversational chain obtained.")

        # Prepare input for the chain
        input_data = {"input_documents": docs, "question": user_question}
        
        context_char_count = sum(len(doc.page_content) for doc in docs)
        st.write(f"DEBUG: Approximate context character count for LLM: {context_char_count}")
        if context_char_count > 250000: # Gemini Pro has a large context, but very large inputs can still be slow/problematic
            st.warning(f"DEBUG: Context character count ({context_char_count}) is very large. This might take time or hit limits.")

        st.write("DEBUG: Making API call to Gemini via chain...")
        response = chain(input_data, return_only_outputs=True)
        st.write("DEBUG: API call successful. Displaying response.")

        st.subheader("Answer:")
        st.write(response["output_text"])

        with st.expander("Show Retrieved Context Chunks"):
            for i, doc_ret in enumerate(docs):
                st.markdown(f"**Chunk {i+1}:**")
                # Display more content for debugging context issues
                st.caption(doc_ret.page_content[:1000] + "..." if len(doc_ret.page_content) > 1000 else doc_ret.page_content)
    except Exception as e:
        st.error(f"Error during question processing: {e}")
        st.error(f"Traceback: {traceback.format_exc()}") # This is crucial
        st.error("This could be an issue with the Gemini API, network, the document content, or environment configuration (like OMP).")

    # --- Apply CSS Styling ---
    st.markdown("""
    <style>
        /* Base app styling */
        .stApp {
            background-color: #f0f2f6; /* Light gray background for the app */
        }

        /* Text input field styling */
        .stTextInput>div>div>input {
            background-color: #ffffff;
            color: #333333 !important; /* Dark text for text input fields */
        }

        /* Button styling */
        .stButton>button {
            background-color: #4CAF50;
            color: white !important;   /* White text for buttons */
            border-radius: 5px;
            padding: 10px 20px;
        }

        /* --- MAIN CONTENT AREA TEXT --- */
        /* General text elements - make these apply broadly first */
        /* Targets elements NOT inside the sidebar */
        body:not(.sidebar) h1,
        body:not(.sidebar) h2,
        body:not(.sidebar) h3,
        body:not(.sidebar) h4,
        body:not(.sidebar) h5,
        body:not(.sidebar) h6,
        body:not(.sidebar) p,
        body:not(.sidebar) li,
        body:not(.sidebar) .stMarkdown,
        body:not(.sidebar) [data-testid="stText"],
        body:not(.sidebar) [data-testid="stHeader"],
        body:not(.sidebar) [data-testid="stSubheader"],
        body:not(.sidebar) [data-testid="stAlert"] div[role="alert"], /* Notifications in main area */
        body:not(.sidebar) [data-testid^="stNotificationContent"] { /* Covers success, error, info, warning notifications in main area */
            color: #333333 !important; /* Dark text for main content */
        }

        /* If the above is too broad and affects other things, revert to simple selectors for main content */
        /* These should target the main area by default if not overridden by sidebar */
        /*
        h1, h2, h3, h4, h5, h6, p, li,
        .stMarkdown,
        [data-testid="stText"],
        [data-testid="stHeader"],
        [data-testid="stSubheader"],
        [data-testid="stAlert"] div[role="alert"],
        [data-testid^="stNotificationContent"] {
            color: #333333 !important;
        }
        */


        /* --- SIDEBAR SPECIFIC STYLING --- */
        /* Text elements within the sidebar */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3, /* Catches st.subheader in sidebar */
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] h5,
        [data-testid="stSidebar"] h6,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] li,
        [data-testid="stSidebar"] label, /* Catches st.file_uploader label, st.text_input label etc. */
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] [data-testid="stText"],
        [data-testid="stSidebar"] [data-testid="stHeader"],
        [data-testid="stSidebar"] [data-testid="stSubheader"] {
            color: #FAFAFA !important; /* Light text color for common elements in sidebar */
        }

        /* Specifically target text within Streamlit's alert/notification components in the sidebar */
        [data-testid="stSidebar"] [data-testid="stAlert"] div[role="alert"],
        [data-testid="stSidebar"] [data-testid^="stNotificationContent"],
        [data-testid="stSidebar"] [data-testid^="stNotificationContent"] p {
            color: #FAFAFA !important; /* Light text for notifications in sidebar */
        }

        /* Ensure sidebar buttons maintain their specific styling if general sidebar rules interfere */
        [data-testid="stSidebar"] .stButton>button {
            background-color: #4CAF50; /* Keep original button background */
            color: white !important;   /* Keep original button text color */
        }

        /* Explicitly style the caption in the main area if it's still light */
        /* Assuming st.caption renders as a specific element or has a unique testid */
        [data-testid="stCaptionContainer"] {
            color: #555555 !important; /* A slightly dimmer dark color for captions */
        }
        /* Or if it's just a paragraph tag inside a caption container */
        [data-testid="stCaptionContainer"] p {
            color: #555555 !important;
        }

    </style>
    """, unsafe_allow_html=True)

# --- Page Header ---
st.header("🤖 Advanced RAG with Gemini ⚡️")
st.caption("Upload PDFs (text, tables, formulas, graphs), ask questions, and get answers.")
if markdownify is None:
    st.warning("`markdownify` library not found. HTML tables will be processed as raw HTML/text. Install with `pip install markdownify` for better table handling in context.")


# --- API Key Management & SDK Configuration ---
if not API_KEY:
    st.warning("Google API Key not found (checked Streamlit secrets and .env). Please provide it below.")
    api_key_input_user = st.text_input("Enter your Google API Key:", type="password", key="api_key_input_user_prompt")
    if api_key_input_user:
        API_KEY = api_key_input_user
        st.session_state.api_key_provided_manually = True
        st.success("API Key provided. Proceeding with configuration...")
    else:
        st.info("Please provide your Google API Key to use the app.")
        st.stop()

if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        if not st.session_state.get('sdk_configured_message_shown', False) or st.session_state.get('api_key_provided_manually', False):
            st.sidebar.success("Google AI SDK configured successfully!")
            st.session_state.sdk_configured_message_shown = True
            if 'api_key_provided_manually' in st.session_state:
                 del st.session_state.api_key_provided_manually

        if "vision_model" not in st.session_state or st.session_state.vision_model is None:
            st.session_state.vision_model = get_vision_model()
            if st.session_state.vision_model is None:
                 st.sidebar.error("Failed to initialize the vision model. Image processing will be impacted. Check API key permissions for vision models.")
    except Exception as e:
        st.error(f"Failed to configure Google AI SDK or initialize vision model: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        st.info("Ensure your API key is correct, has necessary permissions (e.g., for 'gemini-1.5-flash-latest'), and billing is enabled for your Google Cloud project.")
        st.stop()
else:
    st.error("Critical: API Key is missing. The application cannot proceed.")
    st.stop()


# --- Initialize Session State Variables ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# --- Sidebar for Document Upload and Processing ---
with st.sidebar:
    st.subheader("Your Documents 📚")
    pdf_docs = st.file_uploader("Upload PDF Files here", accept_multiple_files=True, type="pdf")

    if st.button("Process Documents 🚀", disabled=not pdf_docs or not API_KEY):
        if pdf_docs:
            with st.spinner("Processing documents (this can take a while for complex PDFs with images)... ⏳"):
                try:
                    vision_model_to_use = st.session_state.get("vision_model")
                    if not vision_model_to_use:
                         st.error("Vision model not available. Image description processing will be skipped.")

                    raw_elements_text = get_document_elements(pdf_docs, vision_model_to_use) # Pass vision model
                    if not raw_elements_text or not raw_elements_text.strip():
                        st.error("No text or elements could be extracted. Please check the PDF(s) or processing logs.")
                    else:
                        text_chunks = get_text_chunks(raw_elements_text)
                        if text_chunks:
                            st.session_state.vector_store = get_vector_store(text_chunks, API_KEY) # Pass API_KEY
                            if st.session_state.vector_store:
                                st.session_state.processed_files = [doc.name for doc in pdf_docs]
                                st.success(f"Documents processed: {', '.join(st.session_state.processed_files)}")
                            else:
                                st.error("Failed to create vector store after processing. Check error messages.")
                        else:
                            st.warning("No text chunks were generated. The processed content might be empty or too short.")
                except Exception as e:
                    st.error(f"An error occurred during document processing: {e}")
                    st.error(f"Traceback: {traceback.format_exc()}")
                    st.session_state.vector_store = None
        else:
            st.warning("Please upload at least one PDF file.")

# --- Main Area for Q&A ---
if st.session_state.processed_files:
    st.info(f"Ready to answer questions based on: {', '.join(st.session_state.processed_files)}")
else:
    st.info("Upload and process documents to start asking questions.")

st.subheader("Ask a Question 🤔")
user_question = st.text_input("Type your question about the documents:", key="user_question_input",
                              disabled=not st.session_state.vector_store)

if user_question and st.session_state.vector_store and API_KEY:
    with st.spinner("Thinking... 🧠"):
        user_input(user_question, st.session_state.vector_store, API_KEY) # Pass API_KEY
elif user_question and (not st.session_state.vector_store or not API_KEY):
    st.warning("Please ensure documents are processed and API key is configured before asking a question.")

st.markdown("---")
st.caption("Powered by Google Gemini & LangChain. Enhanced PDF Parsing by Unstructured & PyMuPDF.")