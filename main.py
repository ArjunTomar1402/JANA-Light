import streamlit as st
import torch
import logging
from modules.ui import render_info_section, render_sidebar, render_batch_processor, display_results
from modules.models import load_models, set_models, get_models
from modules.utils import extract_text_from_file, split_sentences
from modules.processing import process_text_batch
from modules.config import LOG_FILE

# Initialize logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("jana")
logger.info("Starting JANA app")

# Page config
st.set_page_config(
    page_title="JANA Platform - Phase 1",
    page_icon="🗾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load external CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("assets/styles.css")  # Invoke CSS from separate file

# App title
st.markdown('<h1 class="main-header"> J A N A </h1>', unsafe_allow_html=True)
st.markdown('<h2 class="sub-header">Japanese Auto-dialect Normalization & Adaptation - Phase 1</h2>', unsafe_allow_html=True)

def main():
    # Initialize session state defaults
    if 'generate_furigana' not in st.session_state:
        st.session_state.generate_furigana = False
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    if 'translator_name' not in st.session_state:
        st.session_state.translator_name = 'facebook/nllb-200-distilled-600M'

    # Render UI
    render_info_section()
    model_option, custom_model_name, _, use_gpu = render_sidebar()
    st.session_state.model_option = model_option
    
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    logger.info(f"Using device: {device}")

    # Load models
    models = load_models(model_option, device_name=str(device), custom_model_name=custom_model_name)
    set_models(
        models.get('lid_model'),
        models.get('translator_model'),
        models.get('translator_tokenizer'),
        models.get('sudachi_tokenizer_obj'),
        models.get('kakasi')
    )
    
    translator_name = models.get('translator_name') or 'facebook/nllb-200-distilled-600M'
    st.session_state.translator_name = translator_name

    lid_model, translator_model, _, sudachi_tokenizer_obj, _ = get_models()
    if lid_model is None or translator_model is None or sudachi_tokenizer_obj is None:
        st.error("Failed to load required models. Please check the error messages above and logs.")
        st.stop()

    # Single file/text processing
    st.header("Single Document Input")
    
    # Determine if mutual exclusivity applies
    disable_text = 'single_uploader' in st.session_state and st.session_state.single_uploader is not None
    disable_file = 'manual_text' in st.session_state and st.session_state.manual_text.strip() != ""

    uploaded_file = st.file_uploader(
        "Upload document",
        type=["txt", "pdf", "docx"],
        key="single_uploader",
        disabled=disable_file
    )
    if disable_file:
        st.info("File upload disabled while manual text is entered. Clear the text to upload a file.")

    manual_text = st.text_area(
        "Or enter text manually:",
        height=150,
        key="manual_text",
        disabled=disable_text
    )
    if disable_text:
        st.info("Text input disabled while a file is uploaded. Remove the file to enter text.")

    # Decide which input to process
    input_text = None
    if uploaded_file:
        input_text = extract_text_from_file(uploaded_file)
        if input_text:
            st.success("File uploaded successfully!")
            with st.expander("View extracted text"):
                st.markdown(f'<div class="scroll-container">{input_text[:1000]}{"..." if len(input_text) > 1000 else ""}</div>', unsafe_allow_html=True)
    elif manual_text:
        input_text = manual_text

    # Process single input
    if input_text and st.button("Translate to Japanese", type="primary"):
        sentences = split_sentences(input_text)
        results = process_text_batch(sentences, batch_size=1)
        if results:
            st.markdown('<div class="scroll-container">', unsafe_allow_html=True)
            display_results(results, device)
            st.markdown('</div>', unsafe_allow_html=True)

    # Batch processing section
    render_batch_processor(extract_text_from_file, split_sentences, process_text_batch)

    # Footer with GitHub hyperlink
    st.markdown(
        '<div class="footer"><a href="https://github.com/ArjunTomar1402" target="_blank" style="color:#7f7f7f; text-decoration:none;">JANA</a> - Phase 1 Implementation<br>Developed with cultural and linguistic accuracy</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
