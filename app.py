import streamlit as st
from src.pdf_processor import extract_text_from_pdf
from src.text_processor import clean_text, create_chunks
from transformers import pipeline
import torch

# ---------------------------------------------------------
# PERFORMANCE OPTIMIZATION (CPU/GPU)
# ---------------------------------------------------------

# Optimize CPU threads for PyTorch to avoid thread contention.
# Benchmarking showed 2 threads is the sweet spot for performance on this CPU.
if not torch.cuda.is_available():
    torch.set_num_threads(2)

# Automatically use GPU if available, otherwise fallback to CPU
device = 0 if torch.cuda.is_available() else -1


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Research Paper Intelligence System",
    page_icon="📚",
    layout="wide"
)


# ---------------------------------------------------------
# LOAD MODELS ONLY ONCE
# ---------------------------------------------------------

@st.cache_resource(show_spinner="🤖 Loading BART model...")
def load_bart():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=device
    )


@st.cache_resource(show_spinner="🤖 Loading T5 model...")
def load_t5():
    return pipeline(
        "summarization",
        model="t5-small",
        device=device
    )


# ---------------------------------------------------------
# CACHED SUMMARIZATION FUNCTIONS
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def generate_bart_summary(text, num_beams=1):
    bart = load_bart()

    result = bart(
        text,
        max_length=180,
        min_length=50,
        num_beams=num_beams,
        do_sample=False,
        truncation=True
    )

    return result[0]["summary_text"]


@st.cache_data(show_spinner=False)
def generate_t5_summary(text, num_beams=1):
    t5 = load_t5()

    result = t5(
        "summarize: " + text,
        max_length=180,
        min_length=50,
        num_beams=num_beams,
        do_sample=False,
        truncation=True
    )

    return result[0]["summary_text"]


# ---------------------------------------------------------
# CACHED SEMANTIC SEARCH & QA FUNCTIONS
# ---------------------------------------------------------

@st.cache_resource(show_spinner="🤖 Loading embedding model...")
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="🤖 Loading QA model...")
def load_qa_pipeline():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad",
        device=device
    )


@st.cache_data(show_spinner="🧠 Indexing document for semantic search...")
def get_chunk_embeddings(chunks):
    if not chunks:
        return None
    embedder = load_embedder()
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    return embeddings


def perform_semantic_search(query, chunks, embeddings, index, k=3):
    import numpy as np
    embedder = load_embedder()
    query_vector = embedder.encode([query], show_progress_bar=False)
    distances, indices = index.search(np.array(query_vector, dtype=np.float32), k=min(k, len(chunks)))
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:
            results.append((chunks[idx], float(dist)))
    return results


def answer_question(question, context):
    qa_pipe = load_qa_pipeline()
    result = qa_pipe(question=question, context=context)
    return result["answer"], float(result["score"])




# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("📚 AI Research Paper Intelligence System")

st.write(
    "An AI-powered system for research paper summarization, "
    "semantic search, and intelligent question answering."
)

st.divider()


# ---------------------------------------------------------
# PERFORMANCE/SPEED SETTINGS (SIDEBAR)
# ---------------------------------------------------------

st.sidebar.header("⚙️ Summarization Settings")

speed_mode = st.sidebar.radio(
    "Performance Mode:",
    options=[
        "🚀 Fast (Greedy Search - Recommended)",
        "⚖️ Balanced (2 Beams)",
        "🎯 Detailed (4 Beams)"
    ],
    index=0,
    help="Fast mode runs in greedy search mode, which is highly optimized for CPUs. Detailed mode runs full beam search."
)

# Map speed mode to num_beams
num_beams = 1
if "Balanced" in speed_mode:
    num_beams = 2
elif "Detailed" in speed_mode:
    num_beams = 4



# ---------------------------------------------------------
# PDF UPLOAD
# ---------------------------------------------------------

st.header("📤 Upload Research Paper")

uploaded_file = st.file_uploader(
    "Upload a research paper in PDF format",
    type=["pdf"]
)


# ---------------------------------------------------------
# PROCESS PDF
# ---------------------------------------------------------

if uploaded_file is not None:

    st.success("✅ Research paper uploaded successfully!")

    # -----------------------------------------------------
    # EXTRACT TEXT
    # -----------------------------------------------------

    with st.spinner("📄 Extracting text from PDF..."):

        try:
            text = extract_text_from_pdf(uploaded_file)

            if not text or not text.strip():
                st.error("❌ No text could be extracted from this PDF.")
                st.stop()

            cleaned_text = clean_text(text)

        except Exception as e:
            st.error(f"❌ Error while reading PDF: {e}")
            st.stop()


    # -----------------------------------------------------
    # CREATE CHUNKS
    # -----------------------------------------------------

    chunks = create_chunks(
        cleaned_text,
        chunk_size=1200,
        overlap=200
    )


    # -----------------------------------------------------
    # DOCUMENT INFORMATION
    # -----------------------------------------------------

    st.divider()

    st.header("📊 Document Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Characters",
            f"{len(cleaned_text):,}"
        )

    with col2:
        st.metric(
            "Words",
            f"{len(cleaned_text.split()):,}"
        )

    with col3:
        st.metric(
            "Text Chunks",
            len(chunks)
        )


    # -----------------------------------------------------
    # EXTRACTED TEXT
    # -----------------------------------------------------

    with st.expander("📄 View Extracted Text"):

        st.text_area(
            "Extracted Research Paper Text",
            cleaned_text,
            height=300
        )


    # -----------------------------------------------------
    # PREPARE TEXT FOR SUMMARIZATION
    # -----------------------------------------------------

    # Use the first chunk for the current version.
    # This keeps generation reasonably fast.

    if chunks:

        summary_text = chunks[0]

    else:

        summary_text = cleaned_text[:5000]


    # Make sure the input is large enough
    if len(summary_text.split()) < 30:
        summary_text = cleaned_text[:5000]


    # -----------------------------------------------------
    # ABSTRACTIVE SUMMARIZATION
    # -----------------------------------------------------

    st.divider()

    st.header("📝 Abstractive Text Summarization")


    # =====================================================
    # BART
    # =====================================================

    st.subheader("🤖 BART Summary")

    try:

        with st.spinner("Generating BART summary..."):

            bart_summary = generate_bart_summary(summary_text, num_beams=num_beams)

        st.success("✅ BART summary generated")

        st.write(bart_summary)

    except Exception as e:

        bart_summary = "BART summary could not be generated."

        st.error(
            f"❌ BART summarization error: {e}"
        )


    # =====================================================
    # T5
    # =====================================================

    st.subheader("🤖 T5 Summary")

    try:

        with st.spinner("Generating T5 summary..."):

            t5_summary = generate_t5_summary(summary_text, num_beams=num_beams)

        st.success("✅ T5 summary generated")

        st.write(t5_summary)

    except Exception as e:

        t5_summary = "T5 summary could not be generated."

        st.error(
            f"❌ T5 summarization error: {e}"
        )


    # -----------------------------------------------------
    # COMPARISON
    # -----------------------------------------------------

    st.divider()

    st.header("⚖️ BART vs T5 Comparison")

    col1, col2 = st.columns(2)


    with col1:

        st.subheader("🤖 BART")

        st.info(bart_summary)


    with col2:

        st.subheader("🤖 T5")

        st.info(t5_summary)


    # -----------------------------------------------------
    # SUMMARY STATISTICS
    # -----------------------------------------------------

    st.divider()

    st.header("📈 Summary Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Original Words",
            len(summary_text.split())
        )

    with col2:
        st.metric(
            "BART Words",
            len(bart_summary.split())
        )

    with col3:
        st.metric(
            "T5 Words",
            len(t5_summary.split())
        )

    with col4:

        compression = (
            len(bart_summary.split())
            / max(len(summary_text.split()), 1)
        ) * 100

        st.metric(
            "BART Compression",
            f"{compression:.1f}%"
        )


    # -----------------------------------------------------
    # BUILD SEMANTIC SEARCH FAISS INDEX (LAZY EVALUATION)
    # -----------------------------------------------------
    # Only build embeddings and index if chunks are available.
    embeddings = get_chunk_embeddings(chunks)

    if embeddings is not None:
        import numpy as np
        import faiss

        # Re-build FAISS index from cached numpy embeddings
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings, dtype=np.float32))

        st.divider()
        st.header("🔍 Semantic Search & Intelligent Q&A")
        st.write(
            "Query the document using natural language to perform a semantic search "
            "across the text chunks, or ask a question to extract direct answers."
        )

        # UI controls
        search_mode = st.radio(
            "Choose Feature Mode:",
            options=["💡 Ask a Question (Intelligent QA)", "🔎 Search Keywords/Phrases (Semantic Search)"],
            horizontal=True
        )

        if search_mode == "💡 Ask a Question (Intelligent QA)":
            user_question = st.text_input(
                "Ask a question about this research paper:",
                placeholder="e.g., What datasets were used? What is the main finding?"
            )

            if user_question:
                with st.spinner("Analyzing document and extracting answer..."):
                    # Retrieve the top 3 relevant chunks
                    search_results = perform_semantic_search(user_question, chunks, embeddings, index, k=3)

                    if search_results:
                        context = " ".join([res[0] for res in search_results])
                        try:
                            answer, score = answer_question(user_question, context)

                            if answer and answer.strip():
                                st.success(f"🎯 **Answer:** {answer}")
                                st.info(f"Confidence score: {score:.2f}")
                            else:
                                st.warning("⚠️ Could not extract a confident answer. Try rephrasing your question.")

                            # Display sources
                            with st.expander("📚 View Source Context Chunks"):
                                for i, (chunk, score_val) in enumerate(search_results):
                                    st.markdown(f"**Chunk {i+1}** (Distance: {score_val:.4f})")
                                    st.write(chunk)
                                    st.divider()
                        except Exception as e:
                            st.error(f"❌ Error extracting answer: {e}")
                    else:
                        st.warning("⚠️ No relevant sections found in the document to answer this question.")

        else: # Semantic Search Mode
            search_query = st.text_input(
                "Enter search term or phrase:",
                placeholder="e.g., neural network architecture, data preprocessing"
            )

            if search_query:
                with st.spinner("Searching document..."):
                    search_results = perform_semantic_search(search_query, chunks, embeddings, index, k=3)

                    if search_results:
                        st.subheader("🔎 Top Matching Chunks")
                        for i, (chunk, score_val) in enumerate(search_results):
                            st.markdown(f"**Chunk {i+1}** (Relevance Distance: {score_val:.4f})")
                            st.info(chunk)
                    else:
                        st.warning("⚠️ No matching chunks found.")


    # -----------------------------------------------------
    # SUCCESS MESSAGE
    # -----------------------------------------------------

    st.divider()

    st.success(
        "🎉 PDF extraction and system features completed successfully!"
    )