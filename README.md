# 📚 AI Research Paper Intelligence System

An AI-powered Streamlit web application designed for research paper summarization, semantic search, and intelligent question answering (Q&A). This project helps researchers, students, and academics quickly digest scientific literature using state-of-the-art NLP models.

## 🌟 Key Features

1. **📄 PDF Text Extraction & Preprocessing**: Clean text extraction using PyMuPDF (`fitz`) and custom regex-based sentence segmentation/chunking.
2. **📝 Multi-Model Summarization**: Abstractive summarizers using:
   - **BART** (`facebook/bart-large-cnn`)
   - **T5** (`t5-small`)
   - Detailed model comparisons and text compression metrics.
3. **🔎 Semantic Search**: Text chunks indexed using a **FAISS** vector database and encoded with `sentence-transformers` (`all-MiniLM-L6-v2`) for instant semantic retrieval.
4. **💡 Intelligent Q&A**: Real-time extraction of answers from retrieved chunks using a pre-trained QA pipeline (`distilbert-base-cased-distilled-squad`).
5. **⚙️ Performance Controls**: Custom CPU thread optimization (PyTorch) and adjustable speed/quality parameters to balance between fast execution (greedy search) and detailed beam search.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Deep Learning / NLP**: PyTorch, Hugging Face Transformers (`transformers`, `pipeline`), SentenceTransformers
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **PDF Processing**: PyMuPDF (`fitz`)

---

## 🚀 Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/AI-Research-Paper-Intelligence-System.git
   cd AI-Research-Paper-Intelligence-System
   ```

2. **Activate the Virtual Environment**:
   ```bash
   .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Ensure PyTorch, Transformers, SentenceTransformers, and FAISS-CPU are installed).*

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

---

## 💡 How to Use

1. **Upload**: Drag and drop any research paper in PDF format.
2. **Summarize**: Compare the generated abstractive summaries from BART and T5, and view the compression statistics.
3. **Configure**: Use the sidebar options to adjust the Performance Mode (`Fast`, `Balanced`, or `Detailed`).
4. **Query & Ask**:
   - Ask questions like *"What is the main methodology?"* or *"What dataset was used?"* to retrieve the exact answers from the paper with confidence scores.
   - Enter terms to search keywords and retrieve matching paragraphs.
