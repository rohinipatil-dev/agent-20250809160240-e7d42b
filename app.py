import streamlit as st
from openai import OpenAI
import io
import math
import json
import csv

# Initialize OpenAI client
client = OpenAI()

# --------------- Utility Functions ---------------

def read_file_as_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    filename = uploaded_file.name.lower()
    mime = uploaded_file.type or ""

    try:
        if filename.endswith(".pdf") or "pdf" in mime:
            return extract_text_from_pdf(uploaded_file.read())
        elif filename.endswith(".json") or "json" in mime:
            content = uploaded_file.read().decode("utf-8", errors="ignore")
            try:
                parsed = json.loads(content)
                return json.dumps(parsed, indent=2, ensure_ascii=False)
            except Exception:
                return content
        elif filename.endswith(".csv") or "csv" in mime:
            content = uploaded_file.read().decode("utf-8", errors="ignore")
            try:
                reader = csv.reader(io.StringIO(content))
                rows = list(reader)
                out_lines = []
                for r in rows:
                    out_lines.append(", ".join(r))
                return "\n".join(out_lines)
            except Exception:
                return content
        elif filename.endswith(".txt") or filename.endswith(".md") or "text" in mime:
            return uploaded_file.read().decode("utf-8", errors="ignore")
        else:
            # Try best-effort text decode
            return uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        import PyPDF2
    except ImportError:
        st.warning("PyPDF2 not installed. Install it with 'pip install PyPDF2' to enable PDF support.")
        return ""

    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        texts = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
                texts.append(t)
            except Exception:
                continue
        return "\n".join(texts).strip()
    except Exception as e:
        st.error(f"Failed to parse PDF: {e}")
        return ""


def clean_text(text: str) -> str:
    # Normalize whitespace
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list:
    if not text:
        return []

    # Prefer splitting by paragraphs to keep coherence
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    def flush_current():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # If paragraph is too large, split it
        if len(para) > chunk_size:
            start = 0
            while start < len(para):
                end = min(start + chunk_size, len(para))
                piece = para[start:end]
                if current:
                    flush_current()
                chunks.append(piece.strip())
                start = end - overlap if end - overlap > start else end
        else:
            if len(current) + len(para) + 1 <= chunk_size:
                current = (current + "\n\n" + para) if current else para
            else:
                # flush and start new chunk with overlap
                if current:
                    flush_current()
                current = para

    if current:
        flush_current()

    # Add overlaps between adjacent chunks
    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks = []
        for i, ch in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(ch)
            else:
                prev = chunks[i - 1]
                tail = prev[-overlap:] if len(prev) > overlap else prev
                merged = tail + "\n\n" + ch
                overlapped_chunks.append(merged)
        chunks = overlapped_chunks

    # Final cleanup
    chunks = [c.strip() for c in chunks if c.strip()]
    return chunks


def vector_norm(v):
    return math.sqrt(sum(x * x for x in v)) if v else 0.0


def cosine_similarity(a, b):
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = vector_norm(a)
    nb = vector_norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def batched(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def embed_texts(texts: list, batch_size: int = 64) -> list:
    embeddings = []
    for batch in batched(texts, batch_size):
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        # Ensure order is preserved
        for item in resp.data:
            embeddings.append(item.embedding)
    return embeddings


def retrieve_top_k(query: str, chunks: list, chunk_embeddings: list, k: int = 5):
    if not chunks or not chunk_embeddings:
        return []
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    ).data[0].embedding

    scored = []
    for idx, emb in enumerate(chunk_embeddings):
        score = cosine_similarity(q_emb, emb)
        scored.append((score, idx))
    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[: max(1, min(k, len(scored)))]
    return top


def build_context(top_items, chunks, max_chars: int = 6000):
    parts = []
    total = 0
    for score, idx in top_items:
        snippet = chunks[idx]
        header = f"[Chunk #{idx} | Score: {round(score, 4)}]\n"
        block = header + snippet.strip()
        if total + len(block) + 2 > max_chars:
            break
        parts.append(block)
        total += len(block) + 2
    return "\n\n-----\n\n".join(parts)


def answer_question(question: str, context: str, model: str = "gpt-4", temperature: float = 0.0) -> str:
    system_prompt = (
        "You are a helpful assistant that answers user questions strictly using the provided context. "
        "If the answer is not present in the context, reply: 'I couldn't find that in the document.' "
        "Cite which chunk(s) you used when relevant."
    )
    user_content = (
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Instructions:\n"
        "- Use only the information in the context.\n"
        "- If unsure or missing info, say you couldn't find it in the document.\n"
        "- Keep answers concise and point to chunk numbers when helpful."
    )

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
    )
    return response.choices[0].message.content


# --------------- Streamlit App ---------------

def init_session_state():
    defaults = {
        "doc_text": "",
        "chunks": [],
        "embeddings": [],
        "file_name": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def process_uploaded_file(uploaded_file, chunk_size: int, overlap: int):
    text = read_file_as_text(uploaded_file)
    text = clean_text(text)
    if not text:
        st.warning("No readable text extracted from the file.")
        return

    with st.spinner("Chunking document..."):
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    with st.spinner("Creating embeddings..."):
        embeddings = embed_texts(chunks)

    st.session_state.doc_text = text
    st.session_state.chunks = chunks
    st.session_state.embeddings = embeddings
    st.session_state.file_name = uploaded_file.name if uploaded_file else None
    st.success(f"Document processed successfully. {len(chunks)} chunks ready.")


def sidebar_controls():
    st.sidebar.header("Settings")
    model = st.sidebar.selectbox("Chat model", options=["gpt-4", "gpt-3.5-turbo"], index=0)
    chunk_size = st.sidebar.slider("Chunk size (characters)", min_value=400, max_value=2000, value=1200, step=100)
    overlap = st.sidebar.slider("Overlap (characters)", min_value=0, max_value=400, value=200, step=20)
    top_k = st.sidebar.slider("Top-K chunks to retrieve", min_value=1, max_value=10, value=5, step=1)
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
    if st.sidebar.button("Clear session"):
        for key in ["doc_text", "chunks", "embeddings", "file_name"]:
            st.session_state[key] = "" if key == "doc_text" else []
        st.sidebar.success("Session cleared.")
    return model, chunk_size, overlap, top_k, temperature


def main():
    st.set_page_config(page_title="RAG Q&A over Uploaded File", page_icon="📄", layout="wide")
    st.title("📄 RAG Q&A over Uploaded File")
    st.caption("Upload a document and ask questions. The assistant answers using only the document content.")

    init_session_state()
    model, chunk_size, overlap, top_k, temperature = sidebar_controls()

    uploaded_file = st.file_uploader("Upload a document (PDF, TXT, MD, CSV, JSON)", type=["pdf", "txt", "md", "csv", "json"])

    col1, col2 = st.columns([2, 1])

    with col1:
        if uploaded_file is not None:
            if st.session_state.file_name != uploaded_file.name or not st.session_state.chunks:
                process_uploaded_file(uploaded_file, chunk_size, overlap)
        else:
            st.info("Please upload a file to start.")

        if st.session_state.chunks:
            st.success(f"Loaded: {st.session_state.file_name} • {len(st.session_state.chunks)} chunks")

            query = st.text_input("Enter your question about the document")
            ask = st.button("Ask")

            if ask and query.strip():
                with st.spinner("Retrieving relevant context..."):
                    top = retrieve_top_k(query, st.session_state.chunks, st.session_state.embeddings, k=top_k)
                    context = build_context(top, st.session_state.chunks)

                with st.spinner("Generating answer..."):
                    answer = answer_question(query, context, model=model, temperature=temperature)

                st.subheader("Answer")
                st.write(answer)

                with st.expander("Show retrieved context"):
                    st.write(context)
        else:
            st.write("No document loaded yet.")

    with col2:
        st.subheader("Document Preview")
        if st.session_state.doc_text:
            preview = st.session_state.doc_text[:4000]
            st.text_area("Text preview", value=preview, height=500)
        else:
            st.write("No preview available.")


if __name__ == "__main__":
    main()