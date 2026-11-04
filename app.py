
import streamlit as st
import pymupdf
import hashlib

from src.pdf_processor import process_pdf_with_cache
from src.rag_pipeline import (
    build_rag_pipeline,
    retrieve_relevant_chunks
)
from src.llm import generate_answer
from src.summarizer import (
    get_chapters,
    summarize_chunks,
    summarize_chapter
)

st.set_page_config(
    page_title="StudyMate AI",
    page_icon="📚",
    layout="wide"
)

defaults = {
    "chunks": None,
    "index": None,
    "pdf_hash": None,
    "chat_history": [],
    "pdf_summary": None,
    "chapter_summaries": {}
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.title("📚 StudyMate AI")
st.write("Your personal AI-powered study assistant.")

st.sidebar.title("Navigation")

mode = st.sidebar.radio(
    "Choose your mode:",
    ["General AI Assistant", "PDF Study Assistant"]
)

st.sidebar.divider()

ai_mode = st.sidebar.selectbox(
    "Choose AI engine:",
    ["Offline", "Online", "Auto"]
)

if ai_mode == "Offline":
    st.sidebar.success("Local Qwen selected")
elif ai_mode == "Online":
    st.sidebar.info("Gemini selected")
else:
    st.sidebar.info("Gemini first, Qwen fallback")


if mode == "General AI Assistant":
    st.header("Ask a General Question")
    st.info(
        "General AI mode will be connected to the selected "
        "AI engine in a later development step."
    )

else:
    st.header("Study from your PDF")

    uploaded_file = st.file_uploader(
        "Upload your study PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:
        pdf_bytes = uploaded_file.getvalue()
        pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

        if st.session_state.pdf_hash != pdf_hash:
            st.session_state.pdf_hash = pdf_hash
            st.session_state.chunks = None
            st.session_state.index = None
            st.session_state.chat_history = []
            st.session_state.pdf_summary = None
            st.session_state.chapter_summaries = {}

        st.success("PDF uploaded successfully!")

        doc = pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf"
        )
        total_pages = len(doc)
        doc.close()

        st.write("Total pages:", total_pages)

        if st.button("Process PDF"):
            try:
                with st.spinner("Reading and processing PDF..."):
                    chunks = process_pdf_with_cache(pdf_bytes)

                if not chunks:
                    st.warning("No readable text found in this PDF.")
                    st.session_state.chunks = None
                    st.session_state.index = None
                else:
                    with st.spinner(
                        "Creating embeddings and vector database..."
                    ):
                        index = build_rag_pipeline(chunks)

                    st.session_state.chunks = chunks
                    st.session_state.index = index
                    st.session_state.chat_history = []
                    st.session_state.pdf_summary = None
                    st.session_state.chapter_summaries = {}

                    st.success("PDF is ready for studying!")
                    st.metric("Total chunks created", len(chunks))

            except Exception as error:
                st.error(f"Error processing PDF: {error}")

        if (
            st.session_state.chunks is not None
            and st.session_state.index is not None
        ):
            st.divider()

            tab1, tab2, tab3 = st.tabs([
                "PDF Summary",
                "Chapter Summary",
                "Ask Questions"
            ])

            # PDF SUMMARY
            with tab1:
                st.subheader("Full PDF Summary")

                if st.button("Generate PDF Summary"):
                    try:
                        with st.spinner("Generating PDF summary..."):
                            summary, engine = summarize_chunks(
                                st.session_state.chunks,
                                ai_mode
                            )

                        st.session_state.pdf_summary = {
                            "summary": summary,
                            "engine": engine
                        }

                    except Exception as error:
                        st.error(f"Summary error: {error}")

                if st.session_state.pdf_summary:
                    st.caption(
                        "AI engine: "
                        + st.session_state.pdf_summary["engine"]
                    )
                    st.markdown(
                        st.session_state.pdf_summary["summary"]
                    )

            # AUTOMATIC CHAPTER SUMMARY
            with tab2:
                st.subheader("Chapter-wise Summary")

                try:
                    chapters = get_chapters(
                        pdf_bytes,
                        st.session_state.chunks
                    )

                    if not chapters:
                        st.warning(
                            "No chapters could be detected automatically "
                            "from this PDF's bookmarks or OCR text."
                        )
                    else:
                        chapter_names = [
                            chapter["title"]
                            for chapter in chapters
                        ]

                        selected_name = st.selectbox(
                            "Select a chapter:",
                            chapter_names,
                            key="selected_chapter"
                        )

                        selected_chapter = next(
                            chapter for chapter in chapters
                            if chapter["title"] == selected_name
                        )

                        st.caption(
                            f"Detected PDF pages: "
                            f"{selected_chapter['start']}–"
                            f"{selected_chapter['end']}"
                        )

                        if st.button("Generate Chapter Summary"):
                            try:
                                with st.spinner(
                                    "Generating chapter summary..."
                                ):
                                    summary, engine = summarize_chapter(
                                        st.session_state.chunks,
                                        selected_chapter,
                                        ai_mode
                                    )

                                chapter_key = f"{selected_name}_{selected_chapter['start']}_{selected_chapter['end']}"

                                st.session_state.chapter_summaries[chapter_key] = {
                                    "summary": summary,
                                    "engine": engine
                                }

                            except Exception as error:
                                st.error(
                                    f"Chapter summary error: {error}"
                                )

                        chapter_key = f"{selected_name}_{selected_chapter['start']}_{selected_chapter['end']}"

                        if chapter_key in st.session_state.chapter_summaries:
                            result = st.session_state.chapter_summaries[chapter_key]

                            st.caption("AI engine: " + result["engine"])
                            st.markdown(result["summary"])

                except Exception as error:
                    st.error(f"Chapter detection error: {error}")

            
            # ASK QUESTIONS
            with tab3:
                st.subheader("Ask Questions About Your PDF")

                for i, chat in enumerate(
                    reversed(st.session_state.chat_history),
                    start=1
                ):
                    with st.expander(
                        f"Question {i}: {chat['question']}"
                    ):
                        st.markdown(chat["answer"])
                        st.caption(f"AI engine: {chat['engine']}")

                        if chat["pages"]:
                            st.caption(
                                "Retrieved PDF pages: "
                                + ", ".join(map(str, chat["pages"]))
                            )

                question = st.text_input(
                    "Enter your question:",
                    key="pdf_question"
                )

                if st.button("Ask StudyMate AI"):
                    if question.strip():
                        try:
                            with st.spinner("Searching your PDF..."):
                                results = retrieve_relevant_chunks(
                                    question,
                                    st.session_state.chunks,
                                    st.session_state.index,
                                    k=8
                                )

                            if results:
                                with st.spinner(
                                    f"Generating answer using {ai_mode} mode..."
                                ):
                                    answer, engine = generate_answer(
                                        question,
                                        results,
                                        ai_mode
                                    )

                                # Check whether the relevance checker rejected it.
                                rejected = answer.strip().startswith(
                                    "This topic was not found in the uploaded PDF."
                                )

                                # Only keep references for accepted answers.
                                retrieved_pages = (
                                    sorted({
                                        chunk.get("page_number")
                                        for chunk in results
                                        if chunk.get("page_number") is not None
                                    })
                                    if not rejected
                                    else []
                                )

                                st.subheader("StudyMate AI Answer")
                                st.caption(f"AI engine: {engine}")
                                st.markdown(answer)

                                # Show references only for accepted answers.
                                if not rejected:
                                    st.divider()
                                    st.subheader("Retrieved PDF References")

                                    st.write(
                                        "**Retrieved pages:** "
                                        + ", ".join(map(str, retrieved_pages))
                                    )

                                    for chunk in results:
                                        page = chunk.get(
                                            "page_number", "Unknown"
                                        )
                                        chunk_id = chunk.get(
                                            "chunk_id", "Unknown"
                                        )

                                        with st.expander(
                                            f"Page {page} | Chunk {chunk_id}"
                                        ):
                                            st.write(chunk.get("text", ""))

                                            score = chunk.get(
                                                "similarity_score"
                                            )

                                            if isinstance(score, (int, float)):
                                                st.caption(
                                                    f"Similarity score: {score:.4f}"
                                                )
                                            else:
                                                st.caption(
                                                    "Retrieved as neighbouring context"
                                                )

                                st.session_state.chat_history.append({
                                    "question": question,
                                    "answer": answer,
                                    "pages": retrieved_pages,
                                    "engine": engine
                                })

                            else:
                                st.warning("No relevant information found.")

                        except Exception as error:
                            st.error(
                                f"Error generating answer: {error}"
                            )
                    else:
                        st.warning("Please enter a question.")
