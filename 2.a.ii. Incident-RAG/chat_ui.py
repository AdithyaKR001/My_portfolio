"""Streamlit chat UI for the incident knowledge base.

Run:  streamlit run chat_ui.py
(Build the index first: python -m rag.ingest)
"""
from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from rag.pipeline import answer

load_dotenv()
st.set_page_config(page_title="Incident-RAG", page_icon="🛠️")
st.title("🛠️ Incident-RAG — ask the knowledge base")

if prompt := st.chat_input("e.g. How do we fix OrderServiceIT timeouts against hana-test?"):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("retrieving + generating…"):
            result = answer(prompt)
        st.write(result["answer"])
        st.caption("Citations: " + ", ".join(result["citations"]))
        with st.expander("Retrieved context"):
            for h in result["contexts"]:
                st.markdown(f"**[{h['doc_id']}]** {h['title']}")
                st.text(h["content"][:600])
