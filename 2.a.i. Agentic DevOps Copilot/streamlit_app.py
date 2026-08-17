"""Minimal Streamlit chat UI for the DevOps Copilot.

Run:  streamlit run streamlit_app.py

Human-in-the-loop: toggle "Auto-approve write actions" in the sidebar. When OFF
(default), the agent will propose a ticket but not create it. When ON, write
actions are executed (demo convenience).
"""
from __future__ import annotations

import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from app import tools
from app.agent import build_agent

load_dotenv()
st.set_page_config(page_title="DevOps Copilot", page_icon="🤖")
st.title("🤖 Agentic DevOps Copilot")

auto_approve = st.sidebar.checkbox("Auto-approve write actions (demo)", value=False)
tools.CONFIRM_CALLBACK = (lambda p: True) if auto_approve else None

if "agent" not in st.session_state:
    st.session_state.agent = build_agent()
    st.session_state.thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    st.session_state.history = []

for role, text in st.session_state.history:
    st.chat_message(role).write(text)

if prompt := st.chat_input("Ask about pipelines, failures, costs, tickets…"):
    st.session_state.history.append(("user", prompt))
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("thinking…"):
            result = st.session_state.agent.invoke(
                {"messages": [HumanMessage(content=prompt)]}, st.session_state.thread
            )
            reply = result["messages"][-1].content
        st.write(reply)
    st.session_state.history.append(("assistant", reply))
