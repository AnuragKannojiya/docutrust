"""Streamlit demo UI for DocuTrust.

Run with:
    streamlit run ui/streamlit_app.py

The UI talks to the FastAPI service over HTTP — it does not import any
application modules directly. Configure the API base via the sidebar.
"""
from __future__ import annotations

import os

import httpx
import streamlit as st

API_BASE = st.sidebar.text_input(
    "API base URL",
    value=os.environ.get("DOCUTRUST_API", "http://127.0.0.1:8000"),
    help="DocuTrust FastAPI service",
)


def _api(path: str, **kwargs):
    return httpx.get(f"{API_BASE}{path}", timeout=30.0, **kwargs)


def _post(path: str, **kwargs):
    return httpx.post(f"{API_BASE}{path}", timeout=120.0, **kwargs)


st.set_page_config(page_title="DocuTrust", page_icon="📚", layout="wide")
st.title("DocuTrust")
st.caption("Self-correcting enterprise RAG with strict citations.")

# --- Client selector ---------------------------------------------------------

@st.cache_data(ttl=30)
def _list_clients():
    try:
        r = _api("/clients")
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as exc:
        st.error(f"Could not reach API at {API_BASE}: {exc}")
        return []


clients = _list_clients()
if not clients:
    st.warning(
        "No clients yet. Run `python -m scripts.seed_corpus` after starting the API."
    )
    st.stop()

client_labels = {c["_id"]: f"{c['name']} ({c.get('industry', 'n/a')})" for c in clients}
selected = st.sidebar.selectbox(
    "Client",
    options=list(client_labels.keys()),
    format_func=lambda cid: client_labels[cid],
)

st.sidebar.divider()
st.sidebar.markdown(
    "**Three demo flows**\n"
    "1. **Happy path** — ask something answered by the policy. Expect a cited answer.\n"
    "2. **Low relevance → rewrite → web** — ask outside the corpus. Expect a `web_search` step.\n"
    "3. **Refusal** — ask something the LLM is tempted to answer with general knowledge."
)

# --- Chat --------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander(f"Citations ({len(msg['citations'])})"):
                for c in msg["citations"]:
                    st.markdown(
                        f"**{c['section_path']}**  \n"
                        f"source: `{c['source']}`  \n"
                        f"{c.get('url') or ''}  \n"
                        f"> {c['quote']}"
                    )
        if msg.get("steps"):
            with st.expander("Trace steps"):
                for s in msg["steps"]:
                    st.markdown(
                        f"**{s['node']}** ({s['duration_ms']} ms)  \n"
                        f"```json\n{ {k: v for k, v in s.items() if k not in ('node', 'duration_ms')} }\n```"
                    )

question = st.chat_input("Ask a question about the policy corpus…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Running CRAG…"):
            try:
                r = _post(
                    "/ask",
                    json={"client_id": selected, "question": question},
                )
                r.raise_for_status()
                data = r.json()
            except httpx.HTTPError as exc:
                st.error(f"API error: {exc}")
                st.stop()

        if data.get("refused"):
            st.warning(
                f"**Refused** — {data.get('refusal_reason', 'no reason given')}\n\n"
                "Try a question more directly answered by the corpus, or check "
                "the trace steps below for what the system retrieved."
            )
        else:
            st.markdown(data["answer"])

        citations = data.get("citations", [])
        steps = data.get("steps", [])
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": data.get("answer") or "(refused)",
                "citations": citations,
                "steps": steps,
            }
        )

        if citations:
            with st.expander(f"Citations ({len(citations)})"):
                for c in citations:
                    st.markdown(
                        f"**{c['section_path']}**  \n"
                        f"source: `{c['source']}`  \n"
                        f"{c.get('url') or ''}  \n"
                        f"> {c['quote']}"
                    )
        if steps:
            with st.expander("Trace steps"):
                for s in steps:
                    st.markdown(
                        f"**{s['node']}** ({s['duration_ms']} ms)  \n"
                        f"```json\n{ {k: v for k, v in s.items() if k not in ('node', 'duration_ms')} }\n```"
                    )
