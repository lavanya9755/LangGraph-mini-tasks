import queue
import uuid

import streamlit as st
from chatbot2 import chatbot, retrieve_all_threads, submit_async_task, generate_chat_title
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from db_utils import init_titles_table, save_chat_title, load_all_chat_titles
from rag_engine import ingest_files, query_rag, get_rag_stats, clear_rag

# ─────────────────────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Lavanya's Chatbot", page_icon="🤖", layout="wide")

st.markdown("""
<style>
.rag-badge {
    display: inline-block;
    background: #1a3a2a;
    border: 1px solid #2d6a4f;
    color: #74c69d;
    font-size: 0.75rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin: 2px 3px;
}
.rag-info-box {
    background: #0d1f17;
    border-left: 3px solid #2d6a4f;
    border-radius: 6px;
    padding: 8px 14px;
    margin-bottom: 10px;
    font-size: 0.82rem;
    color: #74c69d;
}
.rag-stat { font-weight: 600; color: #52b788; }
.doc-pill {
    background: #1e2d1e;
    border: 1px solid #2d6a4f;
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 0.78rem;
    color: #95d5b2;
    margin: 3px 0;
    display: block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    st.session_state['thread_id'] = generate_thread_id()
    st.session_state['message_history'] = []

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': str(thread_id)}})
    return state.values.get('messages', [])

def extract_display_messages(raw_messages):
    display = []
    for msg in raw_messages:
        if isinstance(msg, HumanMessage):
            text = msg.content if isinstance(msg.content, str) else ""
            if text.strip():
                display.append({"role": "user", "content": text})
        elif isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                text = " ".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ).strip()
            else:
                text = ""
            if text:
                display.append({"role": "assistant", "content": text})
    return display

def build_rag_prompt(user_question: str, rag_result: dict) -> str:
    ctx = rag_result.get("context", "")
    if not ctx:
        return user_question
    return (
        "Use the following document excerpts to help answer the question. "
        "If the documents don't contain enough info, use your own knowledge too.\n\n"
        f"--- DOCUMENT CONTEXT ---\n{ctx}\n--- END CONTEXT ---\n\n"
        f"User question: {user_question}"
    )

# ─────────────────────────────────────────────────────────────────────────────
#  Bootstrap session state
# ─────────────────────────────────────────────────────────────────────────────
init_titles_table()

if 'titles_loaded' not in st.session_state:
    st.session_state['chat_titles'] = load_all_chat_titles()
    st.session_state['chat_threads'] = list(st.session_state['chat_titles'].keys())
    st.session_state['titles_loaded'] = True

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'rag_file_names' not in st.session_state:
    st.session_state['rag_file_names'] = []

#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🤖 LangGraph Chatbot")

repo_url = "https://github.com/lavanya9755/LangGraph-mini-tasks/tree/main/chatbot"
st.sidebar.markdown(
    f'<a href="{repo_url}" target="_blank">'
    f'<button style="background-color:#24292e;color:white;padding:8px 18px;'
    f'border:2px solid grey;border-radius:8px;cursor:pointer;width:100%">'
    f'GitHub Repo</button></a>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

if st.sidebar.button("➕  New Chat", use_container_width=True):
    reset_chat()

# ── Indexed docs ──────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("#### 📄 Indexed Documents")
stats = get_rag_stats()
if stats["names"]:
    for name in stats["names"]:
        st.sidebar.markdown(f'<span class="doc-pill">📎 {name}</span>', unsafe_allow_html=True)
    st.sidebar.caption(f"Total chunks in index: **{stats['chunks']}**")
    if st.sidebar.button("🗑️  Clear all documents", use_container_width=True):
        clear_rag()
        st.session_state['rag_file_names'] = []
        st.rerun()
else:
    st.sidebar.caption("No documents indexed yet.")

# ── Chat history list ─────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("#### 💬 Your Chats")
threads = list(st.session_state['chat_threads'])
threads.reverse()
for thread in threads:
    title = st.session_state["chat_titles"].get(str(thread), "new chat")
    if st.sidebar.button(title, key=str(thread), use_container_width=True):
        st.session_state['thread_id'] = thread
        raw_messages = load_conversation(thread)
        st.session_state['message_history'] = extract_display_messages(raw_messages)

# ─────────────────────────────────────────────────────────────────────────────
#  Main area
# ─────────────────────────────────────────────────────────────────────────────
st.title("🤖 Lavanya's Chatbot")

# ── File uploader ─────────────────────────────────────────────────────────────
with st.expander("➕  Attach documents for RAG  (PDF · DOCX · PPTX)", expanded=False):
    uploaded_files = st.file_uploader(
        label="Upload files",
        type=["pdf", "doc", "docx", "ppt", "pptx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="rag_uploader",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        if uploaded_files:
            new_files = [f for f in uploaded_files
                         if f.name not in st.session_state['rag_file_names']]
            if new_files:
                st.caption(f"Ready to index: {', '.join(f.name for f in new_files)}")
            else:
                st.caption("All selected files are already indexed.")
    with col2:
        if uploaded_files and st.button("📥  Index now", use_container_width=True):
            with st.spinner("Chunking & embedding…"):
                result = ingest_files(uploaded_files)
            st.session_state['rag_file_names'] = result["names"]
            st.success(
                f"✅ Indexed **{result['docs']}** document(s) → "
                f"**{result['chunks']}** chunks total"
            )
            st.rerun()

# ── Chat history display ──────────────────────────────────────────────────────
for msg in st.session_state['message_history']:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Type here…")

if user_input:
    thread_id = str(st.session_state['thread_id'])

    # Persist title on first message of this thread
    if thread_id not in st.session_state["chat_titles"]:
        title = generate_chat_title(user_input)
        st.session_state["chat_titles"][thread_id] = title
        st.session_state["chat_threads"].append(thread_id)
        save_chat_title(thread_id, title)

    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ── RAG retrieval ─────────────────────────────────────────────────────────
    rag_result = query_rag(user_input, k=4)
    has_rag = rag_result["chunks_used"] > 0

    if has_rag:
        docs_used   = rag_result["docs_used"]
        chunks_used = rag_result["chunks_used"]
        sources     = rag_result["sources"]

        seen = set()
        source_pills = []
        for s in sources:
            label = s["file"]
            if s["page"] is not None:
                label += f" · p{int(s['page']) + 1}"
            if label not in seen:
                seen.add(label)
                source_pills.append(label)

        pills_html = "".join(
            f'<span class="rag-badge">📎 {p}</span>' for p in source_pills
        )
        st.markdown(
            f'<div class="rag-info-box">'
            f'🔍 Answering from <span class="rag-stat">{docs_used} document(s)</span> · '
            f'<span class="rag-stat">{chunks_used} chunk(s)</span> retrieved'
            f'<br/>{pills_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

    llm_question = build_rag_prompt(user_input, rag_result) if has_rag else user_input

    CONFIG = {
        "configurable": {"thread_id": thread_id},
        "metadata":     {"thread_id": thread_id},
        "run_name":     "chat_turn",
    }

    # ── Streaming assistant reply ─────────────────────────────────────────────
    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
                    async for message_chunk, metadata in chatbot.astream(
                        {"messages": [HumanMessage(content=llm_question)]},
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        event_queue.put((message_chunk, metadata))
                except Exception as exc:
                    event_queue.put(("error", exc))
                finally:
                    event_queue.put(None)

            submit_async_task(run_stream())

            while True:
                item = event_queue.get()
                if item is None:
                    break
                message_chunk, metadata = item
                if message_chunk == "error":
                    raise metadata

                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage):
                    content = message_chunk.content
                    if isinstance(content, str):
                        yield content
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                yield block.get("text", "")

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )