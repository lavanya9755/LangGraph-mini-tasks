import queue
import uuid

import streamlit as st
from chatbot2 import chatbot, retrieve_all_threads, submit_async_task, generate_chat_title
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from db_utils import init_titles_table, save_chat_title, load_all_chat_titles

st.title("🤖 Lavanya's Chatbot")

# ── Utility functions ────────────────────────────────────────────────────────

def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = []
    # Don't add to chat_threads yet — it will be added when the first message is sent

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': str(thread_id)}})
    return state.values.get('messages', [])

# ── Bootstrap session state on every run ────────────────────────────────────

# Ensure the DB table exists
init_titles_table()

# Load persisted titles from DB once per session
if 'titles_loaded' not in st.session_state:
    st.session_state['chat_titles'] = load_all_chat_titles()   # {thread_id: title}
    # Reconstruct thread list from persisted titles (preserves order of insertion)
    st.session_state['chat_threads'] = list(st.session_state['chat_titles'].keys())
    st.session_state['titles_loaded'] = True

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title('LangGraph Chatbot')

repo_url = "https://github.com/lavanya9755/LangGraph-mini-tasks/tree/main/chatbot"
st.sidebar.markdown(
    f"""
    <a href="{repo_url}" target="_blank">
        <button style="
            background-color:#24292e;
            color:white;
            padding:8px 18px;
            border: 2px solid;
            border-color:grey;
            border-radius:8px;
            cursor:pointer;">
            GitHub Repo 
        </button>
    </a>
    """,
    unsafe_allow_html=True
)

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('Your Chats')

# Show latest first
threads = list(st.session_state['chat_threads'])
threads.reverse()

for thread in threads:
    title = st.session_state["chat_titles"].get(str(thread), "new chat")

    if st.sidebar.button(title, key=str(thread)):
        st.session_state['thread_id'] = thread
        messages = load_conversation(thread)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            # Skip empty content (e.g. tool-call-only AIMessages)
            if msg.content:
                temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages

# ── Chat history display ─────────────────────────────────────────────────────

for msg in st.session_state['message_history']:
    with st.chat_message(msg['role']):
        st.text(msg['content'])

# ── Chat input ───────────────────────────────────────────────────────────────

user_input = st.chat_input('Type here.....')

if user_input:
    thread_id = str(st.session_state['thread_id'])

    # ── Persist title on FIRST message of this thread ──────────────────────
    if thread_id not in st.session_state["chat_titles"]:
        title = generate_chat_title(user_input)
        st.session_state["chat_titles"][thread_id] = title
        st.session_state["chat_threads"].append(thread_id)
        save_chat_title(thread_id, title)          # ← write to SQLite

    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_turn",
    }

    # ── Streaming block ──────────────────────────────────────────────────────
    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
                    async for message_chunk, metadata in chatbot.astream(
                        {"messages": [HumanMessage(content=user_input)]},
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

                # Update status box for tool calls
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

                # Stream only assistant tokens
                if isinstance(message_chunk, AIMessage):
                    content = message_chunk.content
                    if isinstance(content, str):
                        yield content
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                yield item.get("text", "")

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant message to session
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )