import queue
import uuid

import streamlit as st
from chatbot import chatbot, retrieve_all_threads, submit_async_task
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# utility functionsss
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id  not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
#in the output of get_state() of chatbot.invoke() there was the chat convo in Values as dictionary , in messages attribute

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])

# message_history = []
#streamlit has a dictionay name as session state, function: when user press enter, its doesnt reload or empty , the data will remain here

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

add_thread(st.session_state['thread_id'])


#side bar

st.sidebar.title('LangGraph Chatbot')
st.sidebar.button('GitHub Repo')
    
if st.sidebar.button('New Chat'):
    reset_chat()
st.sidebar.header('Your Chats')
for thread in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread)):
        st.session_state['thread_id'] = thread
        messages = load_conversation(thread)

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            temp_messages.append({'role':role,'content':msg.content})

        st.session_state['message_history'] = temp_messages


for msg in st.session_state['message_history']:
    with st.chat_message(msg['role']):
        st.text(msg['content'])


user_input = st.chat_input('Type here.....')

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # streaming block
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

                # Lazily create & update the SAME status container when any tool runs
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

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    content = message_chunk.content

                    # Case 1: normal string
                    if isinstance(content, str):
                        yield content

                    # Case 2: structured output (list of dicts)
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                yield item.get("text", "")

        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
