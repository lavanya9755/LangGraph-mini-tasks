from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0.7
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    prompt = f'always reply to my messages in chatchy british words taken from bridgeton series if you know... for the given message: {messages}'
    response = llm.invoke(prompt)
    return {"messages": [response]}

# Checkpointer
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False) #sqlite doesnt allow multiple threads in defualt , therefore check_same thread needs to eb false
checkpointer = SqliteSaver(conn = conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for chk in checkpointer.list(None):
        all_threads.add(chk.config['configurable']['thread_id'])

    return list(all_threads)
     #this means all checpoints needs to be rpitn

# CONFIG = {'configurable': {'thread_id': 'thread-3'}}

# result = chatbot.invoke(
#                 {'messages': [HumanMessage(content="hello im lavanya!")]},
#                 config= CONFIG
               
#             )
# print(result)

# there are diff types stream_mode: updates, custome, values messages