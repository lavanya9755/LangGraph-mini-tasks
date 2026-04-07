from json import tool
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from pydantic import BaseModel, Field
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import requests

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0.7
)
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}




@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey="
    r = requests.get(url)
    return r.json()



tools = [search_tool, get_stock_price, calculator]
llm_with_tools = llm.bind_tools(tools)

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
graph.add_node("tools", ToolNode)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')

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