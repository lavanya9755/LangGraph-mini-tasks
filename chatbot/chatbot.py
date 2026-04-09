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
from langchain_core.tools import tool, BaseTool
from langgraph.prebuilt import create_react_agent
# from langchain.agents import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_openai import ChatOpenAI
import sqlite3
import requests
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import aiosqlite # type: ignore
import asyncio
import threading
import os
load_dotenv()

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)



llm = ChatOpenAI(
    model="anthropic/claude-3-haiku",  # or any model
    temperature=0.7,
    openai_api_key= os.getenv("OPEN_ROUTER_API"),
    openai_api_base="https://openrouter.ai/api/v1"
)

# llm = ChatGoogleGenerativeAI(
#     model = "gemini-2.5-flash",
#     temperature = 0.7
# )

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
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey= ****"
    r = requests.get(url)
    return r.json()


client = MultiServerMCPClient(
    {
        "task_manager": {
            "transport": "stdio",
            "command": "python",
            "args":["mcp_server.py"]
        },
        "expense": {
            "transport": "streamable_http",  
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
        }
    }
)
def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception as e:
        print("❌ MCP load failed:", e)
        return []


mcp_tools = load_mcp_tools()
print("MCP tools:", mcp_tools)


tools = [search_tool, get_stock_price, *mcp_tools]
llm_with_tools = llm.bind_tools(tools) if tools else llm
agent = create_react_agent(llm_with_tools, tools)  # ✅ correct
# print("TOOLS\n",tools)
# print(llm_with_tools)
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

async def chat_node(state: ChatState):
    messages = state["messages"]

    system_msg = {
        "role": "system",
        "content": "You are a helpful assistant. Greet the user and help with their request."
    }

    response = await agent.ainvoke({
        "messages": [system_msg] + messages
    })

    return {
        "messages": response["messages"]
    }

tool_node = ToolNode(tools)

async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)


    
checkpointer = run_async(_init_checkpointer())
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
# graph.add_node("tools",tool_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads():
    return run_async(_alist_threads())
     #this means all checpoints needs to be print

# CONFIG = {'configurable': {'thread_id': 'thread-3'}}

# result = chatbot.invoke(
#                 {'messages': [HumanMessage(content="hello im lavanya!")]},
#                 config= CONFIG
               
#             )
# print(result)

# there are diff types stream_mode: updates, custome, values messages