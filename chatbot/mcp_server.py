from mcp.server.fastmcp import FastMCP
import uuid

# Initialize MCP server
mcp = FastMCP("task-manager")

# ---- In-memory storage ----
TASKS = {}

# ---- Tool 1: Create Task ----
@mcp.tool()
def create_task(title: str) -> dict:
    task_id = str(uuid.uuid4())

    TASKS[task_id] = {
        "id": task_id,
        "title": title,
        "status": "pending"
    }

    return {
        "message": "Task created successfully",
        "task": TASKS[task_id]
    }

# ---- Tool 2: List Tasks ----
@mcp.tool()
def list_tasks() -> dict:
    return {
        "tasks": list(TASKS.values())
    }

# ---- Tool 3: Update Task ----
@mcp.tool()
def update_task(task_id: str, status: str) -> dict:
    if task_id not in TASKS:
        return {"error": "Task not found"}

    TASKS[task_id]["status"] = status

    return {
        "message": "Task updated",
        "task": TASKS[task_id]
    }

# ---- Tool 4: Delete Task ----
@mcp.tool()
def delete_task(task_id: str) -> dict:
    if task_id not in TASKS:
        return {"error": "Task not found"}

    deleted_task = TASKS.pop(task_id)

    return {
        "message": "Task deleted",
        "task": deleted_task
    }

# ---- Run MCP Server ----
if __name__ == "__main__":
    mcp.run()