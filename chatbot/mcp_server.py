from mcp.server.fastmcp import FastMCP
import uuid
import sqlite3

# Initialize MCP server
mcp = FastMCP("task-manager")

# ---- Database Setup ----
conn = sqlite3.connect("tasks.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    title TEXT,
    status TEXT
)
""")
conn.commit()

# ---- Tool 1: Create Task ----
@mcp.tool()
def create_task(title: str) -> dict:
    task_id = str(uuid.uuid4())
    status = "pending"

    cursor.execute(
        "INSERT INTO tasks (id, title, status) VALUES (?, ?, ?)",
        (task_id, title, status)
    )
    conn.commit()

    return {
        "message": "Task created successfully",
        "task": {
            "id": task_id,
            "title": title,
            "status": status
        }
    }

# ---- Tool 2: List Tasks ----
@mcp.tool()
def list_tasks() -> dict:
    cursor.execute("SELECT id, title, status FROM tasks")
    rows = cursor.fetchall()

    tasks = [
        {"id": row[0], "title": row[1], "status": row[2]}
        for row in rows
    ]

    return {"tasks": tasks}

# ---- Tool 3: Update Task ----
@mcp.tool()
def update_task(task_id: str, status: str) -> dict:
    cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
    task = cursor.fetchone()

    if not task:
        return {"error": "Task not found"}

    cursor.execute(
        "UPDATE tasks SET status = ? WHERE id = ?",
        (status, task_id)
    )
    conn.commit()

    return {
        "message": "Task updated",
        "task": {
            "id": task_id,
            "title": task[1],
            "status": status
        }
    }

# ---- Tool 4: Delete Task ----
@mcp.tool()
def delete_task(task_id: str) -> dict:
    cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
    task = cursor.fetchone()

    if not task:
        return {"error": "Task not found"}

    cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()

    return {
        "message": "Task deleted",
        "task": {
            "id": task[0],
            "title": task[1],
            "status": task[2]
        }
    }

# ---- Run MCP Server ----
if __name__ == "__main__":
    mcp.run()