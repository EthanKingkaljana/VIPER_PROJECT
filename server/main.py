# server/main.py

import os
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .database import get_connection
from .utils_io import save_upload, static_url, ensure_dirs, OUT, UPLOAD_DIR
from .routers import (
    features, matching, alignment, quality,
    classification, enhancement, restoration, segmentation
)

app = FastAPI(title="N2N Unified API")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Static Setup ----------------
ensure_dirs(UPLOAD_DIR, OUT)
app.mount("/static", StaticFiles(directory=OUT), name="static")

# ---------------- Routers ----------------
app.include_router(features.router,       prefix="/api/feature")
app.include_router(matching.router,       prefix="/api/match")
app.include_router(alignment.router,      prefix="/api/alignment")
app.include_router(quality.router,        prefix="/api/quality")
app.include_router(classification.router, prefix="/api/classify")
app.include_router(enhancement.router,    prefix="/api/enhancement")
app.include_router(restoration.router,    prefix="/api/restoration")
app.include_router(segmentation.router,   prefix="/api/segmentation")

# ---------------- Upload Endpoint ----------------
@app.post("/api/upload")
async def api_upload(files: list[UploadFile] = File(...)):
    saved = []
    for f in files:
        path = await save_upload(f, UPLOAD_DIR)

        saved.append({
            "name": f.filename,
            "path": path,
            "url": static_url(path, OUT)
        })
    return {"files": saved}


# =================================================
# =============== WORKFLOW SECTION ===============
# =================================================

# Request model from frontend
class Workflow(BaseModel):
    name: str
    nodes: list
    edges: list
    viewport: dict


# SAVE WORKFLOW
@app.post("/api/workflows")
def save_workflow(workflow: Workflow):
    try:
        conn = get_connection()
        cur = conn.cursor()

        base_name = workflow.name

        # 🔥 Find latest version for this base name
        cur.execute("""
            SELECT COUNT(*)
            FROM workflows
            WHERE name LIKE %s
        """, (f"{base_name} v%",))

        count = cur.fetchone()[0]
        version = count + 1

        versioned_name = f"{base_name} v{version}"

        zoom = workflow.viewport.get("zoom", 1)
        x = workflow.viewport.get("x", 0)
        y = workflow.viewport.get("y", 0)

        # Insert workflow
        cur.execute("""
            INSERT INTO workflows (name, zoom, viewport_x, viewport_y)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """, (versioned_name, zoom, x, y))

        workflow_id = cur.fetchone()[0]

        # Insert nodes
        for node in workflow.nodes:
            cur.execute("""
                INSERT INTO nodes
                (workflow_id, node_id, type, position_x, position_y, data)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                workflow_id,
                node.get("id"),
                node.get("type"),
                node.get("position", {}).get("x", 0),
                node.get("position", {}).get("y", 0),
                json.dumps(node.get("data", {}))
            ))

        # Insert edges
        for edge in workflow.edges:
            cur.execute("""
                INSERT INTO edges
                (workflow_id, source, target, type)
                VALUES (%s, %s, %s, %s)
            """, (
                workflow_id,
                edge.get("source"),
                edge.get("target"),
                edge.get("type")
            ))

        conn.commit()
        cur.close()
        conn.close()

        return {
            "message": "Workflow version saved",
            "id": workflow_id,
            "name": versioned_name
        }

    except Exception as e:
        return {"error": str(e)}



# GET ALL WORKFLOWS (optional but useful)
@app.get("/api/workflows/{workflow_id}")
def get_workflow(workflow_id: int):
    conn = get_connection()
    cur = conn.cursor()

    # Get workflow
    cur.execute("""
        SELECT name, zoom, viewport_x, viewport_y
        FROM workflows
        WHERE id = %s
    """, (workflow_id,))
    wf = cur.fetchone()

    if not wf:
        return {"error": "Workflow not found"}

    # Get nodes
    cur.execute("""
        SELECT node_id, type, position_x, position_y, data
        FROM nodes
        WHERE workflow_id = %s
    """, (workflow_id,))
    nodes_rows = cur.fetchall()

    # Get edges
    cur.execute("""
        SELECT source, target, type
        FROM edges
        WHERE workflow_id = %s
    """, (workflow_id,))
    edges_rows = cur.fetchall()

    cur.close()
    conn.close()

    nodes = []
    for n in nodes_rows:
        nodes.append({
            "id": n[0],
            "type": n[1],
            "position": {"x": n[2], "y": n[3]},
            "data": n[4]
        })

    edges = []
    for e in edges_rows:
        edges.append({
            "source": e[0],
            "target": e[1],
            "type": e[2]
        })

    return {
        "id": workflow_id,
        "name": wf[0],
        "nodes": nodes,
        "edges": edges,
        "viewport": {
            "zoom": wf[1],
            "x": wf[2],
            "y": wf[3]
        }
    }

