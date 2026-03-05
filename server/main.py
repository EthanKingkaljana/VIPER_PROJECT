# server/main.py

import json
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from psycopg2.extras import Json

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
app.include_router(features.router, prefix="/api/feature")
app.include_router(matching.router, prefix="/api/match")
app.include_router(alignment.router, prefix="/api/alignment")
app.include_router(quality.router, prefix="/api/quality")
app.include_router(classification.router, prefix="/api/classify")
app.include_router(enhancement.router, prefix="/api/enhancement")
app.include_router(restoration.router, prefix="/api/restoration")
app.include_router(segmentation.router, prefix="/api/segmentation")

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

class Workflow(BaseModel):
    name: str
    nodes: list
    edges: list
    viewport: dict


# ---------------- SAVE WORKFLOW ----------------
@app.post("/api/workflows")
def save_workflow(workflow: Workflow):

    conn = get_connection()
    cur = conn.cursor()

    try:

        base_name = workflow.name

        cur.execute(
            """
            SELECT COALESCE(MAX(version),0)
            FROM workflows
            WHERE name LIKE %s
            """,
            (f"{base_name} v%",)
        )

        latest_version = cur.fetchone()[0]
        version = latest_version + 1

        versioned_name = f"{base_name} v{version}"

        zoom = workflow.viewport.get("zoom", 1)
        x = workflow.viewport.get("x", 0)
        y = workflow.viewport.get("y", 0)

        # ---------------- CLEAN NODES ----------------
        clean_nodes = []

        for node in workflow.nodes:

            clean_data = node.get("data", {}).copy()

            if "payload" in clean_data:

                payload = clean_data["payload"]
                minimal_payload = {}

                if isinstance(payload, dict):

                    allowed_keys = [
                        "result_image_url",
                        "output_image",
                        "mask_url"
                    ]

                    for k in allowed_keys:
                        if k in payload:
                            minimal_payload[k] = payload[k]

                if minimal_payload:
                    clean_data["payload"] = minimal_payload
                else:
                    clean_data.pop("payload", None)

            clean_nodes.append({
                "id": node.get("id"),
                "type": node.get("type"),
                "position": node.get("position"),
                "data": clean_data
            })

        # ---------------- CLEAN EDGES ----------------
        clean_edges = [
            {
                "id": e.get("id"),
                "source": e.get("source"),
                "target": e.get("target"),
                "type": e.get("type")
            }
            for e in workflow.edges
        ]

        snapshot = {
            "name": workflow.name,
            "nodes": clean_nodes,
            "edges": clean_edges,
            "viewport": workflow.viewport
        }

        # ---------------- INSERT WORKFLOW ----------------
        cur.execute(
            """
            INSERT INTO workflows
            (name, zoom, viewport_x, viewport_y, snapshot, version)
            VALUES (%s,%s,%s,%s,%s,%s)
            RETURNING id
            """,
            (
                versioned_name,
                zoom,
                x,
                y,
                Json(snapshot),
                version
            )
        )

        workflow_id = cur.fetchone()[0]

        # ---------------- INSERT NODES ----------------
        for node in clean_nodes:

            node_id = node.get("id")
            node_data = node.get("data", {})

            cur.execute(
                """
                INSERT INTO nodes
                (workflow_id,node_id,type,position_x,position_y,data)
                VALUES (%s,%s,%s,%s,%s,%s)
                """,
                (
                    workflow_id,
                    node_id,
                    node.get("type"),
                    node.get("position", {}).get("x", 0),
                    node.get("position", {}).get("y", 0),
                    Json(node_data)
                )
            )

            payload = node_data.get("payload", {})

            # INPUT IMAGE
            if "url" in payload:
                cur.execute(
                    """
                    INSERT INTO node_images
                    (workflow_id,version,node_id,image_url,image_role)
                    VALUES (%s,%s,%s,%s,%s)
                    """,
                    (
                        workflow_id,
                        version,
                        node_id,
                        payload["url"],
                        "input"
                    )
                )

            # RESULT IMAGE
            if "result_image_url" in payload:
                cur.execute(
                    """
                    INSERT INTO node_images
                    (workflow_id,version,node_id,image_url,image_role)
                    VALUES (%s,%s,%s,%s,%s)
                    """,
                    (
                        workflow_id,
                        version,
                        node_id,
                        payload["result_image_url"],
                        "result"
                    )
                )

            # MASK IMAGE
            if "mask_url" in payload:
                cur.execute(
                    """
                    INSERT INTO node_images
                    (workflow_id,version,node_id,image_url,image_role)
                    VALUES (%s,%s,%s,%s,%s)
                    """,
                    (
                        workflow_id,
                        version,
                        node_id,
                        payload["mask_url"],
                        "mask"
                    )
                )

        # ---------------- INSERT EDGES ----------------
        for edge in clean_edges:

            cur.execute(
                """
                INSERT INTO edges
                (workflow_id,source,target,type)
                VALUES (%s,%s,%s,%s)
                """,
                (
                    workflow_id,
                    edge.get("source"),
                    edge.get("target"),
                    edge.get("type")
                )
            )

        conn.commit()

        return {
            "message": "Workflow version saved",
            "id": workflow_id,
            "version": version,
            "name": versioned_name
        }

    except Exception as e:

        conn.rollback()
        return {"error": str(e)}

    finally:

        cur.close()
        conn.close()


# ---------------- GET WORKFLOW ----------------
@app.get("/api/workflows/{workflow_id}")
def get_workflow(workflow_id: int):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT name,zoom,viewport_x,viewport_y
        FROM workflows
        WHERE id=%s
        """,
        (workflow_id,)
    )

    wf = cur.fetchone()

    if not wf:
        return {"error": "Workflow not found"}

    cur.execute(
        """
        SELECT node_id,type,position_x,position_y,data
        FROM nodes
        WHERE workflow_id=%s
        """,
        (workflow_id,)
    )

    nodes_rows = cur.fetchall()

    cur.execute(
        """
        SELECT source,target,type
        FROM edges
        WHERE workflow_id=%s
        """,
        (workflow_id,)
    )

    edges_rows = cur.fetchall()

    cur.close()
    conn.close()

    nodes = [
        {
            "id": n[0],
            "type": n[1],
            "position": {"x": n[2], "y": n[3]},
            "data": n[4]
        }
        for n in nodes_rows
    ]

    edges = [
        {
            "source": e[0],
            "target": e[1],
            "type": e[2]
        }
        for e in edges_rows
    ]

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


@app.get("/api/workflows/{workflow_id}/node-images")
def get_node_images(workflow_id: int):

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT node_id,image_url,image_role,created_at
        FROM node_images
        WHERE workflow_id=%s
        ORDER BY created_at
        """,
        (workflow_id,)
    )

    rows = cur.fetchall()

    cur.close()
    conn.close()

    return [
        {
            "node_id": r[0],
            "image_url": r[1],
            "role": r[2],
            "created_at": r[3]
        }
        for r in rows
    ]