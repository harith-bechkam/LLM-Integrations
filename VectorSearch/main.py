import os
from fastapi import FastAPI, UploadFile, File, Query
from db import connect_milvus
from utils import extract_embeddings_with_chunks, chunk_text_for_summary
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager
import time
from pymilvus import utility
from fastapi import HTTPException
from pymilvus import Collection
from fastapi import Body
from fastapi.middleware.cors import CORSMiddleware
import openai
from transformers import pipeline
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm
import requests

clip_model = SentenceTransformer("clip-ViT-B-32")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# openai.api_key = ""
# summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")  # facebook/bart-large-cnn
summarizer = None

collection_main = None
collection_chunks = None

UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def download_with_progress(repo_id, filename_list, cache_dir):
    print(f"Downloading model {repo_id}...")
    for fname in filename_list:
        # Get file size from repo metadata
        url = f"https://huggingface.co/{repo_id}/resolve/main/{fname}"
        r = requests.head(url)
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=fname) as pbar:
            # Download file in chunks
            response = requests.get(url, stream=True)
            file_path = os.path.join(cache_dir, fname)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    print("All files downloaded!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global collection_main, collection_chunks, summarizer, clip_model

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting application...")

    # Milvus connection
    start_milvus = time.time()
    collection_main, collection_chunks = connect_milvus()
    end_milvus = time.time()
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Milvus collections loaded in {end_milvus - start_milvus:.2f}s"
    )

    # Load summarizer model
    model_name = "facebook/bart-large-cnn"
    cache_path = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}"
    )

    if os.path.exists(cache_path):
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model found in cache: {cache_path}"
        )
    else:
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model not in cache. Downloading..."
        )
        # Example: specify the main files in the repo (config, model, tokenizer)
        files_to_download = ["config.json", "pytorch_model.bin", "tokenizer.json"]
        download_with_progress(model_name, files_to_download, cache_path)

    # Initialize pipeline
    start_model = time.time()
    summarizer = pipeline("summarization", model=model_name, device=-1)  # CPU
    end_model = time.time()

    # Calculate total size
    total_bytes = sum(
        os.path.getsize(os.path.join(dirpath, f))
        for dirpath, _, files in os.walk(cache_path)
        for f in files
    )
    total_size_gb = total_bytes / (1024**3)

    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Summarizer model loaded in {end_model - start_model:.2f}s"
    )
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Model cache size: {total_size_gb:.2f} GB"
    )

    yield
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] App shutting down.")


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # <-- Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


@app.get("/")
def root():
    return {"message": "Hello! Milvus connection is ready."}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    # Save file to disk
    with open(file_path, "wb") as f:
        file_bytes = await file.read()
        f.write(file_bytes)
    print(f"Saved file: {file.filename} ({len(file_bytes)} bytes) at {file_path}")

    # Extract embeddings
    full_emb, full_text, chunks = extract_embeddings_with_chunks(file_path)
    print(f"Extracted embedding: full vector dim={len(full_emb)}, chunks={len(chunks)}")

    # Generate unique file_id
    file_id = int(time.time() * 1000)
    print(f"Generated file_id: {file_id}")

    # Insert into main collection
    result_main = collection_main.insert(
        [[file_id], [file.filename], [full_emb], [full_text]]
    )
    collection_main.flush()
    print(f"Inserted into main collection: {result_main}")

    # Insert chunks referencing file_id
    if chunks:
        result_chunks = collection_chunks.insert(
            [
                [file_id] * len(chunks),
                [c["chunk_embedding"] for c in chunks],
                [c["chunk_content"] for c in chunks],
                [c["chunk_index"] for c in chunks],
            ]
        )
        collection_chunks.flush()
        print(f"Inserted {len(chunks)} chunks into chunk collection: {result_chunks}")
    else:
        print("No chunks extracted, skipping chunk insert")

    return {
        "file": file.filename,
        "status": "uploaded with chunk embeddings",
        "chunks_count": len(chunks),
    }


@app.get("/search")
async def search(query: str = Query(...), top_k_files: int = 3, top_k_chunks: int = 5):
    """
    Search across the main collection and chunk collection.
    Returns summary input for LLM and also full matched file details.
    """
    # Convert query to embedding
    query_emb = clip_model.encode(query, convert_to_numpy=True)

    # --- Search main collection (for top matching files) ---
    file_results = collection_main.search(
        data=[query_emb],
        anns_field="full_embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k_files,
        output_fields=["file_name", "full_content"],
    )

    summary_input = f"Query: {query}\n\n"
    matched_files = []  # store full file info here

    for file_hit in file_results[0]:
        file_id = file_hit.id
        file_name = file_hit.entity.get("file_name")
        file_content = file_hit.entity.get("full_content")

        # Prepare base file data
        file_data = {
            "file_id": file_id,
            "file_name": file_name,
            "file_content": file_content,
            "relevant_chunks": [],
        }

        # --- Search within chunks for this file ---
        chunk_results = collection_chunks.search(
            data=[query_emb],
            anns_field="chunk_embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k_chunks,
            expr=f"file_id == {file_id}",
            output_fields=["chunk_content", "chunk_index"],
        )

        for chunk_hit in chunk_results[0]:
            chunk_index = chunk_hit.entity.get("chunk_index")
            chunk_text = chunk_hit.entity.get("chunk_content")
            file_data["relevant_chunks"].append(
                {"chunk_index": chunk_index, "chunk_content": chunk_text}
            )
            summary_input += (
                f"File: {file_name} | Chunk {chunk_index}: {chunk_text[:300]}...\n"
            )

        matched_files.append(file_data)

    return {
        "query": query,
        "summary_input_for_llm": summary_input.strip(),
        "matched_files": matched_files,
    }


@app.get("/documents/details")
async def get_documents_details():
    try:
        # Fetch all document IDs
        doc_ids = [
            doc["id"]
            for doc in collection_main.query(expr="id >= 0", output_fields=["id"])
        ]

        document_details = []

        for doc_id in doc_ids:
            # Fetch document details
            docs = collection_main.query(
                expr=f"id == {doc_id}",
                output_fields=["id", "file_name", "full_content"],
            )
            if not docs:
                continue  # Skip if no document found

            doc = docs[0]  # Assuming there's only one document per ID

            # Fetch associated chunks
            chunks = collection_chunks.query(
                expr=f"file_id == {doc_id}",
                output_fields=["id", "file_id", "chunk_index", "chunk_content"],
            )

            document_details.append(
                {
                    "id": doc["id"],
                    "file_name": doc["file_name"],
                    "full_content": doc["full_content"],
                    # "creation_time": doc.get("creation_time", "N/A"),
                    "chunks_count": len(chunks),
                    "chunks": [
                        {
                            "id": chunk["id"],
                            "file_id": chunk["file_id"],
                            "chunk_index": chunk["chunk_index"],
                            "chunk_content": chunk["chunk_content"],
                            # "creation_time": chunk.get("creation_time", "N/A")
                        }
                        for chunk in chunks
                    ],
                }
            )

        return {"documents": document_details}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/delete")
async def delete_document(file_id: int = Body(..., embed=True)):
    try:
        # Delete associated chunks first
        delete_chunks_result = collection_chunks.delete(expr=f"file_id == {file_id}")
        collection_chunks.flush()

        # Delete document from main collection
        delete_main_result = collection_main.delete(expr=f"id == {file_id}")
        collection_main.flush()

        # Access the delete_count from the MutationResult
        # deleted_count = delete_main_result.delete_count

        return {
            "status": "success",
            "message": f"Deleted document {file_id} and its chunks.",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summary")
async def generate_summary(payload: dict = Body(...)):
    summary_input = payload.get("summary_input_for_llm")
    if not summary_input:
        raise HTTPException(status_code=400, detail="summary_input_for_llm is required")

    try:
        text_chunks = chunk_text_for_summary(summary_input)
        summaries = []
        for chunk in text_chunks:
            result = summarizer(chunk, max_length=200, min_length=50, do_sample=False)
            summaries.append(result[0]["summary_text"])

        combined_summary = " ".join(summaries)
        if len(combined_summary.split()) > 300:
            final_summary = summarizer(
                combined_summary, max_length=200, min_length=50, do_sample=False
            )[0]["summary_text"]
        else:
            final_summary = combined_summary

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

    return {"summary": final_summary, "raw_input": summary_input}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
