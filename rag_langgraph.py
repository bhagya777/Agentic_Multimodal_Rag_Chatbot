"""
RAG Pipeline — LangGraph + LangSmith
=====================================
Set env vars before running:
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=<your-key>
    LANGCHAIN_PROJECT=rag-indexer          # shows in LangSmith UI

Two compiled graphs exported:
    indexing_graph  — processes all PDFs in PDF_DIR
    query_graph     — agentic RAG over indexed content

Usage:
    # Index
    indexing_graph.invoke({})

    # Query
    result = query_graph.invoke({"question": "What is hallucination in LLMs?"})
    print(result["answer"])
"""
# =============================================================================
# LANGSMITH CONFIGURATION  — must be before all LangChain imports
# =============================================================================
import os
from dotenv import load_dotenv
load_dotenv()                       # loads LANGCHAIN_TRACING_V2, API_KEY, PROJECT from .env

os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

import re, time, base64, pickle, hashlib, shutil, gc, traceback, threading, subprocess, requests, asyncio, sys
from io import BytesIO
from typing import TypedDict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from pypdf import PdfReader, PdfWriter

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_classic.storage import LocalFileStore, EncoderBackedStore
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.auto import partition as partition_auto
from flashrank import Ranker, RerankRequest

from langgraph.graph import StateGraph, END
from langsmith import traceable
import logging
import psutil

# Silence noisy HTTP client logs
logging.getLogger("httpx").setLevel(logging.WARNING)

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR    = r"H:\RAG_project"
DOC_DIR     = os.path.join(BASE_DIR, "data", "research_papers")
IMG_DIR     = os.path.join(BASE_DIR, "data", "paper_images")
TMP_DIR     = os.path.join(BASE_DIR, "data", "_tmp_pages")
TMP_IMG_DIR = os.path.join(BASE_DIR, "data", "_tmp_images")
DOC_LOG     = os.path.join(BASE_DIR, "indexed_pdfs.txt")  # keeps same file for backward compat

# Supported document formats
SUPPORTED_EXTENSIONS = (
    ".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".csv",
    ".html", ".htm", ".md", ".txt", ".rtf", ".odt", ".epub",
    ".png", ".jpg", ".jpeg",
)
ELEM_LOG    = os.path.join(BASE_DIR, "indexed_elements.txt")
FILE_LOG    = os.path.join(BASE_DIR, "indexing_progress_log.txt")
CHROMA_DIR  = os.path.join(BASE_DIR, "chroma_db")
STORE_DIR   = os.path.join(BASE_DIR, "doc_store")

MAX_VISION_WORKERS    = 1
CHUNK_SIZE            = 500
MAX_CHUNK_CHARS       = 4000
PDF_IMAGE_DPI         = 100
EMBED_CTX             = 8192
MIN_IMAGE_BYTES       = 2000
VISION_MAX_DIM        = 768
VISION_CTX            = 4096
VISION_COOLDOWN       = 10
VISION_THREAD_TIMEOUT = 600
VISION_FUTURE_TIMEOUT = 600
VISION_OOM_MAX_HITS   = 3

QUERY_MODEL           = "llama3.2"      # general chat model for RAG query graph
RETRIEVAL_K           = 15              # docs to retrieve per query

for d in [CHROMA_DIR, STORE_DIR, IMG_DIR, TMP_DIR, TMP_IMG_DIR]:
    os.makedirs(d, exist_ok=True)


# =============================================================================
# LOGGING
# =============================================================================
def write_log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    full_msg  = f"[{timestamp}] {msg}"
    print(full_msg)
    try:
        with open(FILE_LOG, "a", encoding="utf-8") as f:
            f.write(full_msg + "\n")
    except Exception:
        pass


# =============================================================================
# TESSERACT
# =============================================================================
TESSERACT_DIR = r"C:\Program Files\Tesseract-OCR"
TESSDATA_DIR  = os.path.join(TESSERACT_DIR, "tessdata")
if os.path.exists(TESSERACT_DIR):
    os.environ["PATH"]            = TESSERACT_DIR + os.pathsep + os.environ.get("PATH", "")
    os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR


# =============================================================================
# MODELS
# =============================================================================
write_log("Initializing models...")

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text",
    num_ctx=EMBED_CTX,
    keep_alive=120          # 2 min — auto-unload to free RAM for vision
)

vision_model = ChatOllama(
    model="llama3.2-vision",
    num_ctx=VISION_CTX,
    timeout=VISION_THREAD_TIMEOUT,
    keep_alive="60s",       # short keep_alive — free RAM after use
    num_predict=1024
)

query_llm = ChatOllama(
    model=QUERY_MODEL,
    num_ctx=4096,
    keep_alive=300,
    num_predict=2048,
    temperature=0.0
)


# =============================================================================
# STORAGE
# =============================================================================
vectorstore = Chroma(
    collection_name="research_multimodal_pro",
    embedding_function=embedding_model,
    persist_directory=CHROMA_DIR
)

local_fs = LocalFileStore(STORE_DIR)
store    = EncoderBackedStore(
    store=local_fs,
    key_encoder=lambda k: k,
    value_serializer=lambda v: pickle.dumps(v),
    value_deserializer=lambda v: pickle.loads(v)
)

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key="doc_id",
)
retriever.search_kwargs = {"k": 25}

ranker = Ranker()


# =============================================================================
# OLLAMA HELPERS
# =============================================================================
def _unload_model(model_name: str):
    try:
        requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "keep_alive": 0},
            timeout=10,
        )
        time.sleep(1)
    except Exception:
        pass


def _unload_embed_model():
    _unload_model("nomic-embed-text")
    gc.collect()


def _release_vision_memory():
    _unload_model("llama3.2-vision")


def _unload_all_models():
    try:
        r = requests.get("http://localhost:11434/api/ps", timeout=5)
        if r.status_code == 200:
            for m in r.json().get("models", []):
                _unload_model(m.get("name", ""))
        time.sleep(2)
    except Exception:
        pass


def restart_ollama():
    write_log("    [*] Restarting Ollama...")
    _unload_all_models()
    time.sleep(2)
    gc.collect()
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if 'ollama' in proc.info['name'].lower():
                proc.kill()
                proc.wait(timeout=10)
        except Exception:
            pass
    time.sleep(3)
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW
    )
    for _ in range(30):
        try:
            if requests.get("http://localhost:11434/api/tags", timeout=2).status_code == 200:
                write_log("    [*] Ollama restarted OK.")
                time.sleep(3)
                return True
        except Exception:
            pass
        time.sleep(1)
    write_log("    [!] Ollama restart failed.")
    return False


def _is_oom(exc: Exception) -> bool:
    return "more system memory" in str(exc).lower() or "insufficient memory" in str(exc).lower()


# =============================================================================
# PROGRESS TRACKING
# =============================================================================
def load_done_docs() -> set:
    if not os.path.exists(DOC_LOG):
        return set()
    return set(open(DOC_LOG).read().splitlines())


def mark_doc_done(name: str):
    with open(DOC_LOG, "a") as f:
        f.write(f"{name}\n")


def load_done_elements() -> set:
    if not os.path.exists(ELEM_LOG):
        return set()
    return set(open(ELEM_LOG).read().splitlines())


def mark_element_done(elem_id: str, done: set):
    done.add(elem_id)
    with open(ELEM_LOG, "a") as f:
        f.write(f"{elem_id}\n")


# =============================================================================
# SHARED UTILITIES (unchanged from original)
# =============================================================================
def get_paper_metadata(path, filename):
    try:
        reader = PdfReader(path)
        m      = reader.metadata
        title  = m.title  if m.title  and len(str(m.title))  > 5 else filename
        author = m.author if m.author else "Unknown"
        year   = "Unknown Year"
        try:
            if m.creation_date:
                year = str(m.creation_date.year)
        except Exception:
            pass
        return str(title).strip(), str(author).strip(), str(year), len(reader.pages)
    except Exception:
        return filename, "Unknown", "Unknown Year", 0


def clean_metadata(metadata: dict) -> dict:
    return {k: v for k, v in metadata.items()
            if v is not None and isinstance(v, (str, int, float, bool))}


def resize_and_encode_image(img_path: str) -> str:
    with Image.open(img_path) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        if max(img.size) > VISION_MAX_DIM:
            img.thumbnail((VISION_MAX_DIM, VISION_MAX_DIM), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def chunk_text(text: str) -> list:
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    return [c[:MAX_CHUNK_CHARS] for c in chunks if c.strip()]


def clear_tmp_img_dir():
    for fname in os.listdir(TMP_IMG_DIR):
        try:
            os.remove(os.path.join(TMP_IMG_DIR, fname))
        except Exception:
            pass


def move_images_to_permanent(pdf_img_dir: str, page_num: int) -> dict:
    mapping = {}
    for fname in os.listdir(TMP_IMG_DIR):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        src = os.path.join(TMP_IMG_DIR, fname)
        if os.path.getsize(src) < MIN_IMAGE_BYTES:
            os.remove(src)
            continue
        stable_name = f"p{page_num:04d}_{fname}"
        dst = os.path.join(pdf_img_dir, stable_name)
        shutil.move(src, dst)
        mapping[os.path.normpath(src)] = os.path.normpath(dst)
    return mapping


def remap_img_path(raw_path, mapping):
    if not raw_path:
        return raw_path
    return mapping.get(os.path.normpath(raw_path), os.path.normpath(raw_path))


def analyze_visual_call(v_id: str, img_path: str, meta: dict, oom_hits: int) -> tuple:
    """Returns (Document | None, new_oom_hits)"""
    if oom_hits >= VISION_OOM_MAX_HITS:
        write_log(f"    [!] Vision kill-switch active — skipping {os.path.basename(img_path)}")
        return None, oom_hits

    _unload_embed_model()

    msg = HumanMessage(content=[
        {
            "type": "text",
            "text": (
                "Analyze this figure or table from a research paper.\n"
                "Provide a short title first, then a technical description.\n"
                "Cover: all visible data, axis labels, legends, trends, "
                "numerical values, and the conclusion a reader would draw.\n\n"
                "Format strictly as:\n"
                "TITLE: <title>\n"
                "DESCRIPTION: <detailed description>"
            )
        },
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{resize_and_encode_image(img_path)}"}}
    ])

    for attempt in range(3):
        try:
            result_holder, error_holder = [None], [None]

            def invoke():
                try:
                    result_holder[0] = vision_model.invoke([msg]).content
                except Exception as e:
                    error_holder[0] = e

            t = threading.Thread(target=invoke, daemon=True)
            t.start()
            t.join(timeout=VISION_THREAD_TIMEOUT)

            if t.is_alive():
                raise TimeoutError("Vision thread hung")

            if error_holder[0]:
                raise error_holder[0]

            _release_vision_memory()

            response = result_holder[0]
            if "TITLE:" in response and "DESCRIPTION:" in response:
                v_title = response.split("TITLE:")[1].split("DESCRIPTION:")[0].strip()
                v_desc  = response.split("DESCRIPTION:")[1].strip()
            else:
                v_title, v_desc = "Technical Visual", response

            v_meta = {**meta, "doc_id": v_id, "type": "visual",
                      "image_path": img_path, "image_title": v_title,
                      "image_summary": v_desc[:MAX_CHUNK_CHARS]}
            return Document(page_content=v_desc[:MAX_CHUNK_CHARS], metadata=v_meta), 0  # reset hits

        except TimeoutError:
            raise

        except Exception as e:
            if _is_oom(e):
                oom_hits += 1
                write_log(f"    [!] OOM hit #{oom_hits} — restarting Ollama.")
                _release_vision_memory()
                restart_ollama()
                return None, oom_hits

            if attempt < 2:
                write_log(f"    [!] Vision attempt {attempt+1}: {e}. Retrying in 10s...")
                time.sleep(10)
            else:
                write_log(f"    [!] Visual skip: {e}")
                return None, oom_hits

    return None, oom_hits


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████
#  INDEXING GRAPH
# ██████████████████████████████████████████████████████████████████████████
# =============================================================================

class IndexState(TypedDict):
    # Run-level
    all_docs:       List[str]          # all document filenames to process
    done_docs:      List[str]          # already indexed (from log)
    done_elements:  List[str]          # already indexed elements (from log)
    doc_index:      int                # current position in all_docs
    errors:         List[str]

    # Document-level (reset per document)
    doc_name:       str
    doc_path:       str
    doc_type:       str                # file extension (e.g. ".pdf", ".docx")
    title:          str
    author:         str
    year:           str
    total_pages:    int
    doc_img_dir:    str
    doc_stem:       str

    # Vision state
    vision_queue:   List[dict]         # [{v_id, img_path, meta}]
    oom_hits:       int

    # Accumulation
    batch_docs:     List[Any]
    batch_kv:       List[Any]


# ── Node 1: scan_documents ────────────────────────────────────────────────────
def scan_documents(state: IndexState) -> IndexState:
    write_log("[GRAPH] scan_documents")
    done_docs     = list(load_done_docs())
    done_elements = list(load_done_elements())
    all_docs      = sorted(
        f for f in os.listdir(DOC_DIR)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
        and f not in done_docs
    )
    write_log(f"[*] {len(all_docs)} documents to process. {len(done_elements)} elements already indexed.")
    return {
        **state,
        "all_docs":      all_docs,
        "done_docs":     done_docs,
        "done_elements": done_elements,
        "doc_index":     0,
        "errors":        [],
        "oom_hits":      0,
    }


# ── Node 2: load_document ─────────────────────────────────────────────────────
def load_document(state: IndexState) -> IndexState:
    idx      = state["doc_index"]
    doc_name = state["all_docs"][idx]
    doc_path = os.path.join(DOC_DIR, doc_name)
    doc_stem = os.path.splitext(doc_name)[0]
    doc_ext  = os.path.splitext(doc_name)[1].lower()
    doc_img_dir = os.path.join(IMG_DIR, doc_stem)
    os.makedirs(doc_img_dir, exist_ok=True)

    # PDF-specific metadata; generic fallback for other formats
    if doc_ext == ".pdf":
        title, author, year, total_pages = get_paper_metadata(doc_path, doc_name)
    else:
        title, author, year, total_pages = doc_name, "Unknown", "Unknown Year", 0

    write_log(f"\n[GRAPH] load_document [{idx+1}/{len(state['all_docs'])}]: {title} ({doc_ext})")

    return {
        **state,
        "doc_name":    doc_name,
        "doc_path":    doc_path,
        "doc_type":    doc_ext,
        "doc_stem":    doc_stem,
        "doc_img_dir": doc_img_dir,
        "title":       title,
        "author":      author,
        "year":        year,
        "total_pages": total_pages,
        "vision_queue": [],
        "batch_docs":  [],
        "batch_kv":    [],
    }


# ── Node 3: index_document ────────────────────────────────────────────────────
def index_document(state: IndexState) -> IndexState:
    """Partition document content, process text, collect vision queue.
    Uses partition_pdf (page-by-page hi_res) for PDFs.
    Uses partition_auto (single-pass) for all other formats.
    """
    doc_type = state["doc_type"]
    write_log(f"[GRAPH] index_document: {state['title']} (type: {doc_type})")

    done_set     = set(state["done_elements"])
    batch_docs   = []
    batch_kv     = []
    vision_queue = []

    path        = state["doc_path"]
    doc_stem    = state["doc_stem"]
    doc_img_dir = state["doc_img_dir"]
    meta_base   = {
        "source":      state["doc_name"],
        "paper_title": state["title"],
        "author":      state["author"],
        "year":        state["year"],
        "total_pages": state["total_pages"],
    }

    def _commit(docs, kv):
        if not docs:
            return
        for attempt in range(3):
            try:
                vectorstore.add_documents(docs)
                store.mset(kv)
                for doc in docs:
                    mark_element_done(doc.metadata["doc_id"], done_set)
                write_log(f"    >>> Live Sync: {len(docs)} segments added.")
                break
            except Exception as e:
                err = str(e)
                if "input length exceeds" in err.lower():
                    for doc in docs:
                        doc.page_content = doc.page_content[:MAX_CHUNK_CHARS]
                    if attempt < 2:
                        continue
                elif attempt < 2:
                    write_log(f"    [!] Commit retry {attempt+1}: {e}")
                    time.sleep(5)
                else:
                    write_log(f"    [!] Commit failed: {e}")
        docs.clear()
        kv.clear()

    def _process_elements(elements, page_num, path_mapping):
        """Shared logic: process partitioned elements into text chunks + vision queue."""
        claimed_imgs = set()

        for el in elements:
            raw_text = str(el.text) if el.text else ""
            tmp_img  = getattr(el.metadata, "image_path", None)
            img_path = remap_img_path(tmp_img, path_mapping)
            doc_hash = hashlib.md5((raw_text + (img_path or "")).encode()).hexdigest()

            base_meta = {
                **meta_base,
                "doc_id":      doc_hash,
                "page_number": page_num,
                "category":    el.category,
                "hash":        doc_hash,
            }
            if hasattr(el, "metadata"):
                for k, v in clean_metadata(el.metadata.to_dict()).items():
                    if k not in base_meta:
                        base_meta[k] = v

            # Text
            if raw_text and len(raw_text) > 5:
                content = str(getattr(el.metadata, "text_as_html", None) or raw_text)
                content = content[:MAX_CHUNK_CHARS * 4]
                chunks  = chunk_text(content)
                for cidx, chunk in enumerate(chunks):
                    c_id = f"{doc_hash}_c{cidx}" if len(chunks) > 1 else doc_hash
                    if c_id in done_set:
                        continue
                    c_meta = {**base_meta, "doc_id": c_id}
                    if len(chunks) > 1:
                        c_meta["chunk"] = cidx
                    doc = Document(page_content=chunk, metadata=c_meta)
                    batch_docs.append(doc)
                    batch_kv.append((c_id, doc))
                if len(batch_docs) >= 10:
                    _commit(batch_docs, batch_kv)

            # Image
            if img_path and os.path.exists(img_path):
                claimed_imgs.add(img_path)
                if os.path.getsize(img_path) >= MIN_IMAGE_BYTES:
                    v_id = doc_hash + "_v"
                    if v_id not in done_set:
                        vision_queue.append({"v_id": v_id, "img_path": img_path, "meta": base_meta})

        # Orphan images
        for orphan in set(path_mapping.values()) - claimed_imgs:
            if not os.path.exists(orphan):
                continue
            o_hash = hashlib.md5(orphan.encode()).hexdigest()
            v_id   = o_hash + "_v"
            if v_id not in done_set:
                orphan_meta = {
                    **meta_base,
                    "doc_id":      v_id,
                    "page_number": page_num,
                    "category":    "OrphanImage",
                    "hash":        o_hash,
                }
                vision_queue.append({"v_id": v_id, "img_path": orphan, "meta": orphan_meta})

    # ── PDF path: page-by-page hi_res partitioning ──
    if doc_type == ".pdf":
        reader = PdfReader(path)
        total  = len(reader.pages)

        for i, page in enumerate(reader.pages):
            page_num = i + 1
            tmp_path = os.path.join(TMP_DIR, f"{doc_stem}_p{page_num}.pdf")

            clear_tmp_img_dir()
            writer = PdfWriter()
            writer.add_page(page)
            with open(tmp_path, "wb") as f:
                writer.write(f)

            write_log(f"  [page {page_num}/{total}] Partitioning...")
            elements    = []
            current_dpi = PDF_IMAGE_DPI

            try:
                while current_dpi >= 50:
                    try:
                        elements = partition_pdf(
                            filename=tmp_path,
                            strategy="hi_res",
                            hi_res_model_name="yolox_quantized",
                            pdf_image_dpi=current_dpi,
                            extract_image_block_types=["Image", "Table"],
                            extract_image_block_output_dir=TMP_IMG_DIR,
                        )
                        break
                    except Exception as e:
                        err_msg = str(e).lower()
                        if any(x in err_msg for x in ["memory", "0xe0000008", "pixcreate", "malloc"]):
                            current_dpi -= 25
                            if current_dpi < 50:
                                try:
                                    elements = partition_pdf(filename=tmp_path, strategy="fast")
                                except Exception as fe:
                                    write_log(f"  [!!] Total partition failure p{page_num}: {fe}")
                                    elements = []
                                break
                            clear_tmp_img_dir()
                            time.sleep(2)
                            gc.collect()
                        else:
                            write_log(f"  [!] Partition error p{page_num}: {e}")
                            break
            finally:
                path_mapping = move_images_to_permanent(doc_img_dir, page_num)
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                gc.collect()
                time.sleep(2)

            _process_elements(elements, page_num, path_mapping)

    # ── Non-PDF path: single-pass partition_auto ──
    else:
        write_log(f"  [auto] Partitioning {doc_type} document...")
        clear_tmp_img_dir()
        try:
            elements = partition_auto(
                filename=path,
                extract_image_block_types=["Image", "Table"],
                extract_image_block_output_dir=TMP_IMG_DIR,
            )
            write_log(f"  [auto] Extracted {len(elements)} elements.")
        except Exception as e:
            write_log(f"  [!] partition_auto failed: {e}")
            elements = []

        path_mapping = move_images_to_permanent(doc_img_dir, 1)
        _process_elements(elements, 1, path_mapping)

    _commit(batch_docs, batch_kv)

    return {
        **state,
        "done_elements": list(done_set),
        "vision_queue":  vision_queue,
        "batch_docs":    [],
        "batch_kv":      [],
    }


# ── Node 4: run_vision ────────────────────────────────────────────────────────
def run_vision(state: IndexState) -> IndexState:
    """Process all queued vision tasks sequentially."""
    write_log(f"[GRAPH] run_vision: {len(state['vision_queue'])} visuals queued")

    done_set  = set(state["done_elements"])
    oom_hits  = state["oom_hits"]
    errors    = list(state["errors"])

    with ThreadPoolExecutor(max_workers=MAX_VISION_WORKERS) as executor:
        futures = []
        for task in state["vision_queue"]:
            if oom_hits >= VISION_OOM_MAX_HITS:
                write_log("    [!] Vision kill-switch — skipping remaining visuals.")
                break
            future = executor.submit(
                analyze_visual_call,
                task["v_id"], task["img_path"], task["meta"], oom_hits
            )
            futures.append((future, task["v_id"]))

        for future, v_id in futures:
            try:
                v_doc, oom_hits = future.result(timeout=VISION_FUTURE_TIMEOUT)
                if v_doc:
                    v_doc.page_content = v_doc.page_content[:MAX_CHUNK_CHARS]
                    for attempt in range(3):
                        try:
                            vectorstore.add_documents([v_doc])
                            store.mset([(v_id, v_doc)])
                            mark_element_done(v_id, done_set)
                            write_log(f"    >>> Visual saved: [{v_doc.metadata.get('image_title')}]")
                            break
                        except Exception as ve:
                            if attempt < 2:
                                time.sleep(5)
                            else:
                                write_log(f"    [!] Vision embed failed: {ve}")
            except TimeoutError:
                write_log(f"    [!] Vision OUTER TIMEOUT on {v_id} — restarting Ollama.")
                future.cancel()
                restart_ollama()
            except Exception as e:
                write_log(f"    [!] Vision future error: {e}")
                if _is_oom(e):
                    oom_hits += 1
                restart_ollama()

            time.sleep(VISION_COOLDOWN)

    return {**state, "done_elements": list(done_set), "oom_hits": oom_hits, "errors": errors}


# ── Node 5: finalize_document ─────────────────────────────────────────────────
def finalize_document(state: IndexState) -> IndexState:
    write_log(f"[GRAPH] finalize_document: {state['title']}")
    mark_doc_done(state["doc_name"])
    gc.collect()
    return {
        **state,
        "doc_index": state["doc_index"] + 1,
    }


# ── Conditional edge: more documents? ─────────────────────────────────────────
def route_next_document(state: IndexState) -> str:
    if state["doc_index"] < len(state["all_docs"]):
        return "load_document"
    return END

def route_after_scan(state: IndexState) -> str:
    if len(state["all_docs"]) == 0:
        write_log("[*] No new documents to index.")
        return END
    return "load_document"


# ── Build Indexing Graph ──────────────────────────────────────────────────────
def build_indexing_graph():
    g = StateGraph(IndexState)
    g.add_node("scan_documents",    scan_documents)
    g.add_node("load_document",     load_document)
    g.add_node("index_document",    index_document)
    g.add_node("run_vision",        run_vision)
    g.add_node("finalize_document", finalize_document)

    g.set_entry_point("scan_documents")
    g.add_conditional_edges("scan_documents", route_after_scan, {"load_document": "load_document", END: END})
    g.add_edge("load_document",     "index_document")
    g.add_edge("index_document",    "run_vision")
    g.add_edge("run_vision",        "finalize_document")
    g.add_conditional_edges("finalize_document", route_next_document, {"load_document": "load_document", END: END})

    return g.compile()


indexing_graph = build_indexing_graph()


# =============================================================================
# ██████████████████████████████████████████████████████████████████████████
#  QUERY GRAPH  (Agentic RAG — with Query Expansion + Reranking)
# ██████████████████████████████████████████████████████████████████████████
# =============================================================================

from typing import Annotated, List, Any
from typing_extensions import TypedDict
import operator

class QueryState(TypedDict):
    question:         str
    original_question: str              # preserved original user question
    retrieved_docs:   List[Any]
    relevant_docs: Annotated[List[Any], operator.add]
    answer:           str
    grounded:         bool
    retry_count:      int
    sources_footer:   str
    image_paths:      List[str]
    # ── Agentic RCA fields ──
    failure_reason:   str               # "retrieval_fail" | "generation_fail" | "unanswerable" | ""
    failure_critique: str               # explanation of what went wrong
    rewritten_query:  str               # refined query for re-retrieval
    rca_history:      List[str]         # log of recovery actions taken


GENERATE_PROMPT = """You are a helpful, intelligent Research Assistant. Your goal is to analyze the provided DATA SEGMENTS and answer the USER QUESTION naturally and conversationally.

CRITICAL RULES (MUST FOLLOW):
1. STRICT GROUNDING: You MUST ONLY use the information provided in the DATA SEGMENTS below.
2. ABSOLUTE ZERO EXTERNAL KNOWLEDGE: You are FORBIDDEN from using your own internal memory or 'publicly available information' to answer. If the segments do not contain the answer, you must stop answering. Do NOT invent references or hallucinate facts.
3. DISCLOSURE: If the answer is not in the segments, politely inform the user that the provided papers don't contain the exact answer, and DO NOT guess or infer.
4. CITATION: Use inline citations like [Segment X] when stating facts from the papers.
5. IMAGES: If the DATA SEGMENTS contain descriptions of charts, figures, or diagrams, explicitly mention them in your answer.
6. STRUCTURE: Synthesize the information naturally, but prioritize strict factual accuracy over conversational flow.

DATA SEGMENTS:
{context}

USER QUESTION:
{question}

STRICT RESEARCH REPORT:"""

GROUNDING_PROMPT = """You are a hallucination checker for a RAG research assistant.

Given an ANSWER and the SOURCE DOCUMENTS, determine if the answer is grounded.

Reply with ONLY one word: GROUNDED or HALLUCINATED.

GROUNDED means:
- The answer's claims are supported by or can be reasonably inferred from the documents.
- Paraphrasing, summarizing, or synthesizing information from the documents is GROUNDED.
- If the answer says the information is not available, that is GROUNDED.

HALLUCINATED means:
- The answer introduces specific facts, numbers, model names, or results that are NOT found anywhere in the documents.
- The answer references 'publicly available information' or uses external knowledge.
- The answer invents citations or attributes claims to papers not in the documents.

IMPORTANT: Do NOT flag an answer as HALLUCINATED just because it paraphrases or combines information from multiple segments. Only flag it if it introduces genuinely NEW claims not traceable to any document.

Documents:
{context}

Answer: {answer}"""

RCA_PROMPT = """You are a strict diagnostic agent for a RAG pipeline.

The following ANSWER was generated from the DOCUMENTS below, but was flagged as a HALLUCINATION.
Your job is to diagnose WHY the hallucination occurred.

DOCUMENTS:
{context}

ANSWER:
{answer}

QUESTION:
{question}

Analyze carefully and respond with EXACTLY this JSON format (no extra text):
{{"reason": "<retrieval_fail | generation_fail | unanswerable>", "explanation": "<1-2 sentence explanation of the root cause>", "suggested_query": "<if reason is retrieval_fail, write a better search query targeting the missing information; otherwise empty string>"}}

CLASSIFICATION GUIDE:
- "retrieval_fail": The documents are off-topic or missing the specific info needed to answer. The LLM had to guess because it had nothing relevant.
- "generation_fail": The documents DO contain relevant information, but the LLM added facts, numbers, or claims NOT found in the text.
- "unanswerable": The question is too vague or completely out of scope for these research papers."""


# ── Node 1: retrieve (query expansion + multi-vector + reranking) ─────────────

async def retrieve(state: QueryState) -> QueryState:
    question = state["question"]
    write_log(f"[GRAPH] retrieve: {question[:80]}")

    # 1. Query expansion — generate search variations
    expansion_prompt = (
        f"You are an AI assistant helping with research. "
        f"Generate 3 technical search variations for: {question}. "
        f"CRITICAL: At least ONE variation MUST explicitly ask for 'a diagram, chart, figure, or table' related to the topic. "
        f"Return only questions, one per line."
    )
    try:
        expansion = (await query_llm.ainvoke(expansion_prompt)).content
        expanded = [q.strip() for q in expansion.split('\n') if q.strip()]
    except Exception as e:
        write_log(f"  [!] Query expansion failed: {e}")
        expanded = []
    all_queries = [question] + expanded[:3]
    write_log(f"  Expanded to {len(all_queries)} queries.")

    # 2. Multi-vector retrieval with deduplication
    all_docs = []
    seen = set()
    for q in all_queries:
        docs = await retriever.ainvoke(q)
        for doc in docs:
            content_str = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            if content_str not in seen:
                all_docs.append(doc)
                seen.add(content_str)
    write_log(f"  Retrieved {len(all_docs)} unique candidates.")

    # 3. FlashRank reranking
    passages = []
    for i, doc in enumerate(all_docs):
        passages.append({
            "id": i,
            "text": doc.page_content if hasattr(doc, 'page_content') else str(doc),
            "meta": doc.metadata
        })
    rerank_request = RerankRequest(query=question, passages=passages)
    reranked = await asyncio.to_thread(ranker.rerank, rerank_request)
    top = reranked[:10]

    # 4. Rebuild Document objects from reranked results
    top_docs = []
    for res in top:
        top_docs.append(Document(
            page_content=res["text"],
            metadata=res["meta"]
        ))

    for i, res in enumerate(top):
        meta = res["meta"]
        title = meta.get("paper_title", "Unknown")
        year = meta.get("year", "Unknown")
        if year in ["Unknown Year", "Unknown"]:
            fname = meta.get("source", "")
            m = re.search(r'(\d{2})(\d{2})\.', fname)
            if m: year = "20" + m.group(1)
        page  = doc.metadata.get('page_number', 'Unknown')
        write_log(f"  -> [{i+1}] {title} ({year}) - Page {page}")

    write_log(f"  Top {len(top_docs)} docs after reranking.")
    return {**state, "retrieved_docs": top_docs}


# ── Node 2: grade_docs (LLM-based relevance filter) ──────────────────────────
async def grade_single_doc(state: dict):
    """Grades a single document's relevance."""
    doc = state["doc"]
    question = state["question"]
    
    prompt = (
        f"Relevance Grader: Does this content relate to the question? "
        f"Reply only 'Y' or 'N'.\n"
        f"Q: {question}\n"
        f"Doc: {doc.page_content[:600]}" # Truncated to save CPU tokens
    )
    
    response = (await query_llm.ainvoke([HumanMessage(content=prompt)])).content.strip().upper()
    
    if "Y" in response:
        return {"relevant_docs": [doc]}
    return {"relevant_docs": []}

from langgraph.constants import Send

def start_grading_fan_out(state: QueryState):
    """Dispatches parallel grading tasks."""
    return [
        Send("grade_single_doc", {"doc": d, "question": state["question"]}) 
        for d in state["retrieved_docs"]
    ]


# ── Node 3: generate ──────────────────────────────────────────────────────────
async def generate(state: QueryState) -> QueryState:
    write_log("[GRAPH] generate")
    docs = state["relevant_docs"] or state["retrieved_docs"]

    # Build rich context with segment IDs, sources, and keywords
    context_text = ""
    source_lines = []
    image_paths_to_open = []
    for i, doc in enumerate(docs):
        meta = doc.metadata
        title = meta.get("paper_title", "Unknown")
        year = meta.get("year", "Unknown")
        if year in ["Unknown Year", "Unknown"]:
            fname = meta.get("source", "")
            m = re.search(r'(\d{2})(\d{2})\.', fname)
            if m: year = "20" + m.group(1)
        page = str(meta.get("page_number", "?"))
        keywords = meta.get("keywords", "N/A")

        context_text += f"\n--- DATA SEGMENT {i+1} ---\n"
        context_text += f"SOURCE: {title} ({year}), Page {page}\n"
        context_text += f"KEYWORDS: {keywords}\n"
        context_text += f"CONTENT: {doc.page_content}\n"

        stype = meta.get('type', 'TEXT').upper()
        if stype in ["IMAGE", "VISUAL"] and 'image_path' in meta:
            image_paths_to_open.append(meta['image_path'])

        source_lines.append(f"Segment {i+1}: {title} (Pg {page})")

    prompt = GENERATE_PROMPT.format(context=context_text[:8000], question=state["question"])

    # ── Agentic: inject critique feedback on stricter retry ──
    critique = state.get("failure_critique", "")
    if critique and state.get("failure_reason") == "generation_fail":
        critique_addendum = (
            f"\n\n⚠️ PREVIOUS ATTEMPT WAS FLAGGED AS HALLUCINATION.\n"
            f"SPECIFIC ERROR: {critique}\n"
            f"You MUST NOT repeat this error. Use ONLY information from the "
            f"DATA SEGMENTS above. If you cannot find the answer, say so explicitly."
        )
        prompt += critique_addendum
        write_log(f"  [Agentic] Injected critique feedback into prompt.")

    answer = (await query_llm.ainvoke([HumanMessage(content=prompt)])).content.strip()

    footer = "\n\n" + "="*20 + " VERIFIED SOURCES " + "="*20 + "\n"
    footer += "\n".join(source_lines)

    write_log(f"  Generated answer ({len(answer)} chars).")
    return {**state, "answer": answer, "sources_footer": footer, "image_paths": image_paths_to_open}


# ── Node 4: check_grounding ───────────────────────────────────────────────────
async def check_grounding(state: QueryState) -> QueryState:
    write_log("[GRAPH] check_grounding")
    docs = state["relevant_docs"] or state["retrieved_docs"]
    context = "\n\n".join(d.page_content[:800] for d in docs)
    prompt  = GROUNDING_PROMPT.format(context=context[:6000], answer=state["answer"])
    verdict = (await query_llm.ainvoke([HumanMessage(content=prompt)])).content.strip().upper()
    grounded = "HALLUCINATED" not in verdict
    write_log(f"  Grounding verdict: {'GROUNDED' if grounded else 'HALLUCINATED'}")
    return {**state, "grounded": grounded, "retry_count": state.get("retry_count", 0) + 1}


# ── Conditional edge: route to RCA on hallucination ──────────────────────────
def route_grounding(state: QueryState) -> str:
    if state["grounded"]:
        return END
    if state.get("retry_count", 0) >= 3:
        write_log("  [!] Hard retry limit reached (3/3) — ending.")
        return END
    write_log("  [!] Hallucination detected — routing to Root Cause Analysis.")
    return "analyze_failure"


# ── Node 5: analyze_failure — Root Cause Analysis ─────────────────────────────
async def analyze_failure(state: QueryState) -> QueryState:
    """Diagnose WHY a hallucination occurred and classify the failure mode."""
    write_log("[GRAPH] analyze_failure (Root Cause Analysis)")

    docs = state["relevant_docs"] or state["retrieved_docs"]
    context = "\n\n".join(d.page_content[:600] for d in docs)
    rca_history = list(state.get("rca_history", []))

    # Safety: if we already tried re-retrieval, don't allow retrieval_fail again
    already_re_retrieved = "retrieval_fail" in rca_history

    prompt = RCA_PROMPT.format(
        context=context[:5000],
        answer=state["answer"][:2000],
        question=state.get("original_question", state["question"]),
    )

    if already_re_retrieved:
        prompt += (
            "\n\nIMPORTANT: A re-retrieval was ALREADY attempted. "
            "Do NOT classify as 'retrieval_fail' again. "
            "Choose 'generation_fail' or 'unanswerable' only."
        )

    try:
        response = (await query_llm.ainvoke([HumanMessage(content=prompt)])).content.strip()
        write_log(f"  RCA raw response: {response[:200]}")

        # Parse JSON from response (handle LLM wrapping it in markdown etc.)
        import json as _json
        json_str = response
        if "```" in json_str:
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
        # Try to find JSON object in the response
        start_idx = json_str.find("{")
        end_idx = json_str.rfind("}") + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = json_str[start_idx:end_idx]

        rca = _json.loads(json_str)
        reason = rca.get("reason", "generation_fail")
        explanation = rca.get("explanation", "Unknown failure")
        suggested_query = rca.get("suggested_query", "")

        # Validate reason
        valid_reasons = {"retrieval_fail", "generation_fail", "unanswerable"}
        if reason not in valid_reasons:
            reason = "generation_fail"

        # Enforce: no double re-retrieval
        if reason == "retrieval_fail" and already_re_retrieved:
            reason = "unanswerable"
            explanation = "Re-retrieval already attempted. The indexed papers likely don't cover this topic."
            suggested_query = ""

    except Exception as e:
        write_log(f"  [!] RCA parsing error: {e}. Defaulting to generation_fail.")
        reason = "generation_fail"
        explanation = "Could not parse RCA response; defaulting to stricter re-generation."
        suggested_query = ""

    rca_history.append(reason)
    write_log(f"  RCA verdict: {reason} — {explanation}")

    return {
        **state,
        "failure_reason":   reason,
        "failure_critique":  explanation,
        "rewritten_query":  suggested_query,
        "rca_history":      rca_history,
    }


# ── Node 6: rewrite_query — Query Reformulation ──────────────────────────────
async def rewrite_query(state: QueryState) -> QueryState:
    """Rewrite the search query based on RCA feedback before re-retrieval."""
    suggested = state.get("rewritten_query", "").strip()
    original  = state.get("original_question", state["question"])

    if suggested:
        new_query = suggested
    else:
        # Fallback: ask the LLM to reformulate
        rewrite_prompt = (
            f"The following research question did not return useful results: "
            f"\"{original}\"\n"
            f"Rewrite it as a more specific, keyword-rich search query. "
            f"Return ONLY the rewritten query, nothing else."
        )
        new_query = (await query_llm.ainvoke([HumanMessage(content=rewrite_prompt)])).content.strip()

    write_log(f"  [Agentic] Rewritten query: {new_query[:100]}")
    # Reset retrieval state for fresh search
    return {
        **state,
        "question":       new_query,
        "retrieved_docs": [],
        "relevant_docs":  [],
    }


# ── Router: RCA decision branching ───────────────────────────────────────────
def route_rca_decision(state: QueryState) -> str:
    reason = state.get("failure_reason", "")
    if reason == "retrieval_fail":
        write_log("  [Agentic] Decision: Re-retrieve with rewritten query.")
        return "rewrite_query"
    elif reason == "generation_fail":
        write_log("  [Agentic] Decision: Re-generate with critique feedback.")
        return "generate"
    else:  # "unanswerable"
        write_log("  [Agentic] Decision: Question unanswerable from indexed papers. Ending.")
        return END


# ── Build Query Graph (Agentic Self-Corrective RAG) ──────────────────────────
def build_query_graph():
    g = StateGraph(QueryState)
    g.add_node("retrieve",         retrieve)
    g.add_node("grade_single_doc", grade_single_doc)
    g.add_node("generate",         generate)
    g.add_node("check_grounding",  check_grounding)
    g.add_node("analyze_failure",  analyze_failure)     # Agentic RCA
    g.add_node("rewrite_query",    rewrite_query)       # Query reformulation

    g.set_entry_point("retrieve")

    # Parallel fan-out for document grading
    g.add_conditional_edges("retrieve", start_grading_fan_out, ["grade_single_doc"])

    # Fan-in (wait for all graders to finish)
    g.add_edge("grade_single_doc", "generate")

    g.add_edge("generate", "check_grounding")

    # Agentic routing: hallucination → RCA → targeted recovery
    g.add_conditional_edges("check_grounding", route_grounding,
                            {"analyze_failure": "analyze_failure", END: END})
    g.add_conditional_edges("analyze_failure", route_rca_decision,
                            {"rewrite_query": "rewrite_query", "generate": "generate", END: END})
    g.add_edge("rewrite_query", "retrieve")

    return g.compile()


# Important: Limit concurrency for 16GB RAM stability
query_graph = build_query_graph()
# When calling, use: query_graph.invoke(inputs, config={"max_concurrency": 2})


# =============================================================================
# ENTRYPOINTS
# =============================================================================
if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    if len(sys.argv) > 1 and sys.argv[1] == "query":
        # Usage: python rag_langgraph.py query "your question here"
        question = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "What is hallucination in LLMs?"
        result   = asyncio.run(query_graph.ainvoke({
            "question":       question,
            "original_question": question,
            "retrieved_docs": [],
            "relevant_docs":  [],
            "answer":         "",
            "grounded":       False,
            "retry_count":    0,
            "sources_footer":  "",
            "image_paths":    [],
            "failure_reason":  "",
            "failure_critique": "",
            "rewritten_query": "",
            "rca_history":    [],
        }, config={"run_name": f"query: {question[:50]}","max_concurrency": 2}))
        print("\n" + "="*60)
        print("🤖 ANSWER:")
        print("="*60)
        print(result["answer"])
        print(result.get("sources_footer", ""))
        print(f"\n✅ GROUNDED: {result['grounded']}")

        for img_p in list(set(result.get("image_paths", []))):
            if os.path.exists(img_p):
                os.startfile(img_p)
    else:
        # Default: run indexing
        indexing_graph.invoke({
            "all_docs":      [],
            "done_docs":     [],
            "done_elements": [],
            "doc_index":     0,
            "errors":        [],
            "doc_name":      "",
            "doc_path":      "",
            "doc_type":      "",
            "doc_stem":      "",
            "doc_img_dir":   "",
            "title":         "",
            "author":        "",
            "year":          "",
            "total_pages":   0,
            "vision_queue":  [],
            "oom_hits":      0,
            "batch_docs":    [],
            "batch_kv":      [],
        }, config={"run_name": "indexing-run"})