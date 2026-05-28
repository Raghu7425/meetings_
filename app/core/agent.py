"""
Handles AI agent logic, memory, and response generation.

- Initializes and manages LLM (Ollama) and embedding model
- Loads company data and builds vector database for retrieval
- Converts user query into embeddings and finds relevant context
- Generates AI response using LLM with streaming output

- Stores session memory:
  • conversation history
  • user intent and emotion
  • selected voice
- Supports caching to speed up repeated responses

- Provides helper APIs:
  • get session state and history
  • update voice and clear session
- Generates backchannel responses during conversation

This file is responsible for understanding user input and generating AI responses.
"""


import os
import re
import time
import requests
import json
import asyncio
import logging
import random
import faiss
import httpx
import PyPDF2
import docx
import numpy as np
from rapidfuzz import fuzz
from collections import OrderedDict
from typing import AsyncIterator, TypedDict
from sentence_transformers import SentenceTransformer
from langchain_core.messages import HumanMessage, AIMessage
from app.core.prompts import SYSTEM_PROMPT
from app.config import (BC_COOLDOWN, MAX_CACHE_SIZE, SENTENCE_TRANSFORMER_MODEL, LLM_MODEL,
                        LLM_API_TIMEOUT, LLM_WARMUP_TIMEOUT, LLM_WARMUP_NUM_PREDICT,
                        LLM_STREAM_REQUEST_TIMEOUT, LLM_MAX_GENERATION_TOKENS, LLM_TEMPERATURE,
                        LLM_CONTEXT_WINDOW_TOKENS, LLM_CHAT_HISTORY_LIMIT, CHUNK_SIZE, RETRIEVAL_TOP_K,
                        FUZZY_MATCH_THRESHOLD, ENTITY_CORRECTIONS, MAX_HISTORY_TURNS, INPUT_DIR,
                        FAISS_INDEX_DIR, CHUNKS_NPY_DIR, VECTOR_DB_DIR, OLLAMA_BASE_URL)


log = logging.getLogger("agent")

embed_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL, device="cpu")

_BACKCHANNELS = ["Okay…", "Got it…", "Alright…", "I see…", "Sure…"]


class SessionMemory(TypedDict):
    intent: str
    emotion: str
    history: list
    voice: str


_last_bc_time: dict[str, float] = {}
_memories: dict[str, SessionMemory] = {}
_response_cache: OrderedDict[str, str] = OrderedDict()

_index = None
_chunks = None
_system_ready = False
_init_lock = asyncio.Lock()


def _get_memory(session_id: str) -> SessionMemory:
    if session_id not in _memories:
        _memories[session_id] = {
            "intent": "information_request",
            "emotion": "neutral",
            "history": [],
            "voice": "",
        }
    return _memories[session_id]



def _prune_history(history: list) -> list:
    if len(history) > MAX_HISTORY_TURNS * 2:
        return history[-(MAX_HISTORY_TURNS * 2):]
    return history



def keep_model_warm():
    try:
        requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": "",
                "keep_alive": -1,
            },
            timeout=LLM_API_TIMEOUT,
        )
        log.info("[agent] Model kept warm in memory")

    except Exception as e:
        log.warning(f"[agent] Could not set keep_alive: {e}")



def LLM_MODEL_initialization():
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=LLM_API_TIMEOUT)
        resp.raise_for_status()
        log.info("[agent] Ollama is reachable at %s", OLLAMA_BASE_URL)
    except Exception as e:
        log.warning("[agent] Ollama not reachable at %s: %s — LLM features will be degraded", OLLAMA_BASE_URL, e)



def split_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    
    sections = [s.strip() for s in text.split("---") if s.strip()]
    chunks = []

    for section in sections:
        words = section.split()
        if len(words) <= chunk_size:
            chunks.append(section)
        else:
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                if chunk.strip():
                    chunks.append(chunk)

    return chunks



# def load_knowledge_base_data() -> str:
#     try:
#         with open(KNOWLEDGE_BASE_DIR, "r", encoding="utf-8") as f:
#             return f.read()
    
#     except FileNotFoundError:
#         raise FileNotFoundError(f"knowledge_base.txt not found at: {KNOWLEDGE_BASE_DIR}")



def _read_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()



def _read_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])



def _read_pdf(file_path: str) -> str:

    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)

        # Handle encrypted PDFs
        if reader.is_encrypted:
            try:
                reader.decrypt("")  # try empty password
                log.info(f"[agent] Decrypted PDF: {file_path}")
            except Exception as e:
                log.error(f"[agent] Could not decrypt PDF: {file_path} ({e})")
                return ""

        for page in reader.pages:
            text += page.extract_text() or ""

    return text



def load_knowledge_base_data() -> str:
    
    all_text = ""

    for filename in os.listdir(INPUT_DIR):
        file_path = os.path.join(INPUT_DIR, filename)

        if not os.path.isfile(file_path):
            continue

        try:
            if filename.endswith(".txt"):
                log.info(f"[agent] Reading TXT: {filename}")
                all_text += _read_txt(file_path) + "\n\n"

            elif filename.endswith(".pdf"):
                log.info(f"[agent] Reading PDF: {filename}")
                all_text += _read_pdf(file_path) + "\n\n"

            elif filename.endswith(".docx"):
                log.info(f"[agent] Reading DOCX: {filename}")
                all_text += _read_docx(file_path) + "\n\n"

            else:
                log.warning(f"[agent] Skipping unsupported file: {filename}")

        except Exception as e:
            log.error(f"[agent] Failed to read {filename}: {e}")

    return all_text



def vector_database():

    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    if os.path.exists(FAISS_INDEX_DIR) and os.path.exists(CHUNKS_NPY_DIR):
        log.info("[agent] Loading existing vector DB...")
        index = faiss.read_index(FAISS_INDEX_DIR)
        chunks = np.load(CHUNKS_NPY_DIR, allow_pickle=True)
        return index, chunks

    log.info("[agent] Creating vector DB...")
    text = load_knowledge_base_data()
    if not text.strip():
        raise ValueError("[agent] No readable content found in input/ folder.")
    
    chunks = split_text(text)

    embeddings = embed_model.encode(chunks, normalize_embeddings=True).astype("float32")
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, FAISS_INDEX_DIR)
    np.save(CHUNKS_NPY_DIR, chunks)

    return index, chunks



def normalize_query(query: str) -> str:

    query = query.lower().strip()
    query = re.sub(r"[^\w\s]", "", query)

    for correct, mistakes in ENTITY_CORRECTIONS.items():
        for mistake in mistakes:
            if fuzz.ratio(query, mistake) >= FUZZY_MATCH_THRESHOLD:
                query = query.replace(mistake, correct)

    return query



def get_top_chunks(query: str, index, chunks, k: int = RETRIEVAL_TOP_K):

    q_embed = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    _, indices = index.search(q_embed, k)

    return [chunks[i] for i in indices[0]]



def initialize_system():

    LLM_MODEL_initialization()
    index, chunks = vector_database()

    log.info("[agent] Warming up model...")
    keep_model_warm()

    try:
        requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": "hi",
                "keep_alive": -1,
                "options": {"num_predict": LLM_WARMUP_NUM_PREDICT},
            },
            timeout=LLM_WARMUP_TIMEOUT,
        )
    except Exception as e:
        log.warning(f"[agent] Warmup request failed: {e}")

    log.info("[agent] System ready")
    return index, chunks



async def ensure_system_initialized():

    global _index, _chunks, _system_ready

    if _system_ready and _index is not None and _chunks is not None:
        return _index, _chunks

    async with _init_lock:
        if _system_ready and _index is not None and _chunks is not None:
            return _index, _chunks

        try:
            _index, _chunks = await asyncio.to_thread(initialize_system)
            _system_ready = True
        except Exception as e:
            log.warning("[agent] System initialization failed: %s — continuing degraded", e)
            _system_ready = True  # prevent repeated retries on every request

        return _index, _chunks



async def generate_response_stream(index, chunks, query: str, session_id: str):
    
    query_clean = normalize_query(query=query)
    top_chunks = get_top_chunks(query=query_clean, index=index, chunks=chunks)
    context = "\n\n".join(top_chunks)
    conversation_history = get_last_conversations(session_id=session_id)

    prompt = SYSTEM_PROMPT.format(conversation_history=conversation_history, context=context, query=query)
    
    async with httpx.AsyncClient(timeout=LLM_STREAM_REQUEST_TIMEOUT) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": True,
                "keep_alive": -1,
                "options": {
                    "temperature": LLM_TEMPERATURE,
                    "num_predict": LLM_MAX_GENERATION_TOKENS,
                    "num_ctx": LLM_CONTEXT_WINDOW_TOKENS,
                },
            },
        ) as response:
            async for line in response.aiter_lines():
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    log.warning(f"[{session_id}] Skipping malformed Ollama stream line")
                    continue

                token = data.get("response", "")
                if token:
                    yield token

                if data.get("done"):
                    break



def get_session_state(session_id: str) -> dict:

    mem = _get_memory(session_id)
    return {
        "intent": mem["intent"],
        "emotion": mem["emotion"],
        "turns": len([m for m in mem["history"] if isinstance(m, HumanMessage)]),
        "voice": mem["voice"],
    }



def get_conversation_history(session_id: str) -> list[dict]:

    mem = _get_memory(session_id)

    result = []
    for msg in mem["history"]:
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})

    return result



def get_last_conversations(session_id: str, limit: int = LLM_CHAT_HISTORY_LIMIT) -> str:

    history = get_conversation_history(session_id)
    history = [h for h in history if h.get("content")]
    history = history[-limit:]

    lines = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")

    return "\n".join(lines)



def get_last_assistant_response(session_id: str) -> str:

    mem = _get_memory(session_id)
    for msg in reversed(mem["history"]):
        if isinstance(msg, AIMessage) and msg.content.strip():
            return msg.content.strip()

    return ""



def update_session_voice(session_id: str, voice: str) -> None:
    _get_memory(session_id)["voice"] = voice



def get_backchannel(session_id: str) -> str:

    now = time.monotonic()
    if now - _last_bc_time.get(session_id, 0.0) < BC_COOLDOWN:
        return ""
    _last_bc_time[session_id] = now

    return random.choice(_BACKCHANNELS)



async def _classify_and_store(user_text: str, session_id: str):

    text = user_text.lower()
    emotion = "neutral"
    intent = "information_request"

    if any(x in text for x in ["hi", "hello", "hey"]):
        intent = "greeting"
    elif any(x in text for x in ["bye", "goodbye"]):
        intent = "goodbye"
    elif any(x in text for x in ["problem", "issue", "not working"]):
        intent = "complaint"

    if any(x in text for x in ["angry", "bad", "worst"]):
        emotion = "angry"
    elif any(x in text for x in ["happy", "great", "awesome"]):
        emotion = "happy"

    mem = _get_memory(session_id)
    mem["intent"] = intent
    mem["emotion"] = emotion



def _append_user(session_id: str, text: str):
    mem = _get_memory(session_id)
    mem["history"].append(HumanMessage(content=text))
    mem["history"] = _prune_history(mem["history"])


def _append_assistant(session_id: str, text: str):
    mem = _get_memory(session_id)
    mem["history"].append(AIMessage(content=text))
    mem["history"] = _prune_history(mem["history"])


def truncate_last_assistant(session_id: str, spoken_text: str):
    history = _get_memory(session_id)["history"]
    if history and isinstance(history[-1], AIMessage):
        history[-1] = AIMessage(content=spoken_text.strip())


def clear_session(session_id: str):
    _memories.pop(session_id, None)
    _last_bc_time.pop(session_id, None)


def get_cache(key: str) -> str | None:
    if key in _response_cache:
        _response_cache.move_to_end(key)
        return _response_cache[key]
    return None



def set_cache(key: str, value: str) -> None:

    if key in _response_cache:
        _response_cache.move_to_end(key)

    _response_cache[key] = value
    if len(_response_cache) > MAX_CACHE_SIZE:
        _response_cache.popitem(last=False)



async def chat_stream(user_text: str, session_id: str) -> AsyncIterator[str]:

    yield " "
    log.debug(f"[{session_id}] user input received (len={len(user_text)})")

    index, chunks = await ensure_system_initialized()

    _append_user(session_id, user_text)
    asyncio.ensure_future(_classify_and_store(user_text, session_id))

    cache_key = f"{session_id}:{user_text.strip().lower()}"
    cached = get_cache(cache_key)

    if cached is not None:
        for word in cached.split():
            yield word + " "
            await asyncio.sleep(0.001)
        _append_assistant(session_id, cached)
        return

    full_text = ""

    try:
        async for token in generate_response_stream(index, chunks, user_text, session_id):
            full_text += token
            yield token

    except Exception as e:
        log.error(f"[{session_id}] Ollama stream error: {e}")
        fallback = "Sorry, something went wrong."
        yield fallback
        _append_assistant(session_id, fallback)
        return

    set_cache(cache_key, full_text)
    _append_assistant(session_id, full_text)



async def chat(user_text: str, session_id: str) -> str:

    parts = []
    async for token in chat_stream(user_text, session_id):
        parts.append(token)

    return "".join(parts).strip()

