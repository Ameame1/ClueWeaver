"""Paragraph-based segmentation with optional retrieval anchors."""
from __future__ import annotations
import hashlib
import math
import os
import re
from dataclasses import dataclass

import numpy as np


@dataclass
class Segment:
    seg_id: int
    start_para: int  # 1-indexed (inclusive)
    end_para: int  # 1-indexed (inclusive)
    text: str
    kind: str = "window"


RE_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{1,}|\d+")
STOP_WORDS = set(
    "a an the of in on at by to from for with as is are was were be been being have has had "
    "do does did will would shall should may might can could and or but not no nor so than "
    "that this these those i you he she it we they him her us them my your his its our their "
    "what which who whom whose why when where how does did was were about into onto over under "
    "after before during between among one two three four following option options true false claim "
    "question answer"
    .split()
)


def _tokenize(text: str) -> list[str]:
    tokens = []
    for m in RE_WORD.finditer(text):
        word = m.group(0).lower()
        if word in STOP_WORDS:
            continue
        tokens.append(word)
    return tokens


def _word_count(text: str) -> int:
    return len(text.split())


def _format_segment(paragraphs: list[str], start: int, end: int) -> str:
    return "\n".join(f"[{i + 1}] {paragraphs[i]}" for i in range(start, end) if paragraphs[i])


def _trim_words(text: str, max_words: int | None) -> str:
    if not max_words:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "\n... [segment truncated to max words]"


def _query_text(question: str, options: dict[str, str] | None = None) -> str:
    if not options:
        return question
    return question + " " + " ".join(options.get(k, "") for k in "ABCD")


def paragraphs_to_segments(
    paragraphs: list[str],
    paras_per_segment: int = 60,
    overlap_paras: int = 5,
    max_segment_words: int | None = None,
) -> list[Segment]:
    """Split paragraph list into windows of `paras_per_segment` with `overlap_paras` overlap.

    Paragraphs are stored 0-indexed but exposed as 1-indexed source numbering.
    """
    n = len(paragraphs)
    segs: list[Segment] = []
    seg_id = 0
    i = 0
    stride = max(1, paras_per_segment - overlap_paras)
    word_counts = [_word_count(p) for p in paragraphs]
    prefix = [0]
    for wc in word_counts:
        prefix.append(prefix[-1] + wc)

    while i < n:
        end = min(i + paras_per_segment, n)
        if max_segment_words:
            while end > i + 1 and prefix[end] - prefix[i] > max_segment_words:
                end -= 1
        text = _format_segment(paragraphs, i, end)
        if text.strip():
            segs.append(Segment(seg_id=seg_id, start_para=i + 1, end_para=end, text=text, kind="window"))
            seg_id += 1
        if end >= n:
            break
        i += stride
    return segs


def _span_to_window_segments(
    paragraphs: list[str],
    start: int,
    end: int,
    seg_id: int,
    paras_per_segment: int,
    overlap_paras: int,
    max_segment_words: int | None,
) -> tuple[list[Segment], int]:
    """Window a half-open paragraph span [start, end) without crossing anchors."""
    if start >= end:
        return [], seg_id
    segs: list[Segment] = []
    stride = max(1, paras_per_segment - overlap_paras)
    word_counts = [_word_count(p) for p in paragraphs]
    prefix = [0]
    for wc in word_counts:
        prefix.append(prefix[-1] + wc)

    i = start
    while i < end:
        j = min(i + paras_per_segment, end)
        if max_segment_words:
            while j > i + 1 and prefix[j] - prefix[i] > max_segment_words:
                j -= 1
        text = _format_segment(paragraphs, i, j)
        if text.strip():
            segs.append(Segment(seg_id=seg_id, start_para=i + 1, end_para=j, text=text, kind="window"))
            seg_id += 1
        if j >= end:
            break
        i += stride
    return segs, seg_id


def bm25_top_paragraphs(paragraphs: list[str], query: str, top_k: int = 8) -> list[int]:
    """Return 0-indexed paragraph ids using a small built-in BM25 scorer."""
    q_terms = _tokenize(query)
    if not q_terms:
        return []
    q_set = set(q_terms)
    docs = [_tokenize(p) for p in paragraphs]
    n_docs = len(docs)
    if not n_docs:
        return []
    avgdl = sum(len(d) for d in docs) / max(1, n_docs)
    df = {}
    for d in docs:
        for t in set(d):
            if t in q_set:
                df[t] = df.get(t, 0) + 1
    idf = {t: math.log(1 + (n_docs - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5)) for t in q_set}
    k1 = 1.5
    b = 0.75
    scored = []
    for i, d in enumerate(docs):
        if not d:
            continue
        tf = {}
        for t in d:
            if t in q_set:
                tf[t] = tf.get(t, 0) + 1
        if not tf:
            continue
        dl = len(d)
        score = 0.0
        denom_base = k1 * (1 - b + b * dl / max(1e-6, avgdl))
        for t, f in tf.items():
            score += idf.get(t, 0.0) * f * (k1 + 1) / (f + denom_base)
        if score > 0:
            scored.append((score, i))
    scored.sort(reverse=True)
    return [i for _, i in scored[:top_k]]


_DENSE_MODEL = None
_DENSE_TOKENIZER = None
_DENSE_DEVICE = None
_DENSE_BACKEND = None
_DENSE_CACHE: dict[str, np.ndarray] = {}


def _dense_backend():
    """Lazy-load a local dense retriever through transformers.

    BGE-M3 is the default dense retriever used in the paper. DENSE_MODEL_PATH
    can be set to another local embedding model without changing callers.
    """
    global _DENSE_MODEL, _DENSE_TOKENIZER, _DENSE_DEVICE, _DENSE_BACKEND
    if _DENSE_MODEL is not None:
        return _DENSE_TOKENIZER, _DENSE_MODEL, _DENSE_DEVICE, _DENSE_BACKEND
    from transformers import AutoModel, AutoTokenizer
    import torch

    model_path = os.environ.get("DENSE_MODEL_PATH") or os.environ.get("BGE_M3_PATH", "models/bge-m3")
    device = os.environ.get("RETRIEVER_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    trust_remote_code = os.environ.get("DENSE_TRUST_REMOTE_CODE", "").lower() in {"1", "true", "yes"} \
        or "NV-Embed" in model_path
    _DENSE_TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model_kwargs = {"trust_remote_code": trust_remote_code}
    if "NV-Embed" in model_path or "nv-embed" in model_path.lower():
        # Some remote-code embedding models need explicit fp16 loading to avoid
        # excessive memory use.
        model_kwargs["torch_dtype"] = torch.float16
    _DENSE_MODEL = AutoModel.from_pretrained(model_path, **model_kwargs)
    # Disable generation cache for embedding-only remote-code models when needed.
    if "NV-Embed" in model_path or "nv-embed" in model_path.lower():
        if hasattr(_DENSE_MODEL, "config"):
            _DENSE_MODEL.config.use_cache = False
        inner = getattr(_DENSE_MODEL, "embedding_model", None)
        if inner is not None and hasattr(inner, "config"):
            inner.config.use_cache = False
    _DENSE_MODEL.to(device)
    _DENSE_MODEL.eval()
    _DENSE_DEVICE = device
    _DENSE_BACKEND = "nvembed" if hasattr(_DENSE_MODEL, "encode") and "NV-Embed" in model_path else "mean_pool"
    return _DENSE_TOKENIZER, _DENSE_MODEL, _DENSE_DEVICE, _DENSE_BACKEND


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _mean_pool(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def _dense_embed(texts: list[str], batch_size: int = 16, max_length: int = 512, is_query: bool = False) -> np.ndarray:
    import torch

    tokenizer, model, device, backend = _dense_backend()
    out = []
    missing = []
    missing_keys = []
    for text in texts:
        key = _hash_text(f"{backend}|{'q' if is_query else 'p'}|{text}")
        cached = _DENSE_CACHE.get(key)
        if cached is not None:
            out.append(cached)
        else:
            out.append(None)
            missing.append(text)
            missing_keys.append(key)

    encoded_missing = []
    if backend == "nvembed":
        instruction = os.environ.get(
            "NV_EMBED_QUERY_INSTRUCTION",
            "Instruct: Given a question, retrieve passages that answer the question\nQuery: ",
        ) if is_query else os.environ.get("NV_EMBED_PASSAGE_INSTRUCTION", "")
        with torch.no_grad():
            for i in range(0, len(missing), batch_size):
                batch = missing[i:i + batch_size]
                emb = model.encode(batch, instruction=instruction, max_length=max_length)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                encoded_missing.extend(emb.detach().cpu().float().numpy())
    else:
        with torch.no_grad():
            for i in range(0, len(missing), batch_size):
                batch = missing[i:i + batch_size]
                inputs = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                emb = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                encoded_missing.extend(emb.detach().cpu().float().numpy())

    j = 0
    for key, emb in zip(missing_keys, encoded_missing):
        _DENSE_CACHE[key] = emb
    final = []
    for item in out:
        if item is None:
            final.append(encoded_missing[j])
            j += 1
        else:
            final.append(item)
    return np.vstack(final) if final else np.zeros((0, 1), dtype=np.float32)


def dense_top_paragraphs(
    paragraphs: list[str],
    query: str,
    candidate_ids: list[int] | None = None,
    top_k: int = 8,
    batch_size: int = 16,
) -> list[int]:
    """Dense rerank candidate paragraphs with local BGE-M3 embeddings."""
    if candidate_ids is None:
        candidate_ids = list(range(len(paragraphs)))
    if not candidate_ids or top_k <= 0:
        return []
    remote_ids = _remote_dense_top_paragraphs(paragraphs, query, candidate_ids, top_k, batch_size)
    if remote_ids is not None:
        return remote_ids
    try:
        batch_size = int(os.environ.get("DENSE_BATCH_SIZE", str(batch_size)))
        candidate_texts = [paragraphs[i] for i in candidate_ids]
        max_length = int(os.environ.get("DENSE_MAX_LENGTH", "512"))
        query_emb = _dense_embed([query], batch_size=1, max_length=max_length, is_query=True)
        doc_emb = _dense_embed(candidate_texts, batch_size=batch_size, max_length=max_length, is_query=False)
        scores = doc_emb @ query_emb[0]
        order = np.argsort(-scores)[:top_k]
        return [candidate_ids[int(i)] for i in order]
    except Exception as exc:
        print(f"[retrieval] dense disabled after error: {type(exc).__name__}: {exc}", flush=True)
        return []


def _remote_dense_top_paragraphs(
    paragraphs: list[str],
    query: str,
    candidate_ids: list[int],
    top_k: int,
    batch_size: int,
) -> list[int] | None:
    """Call a persistent dense retriever service when configured."""
    base_url = os.environ.get("DENSE_RETRIEVER_URL", "").strip().rstrip("/")
    if not base_url:
        return None
    try:
        import json
        import urllib.request

        payload = {
            "paragraphs": paragraphs,
            "query": query,
            "candidate_ids": candidate_ids,
            "top_k": top_k,
            "batch_size": batch_size,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        timeout = float(os.environ.get("DENSE_RETRIEVER_TIMEOUT", "120"))
        req = urllib.request.Request(
            f"{base_url}/topk",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        ids = body.get("ids")
        if not isinstance(ids, list):
            return []
        return [int(i) for i in ids[:top_k]]
    except Exception as exc:
        print(f"[retrieval] remote dense disabled after error: {type(exc).__name__}: {exc}", flush=True)
        return None


def retrieval_aware_segments(
    paragraphs: list[str],
    question: str,
    options: dict[str, str] | None = None,
    paras_per_segment: int = 60,
    overlap_paras: int = 5,
    max_segment_words: int | None = 1600,
    bm25_top_k: int = 8,
    dense_top_k: int = 8,
    dense_candidates: int = 64,
    option_bm25_top_k: int = 2,
    option_dense_top_k: int = 1,
    option_dense_candidates: int = 32,
    tail_bm25_top_k: int = 2,
    option_tail_bm25_top_k: int = 8,
    tail_fraction: float = 0.25,
    dense_batch_size: int = 16,
    use_dense: bool = True,
) -> list[Segment]:
    """Cut the story around hybrid-retrieval anchors.

    BM25 and dense hits are forced to be standalone one-paragraph segments.
    Remaining gaps are windowed normally, but windows never cross an anchor. This
    uses RAG only to improve segmentation boundaries, not as a separate evidence
    channel.
    """
    retrieval_ids: list[int] = []
    seen_ids: set[int] = set()
    bm25_hit_ids: set[int] = set()
    dense_hit_ids: set[int] = set()
    tail_hit_ids: set[int] = set()

    def add_query_hits(query: str, bm25_k: int, dense_k: int, dense_pool: int) -> None:
        bm25_ids = bm25_top_paragraphs(paragraphs, query, top_k=max(bm25_k, dense_pool))
        dense_ids = dense_top_paragraphs(
            paragraphs,
            query,
            candidate_ids=bm25_ids[:dense_pool],
            top_k=dense_k,
            batch_size=dense_batch_size,
        ) if use_dense and dense_k > 0 else []
        bm25_hit_ids.update(bm25_ids[:bm25_k])
        dense_hit_ids.update(dense_ids)
        for pid in bm25_ids[:bm25_k] + dense_ids:
            if pid not in seen_ids:
                seen_ids.add(pid)
                retrieval_ids.append(pid)

    def add_tail_hits(query: str, bm25_k: int) -> None:
        if bm25_k <= 0 or not paragraphs:
            return
        start = max(0, int(len(paragraphs) * (1.0 - tail_fraction)))
        local_ids = bm25_top_paragraphs(paragraphs[start:], query, top_k=bm25_k)
        for local_pid in local_ids:
            pid = start + local_pid
            bm25_hit_ids.add(pid)
            tail_hit_ids.add(pid)
            if pid not in seen_ids:
                seen_ids.add(pid)
                retrieval_ids.append(pid)

    query = _query_text(question, options)
    tail_enabled = bool(options and re.search(
        r"\b(false|not true|except)\b|most likely .*responsible|object.*switch|switch.*object",
        question,
        re.I,
    ))
    who_killed_tail = bool(options and re.search(r"\bwho (killed|pushed)\b", question, re.I)) and not re.search(
        r"\b(perspective|suspect|suspected|police|inspector)\b",
        question,
        re.I,
    )
    add_query_hits(query, bm25_top_k, dense_top_k, dense_candidates)
    if options and (tail_enabled or who_killed_tail):
        add_tail_hits(query, tail_bm25_top_k)
    if options and tail_enabled:
        for letter in "ABCD":
            opt = options.get(letter, "").strip()
            if not opt:
                continue
            add_query_hits(f"{question}\nOption {letter}: {opt}", option_bm25_top_k, option_dense_top_k,
                           option_dense_candidates)
            if tail_enabled:
                add_tail_hits(f"Option {letter}: {opt}", option_tail_bm25_top_k)

    anchor_set = set(retrieval_ids)
    segs: list[Segment] = []
    seg_id = 0

    cursor = 0
    for pid in sorted(anchor_set):
        gap_segs, seg_id = _span_to_window_segments(
            paragraphs,
            cursor,
            pid,
            seg_id,
            paras_per_segment,
            overlap_paras,
            max_segment_words,
        )
        segs.extend(gap_segs)

        text = _trim_words(f"[{pid + 1}] {paragraphs[pid]}", max_segment_words)
        if not text.strip():
            cursor = pid + 1
            continue
        prefix = "retrieval_tail" if pid in tail_hit_ids else "retrieval"
        kind = f"{prefix}_bm25_dense" if pid in bm25_hit_ids and pid in dense_hit_ids else (
            f"{prefix}_dense" if pid in dense_hit_ids else f"{prefix}_bm25"
        )
        segs.append(Segment(seg_id=seg_id, start_para=pid + 1, end_para=pid + 1, text=text, kind=kind))
        seg_id += 1
        cursor = pid + 1

    tail_segs, seg_id = _span_to_window_segments(
        paragraphs,
        cursor,
        len(paragraphs),
        seg_id,
        paras_per_segment,
        overlap_paras,
        max_segment_words,
    )
    segs.extend(tail_segs)
    return segs
