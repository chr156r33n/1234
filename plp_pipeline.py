from __future__ import annotations

import csv
import io
import logging
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Callable, TypeVar

import vertexai
from vertexai.generative_models import GenerativeModel, Part, Content

from mcp_client import SemrushMCPClient  # your existing class

# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Attach a basic handler if not already configured by caller
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)

# Simple counters so you can see how much you burned on a run
MCP_CALL_COUNT = 0
LLM_CALL_COUNT = 0

T = TypeVar("T")


def with_retries(
    func: Optional[Callable[..., T]] = None,
    *,
    retries: int = 2,
    delay: float = 1.0,
    backoff: float = 2.0,
    label: str = "call",
) -> Callable[..., T]:
    """
    Tiny retry wrapper to avoid throwing the entire run over a transient error.
    Can be used as @with_retries or @with_retries(label="...")
    """
    def decorator(f: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            attempts = 0
            current_delay = delay
            while True:
                try:
                    return f(*args, **kwargs)
                except Exception as exc:
                    attempts += 1
                    if attempts > retries:
                        logger.error(f"{label} failed after {attempts} attempts: {exc}")
                        raise
                    logger.warning(f"{label} failed (attempt {attempts}), retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper

    if func is None:
        return decorator
    return decorator(func)


# -------------------------------------------------------------------
# Data structures
# -------------------------------------------------------------------

@dataclass
class SeedKeyword:
    keyword: str
    priority: float
    rationale: str


@dataclass
class KeywordCandidate:
    keyword: str
    volume: int
    cpc: float
    competition: float
    # num_results: Optional[int] = None
    current_rank: Optional[int] = None
    serp_top3: Optional[List[Dict[str, str]]] = None
    best_page_type: Optional[str] = None
    page_type_confidence: Optional[float] = None
    selection_role: Optional[str] = None
    selection_reason: Optional[str] = None
    # cluster_id: Optional[int] = None
    cluster_label: Optional[str] = None
    page_type_status: Optional[str] = None  
    page_type_explanation: Optional[str] = None 


# -------------------------------------------------------------------
# LLM wrapper (no tools, just text)
# -------------------------------------------------------------------

class VertexLLM:
    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        model_name: str = "gemini-2.5-flash",
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
    ) -> None:
        vertexai.init(project=project_id, location=location)
        parts = [Part.from_text(system_instruction)] if system_instruction else None
        self.model = GenerativeModel(
            model_name=model_name,
            system_instruction=parts,
        )
        self.generation_config = {
            "temperature": temperature,
            "max_output_tokens": 30096,
            "top_p": 0.95,
            "top_k": 32,
        }

    @with_retries(label="LLM.generate")
    def generate(self, prompt: str) -> str:
        global LLM_CALL_COUNT
        LLM_CALL_COUNT += 1
        logger.debug(f"[LLM] Prompt length: {len(prompt)} chars")

        resp = self.model.generate_content(
            [Content(role="user", parts=[Part.from_text(prompt)])],
            generation_config=self.generation_config,
        )
        if not resp.candidates:
            raise RuntimeError("No candidates from model")

        cand = resp.candidates[0]

        finish_reason = getattr(cand, "finish_reason", None)
        if finish_reason:
            finish_reason_str = str(finish_reason)
            if hasattr(finish_reason, "name"):
                finish_reason_str = finish_reason.name
            if "MAX_TOKENS" in finish_reason_str or "max_tokens" in finish_reason_str.lower():
                logger.warning("[LLM] Response was truncated due to MAX_TOKENS limit.")

        try:
            text = cand.text
        except Exception:
            text = ""

        if not text:
            chunks = []
            for p in cand.content.parts:
                t = getattr(p, "text", None)
                if t:
                    chunks.append(t)
            text = "\n".join(chunks).strip()

        logger.debug(f"[LLM] Response length: {len(text)} chars")
        if finish_reason and ("MAX_TOKENS" in str(finish_reason) or (hasattr(finish_reason, "name") and "MAX_TOKENS" in finish_reason.name)):
            logger.warning(f"[LLM] Response may be incomplete - truncated at {len(text)} chars")
        if not text:
            raise RuntimeError("Model returned no user-visible text")

        return text.strip()


# -------------------------------------------------------------------
# MCP helpers
# -------------------------------------------------------------------

@with_retries(label="SemrushMCP.call")
def mcp_call_with_logging(
    mcp: SemrushMCPClient,
    tool_name: str,
    args: Dict[str, Any],
) -> Dict[str, Any]:
    global MCP_CALL_COUNT
    MCP_CALL_COUNT += 1    # noqa: PLW0603
    logger.debug(f"[MCP] Calling {tool_name} with args={args}")
    resp = mcp.call(tool_name, args)
    logger.debug(f"[MCP] {tool_name} raw response keys: {list(resp.keys())}")
    return resp


def _extract_first_text_content(mcp_response: Dict[str, Any]) -> str:
    """
    Extract the first text content from an MCP tool response.
    """
    if not mcp_response.get("success", False):
        error_msg = mcp_response.get("error") or mcp_response.get("message", "Unknown error")
        logger.warning(f"[MCP] Response indicates failure: {error_msg}")
        return ""

    data = mcp_response.get("data")
    if data is None:
        logger.warning("[MCP] No data field in response")
        return ""

    if isinstance(data, list):
        contents = data
    elif isinstance(data, dict) and "content" in data:
        contents = data.get("content", [])
    elif isinstance(data, dict) and "result" in data:
        result = data.get("result", {})
        contents = result.get("content", [])
    else:
        logger.warning(f"[MCP] Unexpected data format: {type(data)}. Keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
        return ""

    if not contents:
        logger.warning("[MCP] Empty content array in response")
        return ""

    first = contents[0]
    if not isinstance(first, dict):
        logger.warning(f"[MCP] First content item is not a dict: {type(first)}")
        return ""

    text = first.get("text", "")
    if not text:
        logger.warning("[MCP] First content item has no 'text'")
        return ""

    if text.strip().startswith("ERROR"):
        logger.warning(f"[MCP] API returned error: {text.strip()}")
        return ""

    return text


def _parse_semrush_csv_text(csv_text: str) -> List[Dict[str, str]]:
    """Semrush MCP responds with 'Keyword;Search Volume;...' CSV-ish strings."""
    csv_text = csv_text.strip()
    if not csv_text:
        return []

    if csv_text.startswith("ERROR"):
        logger.debug(f"[MCP] Skipping error message: {csv_text[:100]}")
        return []

    normalized = csv_text.replace(";", ",")
    try:
        reader = csv.DictReader(io.StringIO(normalized))
        rows = [row for row in reader]
        logger.debug(f"[MCP] Parsed {len(rows)} rows from CSV text")
        return rows
    except Exception as exc:
        logger.warning(f"[MCP] Failed to parse CSV text: {exc}. Text preview: {csv_text[:200]}")
        return []


# -------------------------------------------------------------------
# Seed extraction
# -------------------------------------------------------------------

SEED_EXTRACT_PROMPT = """You are an SEO specialist.

You are given the content of a product listing page (PLP) or category page.
Your job is to extract 5-7 candidate *seed keywords* that best describe the
core commercial topic of this page in search.

Rules:
- Focus on non-branded or lightly branded head and mid-tail terms.
- Ignore ultra-generic one-word terms (e.g. "skin", "face") unless they are clearly core.
- Prefer queries that a real user would type when looking for the products on this page.
- Do NOT output anything except JSON.
- Keep rationale text concise (under 100 characters) to avoid truncation.

Return JSON in this exact shape:
{{
  "seeds": [
    {{
      "keyword": "mens skincare",
      "priority": 0.95,
      "rationale": "Core PLP theme and strong commercial intent."
    }}
  ]
}}

CRITICAL: Output complete, valid JSON only. Do not truncate your response.
Now analyse the page and respond with JSON only.
PAGE TITLE:
{title}

H1:
{h1}

BODY (truncated if long):
{body}
"""


def _strip_markdown_code_blocks(text: str) -> str:
    """Strip markdown code block markers (```json, ```, etc.) from text."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        else:
            text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def extract_seeds_from_page(
    llm: VertexLLM,
    title: str,
    h1: str,
    body_text: str,
) -> List[SeedKeyword]:
    from json import loads

    prompt = SEED_EXTRACT_PROMPT.format(title=title, h1=h1, body=body_text[:8000])
    raw = llm.generate(prompt)
    raw = _strip_markdown_code_blocks(raw)

    try:
        data = loads(raw)
    except Exception as exc:
        logger.error(f"[Seeds] Failed to parse JSON from LLM: {exc}")
        logger.error(f"[Seeds] Response length: {len(raw)} chars")
        logger.error(f"[Seeds] Response (first 800 chars): {raw[:800]}")
        if len(raw) > 800:
            logger.error(f"[Seeds] Response (last 200 chars): ...{raw[-200:]}")
        raise

    seeds: List[SeedKeyword] = []
    for item in data.get("seeds", []):
        try:
            seeds.append(
                SeedKeyword(
                    keyword=item["keyword"].strip(),
                    priority=float(item.get("priority", 0.5)),
                    rationale=item.get("rationale", "").strip(),
                )
            )
        except Exception as exc:
            logger.warning(f"[Seeds] Skipping malformed seed item {item}: {exc}")

    logger.info(f"[Seeds] Extracted {len(seeds)} seeds")
    return seeds


# -------------------------------------------------------------------
# Keyword discovery
# -------------------------------------------------------------------

def discover_keywords_with_semrush(
    mcp: SemrushMCPClient,
    seeds: List[SeedKeyword],
    database: str,
    max_per_seed: int = 20,
    max_seeds: int = 5,
) -> List[KeywordCandidate]:
    if not seeds:
        logger.warning("[Discover] No seeds provided, skipping Semrush discovery")
        return []

    seeds = sorted(seeds, key=lambda s: s.priority, reverse=True)[:max_seeds]
    logger.info(f"[Discover] Using {len(seeds)} seeds (capped at {max_seeds})")

    candidates: Dict[str, KeywordCandidate] = {}

    def add_rows(rows: List[Dict[str, str]]):
        for row in rows:
            kw = row.get("Keyword") or row.get("keyword")
            if not kw:
                continue
            kw = kw.strip()
            if not kw:
                continue

            try:
                vol = int(row.get("Search Volume") or row.get("search_volume") or 0)
            except ValueError:
                vol = 0
            try:
                cpc = float(row.get("CPC") or row.get("cpc") or 0.0)
            except ValueError:
                cpc = 0.0
            try:
                comp = float(row.get("Competition") or row.get("competition") or 0.0)
            except ValueError:
                comp = 0.0
           # try:
             #   nres = row.get("Number of Results") or row.get("results") or None
             #   num_results = int(nres) if nres is not None else None
            #except ValueError:
               # num_results = None

            if kw in candidates:
                continue

            candidates[kw] = KeywordCandidate(
                keyword=kw,
                volume=vol,
                cpc=cpc,
                competition=comp,
               # num_results=num_results,
            )

    for seed in seeds:
        seed_kw = seed.keyword
        logger.info(f"[Discover] Processing seed '{seed_kw}'")

        try:
            overview_resp = mcp_call_with_logging(
                mcp,
                "get_keyword_overview",
                {"keyword": seed_kw, "database": database},
            )
            overview_text = _extract_first_text_content(overview_resp)
            add_rows(_parse_semrush_csv_text(overview_text))
        except Exception as exc:
            logger.warning(f"[Discover] overview failed for '{seed_kw}': {exc}")

        try:
            fullsearch_resp = mcp_call_with_logging(
                mcp,
                "get_keyword_fullsearch",
                {"keyword": seed_kw, "database": database, "display_limit": max_per_seed},
            )
            fullsearch_text = _extract_first_text_content(fullsearch_resp)
            add_rows(_parse_semrush_csv_text(fullsearch_text))
        except Exception as exc:
            logger.warning(f"[Discover] fullsearch failed for '{seed_kw}': {exc}")

        try:
            related_resp = mcp_call_with_logging(
                mcp,
                "get_keyword_related",
                {"keyword": seed_kw, "database": database, "display_limit": max_per_seed},
            )
            related_text = _extract_first_text_content(related_resp)
            add_rows(_parse_semrush_csv_text(related_text))
        except Exception as exc:
            logger.warning(f"[Discover] related failed for '{seed_kw}': {exc}")

        try:
            questions_resp = mcp_call_with_logging(
                mcp,
                "get_keyword_questions",
                {"keyword": seed_kw, "database": database, "display_limit": max_per_seed},
            )
            questions_text = _extract_first_text_content(questions_resp)
            add_rows(_parse_semrush_csv_text(questions_text))
        except Exception as exc:
            logger.warning(f"[Discover] questions failed for '{seed_kw}': {exc}")

    logger.info(f"[Discover] Collected {len(candidates)} unique candidate keywords")
    return list(candidates.values())


# -------------------------------------------------------------------
# Rankings enrichment
# -------------------------------------------------------------------

def enrich_with_current_rank(
    mcp: SemrushMCPClient,
    candidates: List[KeywordCandidate],
    target_domain: str,
    database: str,
    serp_limit: int = 20,
    max_keywords: int = 50,
) -> None:
    if not candidates:
        logger.warning("[Rank] No candidates to enrich with rankings")
        return

    target_domain_norm = target_domain.lower().strip().lstrip("www.")
    logger.info(f"[Rank] Enriching rankings for up to {max_keywords} keywords")

    for cand in candidates[:max_keywords]:
        kw = cand.keyword
        try:
            serp_resp = mcp_call_with_logging(
                mcp,
                "get_keyword_organic",
                {
                    "keyword": kw,
                    "database": database,
                    "display_limit": serp_limit,
                },
            )
        except Exception as exc:
            logger.warning(f"[Rank] SERP fetch failed for '{kw}': {exc}")
            cand.page_type_status = cand.page_type_status or "no_serp"
            cand.page_type_explanation = cand.page_type_explanation or "SERP fetch failed."
            continue

        serp_text = _extract_first_text_content(serp_resp)
        rows = _parse_semrush_csv_text(serp_text)
        serp_top3: List[Dict[str, str]] = []
        cand.current_rank = None

        if not rows:
            cand.page_type_status = cand.page_type_status or "no_serp"
            cand.page_type_explanation = cand.page_type_explanation or "No SERP data available for this keyword."
            cand.serp_top3 = []
            continue

        for j, row in enumerate(rows):
            dom = (row.get("Domain") or row.get("domain") or "").strip()
            url = (row.get("Url") or row.get("URL") or row.get("url") or "").strip()
            clean_dom = dom.lower().lstrip("www.")
            if j < 3 and dom:
                serp_top3.append({"domain": dom, "url": url})
            if clean_dom and clean_dom in target_domain_norm:
                pos_str = row.get("Position") or row.get("position")
                if pos_str:
                    try:
                        cand.current_rank = int(pos_str)
                    except ValueError:
                        cand.current_rank = None
                else:
                    cand.current_rank = j + 1

        cand.serp_top3 = serp_top3
        if serp_top3 and not cand.page_type_status:
            cand.page_type_status = "pending"
            cand.page_type_explanation = None

    logger.info("[Rank] Ranking enrichment completed")


# -------------------------------------------------------------------
# Page-type inference
# -------------------------------------------------------------------

PAGE_TYPE_PROMPT = """You are an SEO analyst.

You are given one or more keywords and, for each, the top ranking URLs.
For each keyword, infer what type of page search engines favour:

Allowed page types (choose ONE per keyword):
- "PLP" (category / listing page of multiple products)
- "PDP" (single product page)
- "Blog" (informational article / editorial)
- "Guide" (long-form buying guide / how-to)
- "Brand" (brand home page or brand hub)
- "Other" (forum, Q&A, directory, etc.)

Return strictly JSON in this format:

{{
  "results": [
    {{
      "keyword": "...",
      "best_page_type": "PLP",
      "confidence": 0.86,
      "reason": "Most top URLs are category pages listing multiple products."
    }}
  ]
}}

Here is the data:

{data}
"""


def infer_page_types(
    llm: VertexLLM,
    candidates: List[KeywordCandidate],
    max_keywords: int = 30,
) -> None:
    from json import dumps, loads

    subset = [c for c in candidates if c.serp_top3]
    subset = subset[:max_keywords]

    if not subset:
        logger.warning("[PageType] No candidates with SERP data to infer page types")
        return

    payload = []
    for cand in subset:
        payload.append(
            {
                "keyword": cand.keyword,
                "serp_top3": cand.serp_top3 or [],
            }
        )

    prompt = PAGE_TYPE_PROMPT.format(data=dumps(payload, indent=2))
    raw = llm.generate(prompt)
    raw = _strip_markdown_code_blocks(raw)

    try:
        data = loads(raw)
    except Exception as exc:
        logger.error(f"[PageType] Failed to parse JSON from LLM: {exc}")
        logger.error(f"[PageType] Response length: {len(raw)} chars")
        logger.error(f"[PageType] Response (first 800 chars): {raw[:800]}")
        if len(raw) > 800:
            logger.error(f"[PageType] Response (last 200 chars): ...{raw[-200:]}")
        logger.warning("[PageType] Marking candidates with SERP as llm_error")
        for cand in subset:
            cand.page_type_status = "llm_error"
            cand.page_type_explanation = "Page type inference failed: could not parse LLM JSON response."
        return

    mapping = {item["keyword"]: item for item in data.get("results", [])}

    for cand in subset:
        info = mapping.get(cand.keyword)
        if not info:
            if cand.page_type_status in (None, "pending"):
                cand.page_type_status = "llm_missing"
                cand.page_type_explanation = "LLM did not return a page type for this keyword."
            continue

        cand.best_page_type = info.get("best_page_type") or "Unknown"
        try:
            cand.page_type_confidence = float(info.get("confidence", 0.5))
        except (TypeError, ValueError):
            cand.page_type_confidence = 0.5

        cand.page_type_status = "inferred"
        cand.page_type_explanation = (info.get("reason") or "").strip() or None

    logger.info(f"[PageType] Inferred page type for {len(mapping)} keywords")


# -------------------------------------------------------------------
# PLP keyword selection (with clustering)
# -------------------------------------------------------------------

PLP_SELECTION_PROMPT = """You are helping select the best PLP (product listing page) target keywords.

You are given a list of candidate keywords with metrics:

- keyword
- search volume
- CPC
- competition
- current_rank (existing ranking position, or null)
- best_page_type (PLP, PDP, Blog, Guide, Brand, Other)
- page_type_confidence

Your task:
1. Select 10–20 keywords that are the best targets for a PLP.
2. Cluster these selected keywords into closely-related groups.
3. Prefer:
   - keywords where best_page_type is "PLP" with high confidence
   - good search volume without impossible competition
   - existing rankings where current_rank is between 2 and 20 (good uplift opportunity)
4. Allow a mix of:
   - 1–3 primary core terms
   - several mid-tail support terms
   - a few long-tail terms if they clearly fit the PLP

Return strictly JSON in this format (no extra text):

{{
  "selection": [
    {{
      "keyword": "example keyword",
      "keep": true,
      "role": "core",
      "cluster_label": "short human-readable label for this cluster",
      "reason": "Short rationale referencing volume, competition, rankings and cluster fit."
    }}
  ]
}}

Rules:
- role must be one of "core", "support", "long_tail".
- cluster_label should describe the shared theme (e.g. "anti aging moisturiser", "oil-free face cream").
- Keywords in the same semantic group share the same cluster_label.

Here is the candidate data:

{data}
"""


def select_plp_keywords(
    llm: VertexLLM,
    candidates: List[KeywordCandidate],
    max_candidates: int = 60,
) -> List[KeywordCandidate]:
    from json import dumps, loads

    if not candidates:
        logger.warning("[Select] No candidates provided, skipping selection")
        return []

    subset = candidates[:max_candidates]
    payload = []
    for c in subset:
        payload.append(
            {
                "keyword": c.keyword,
                "volume": c.volume,
                "cpc": c.cpc,
                "competition": c.competition,
                "current_rank": c.current_rank,
                "best_page_type": c.best_page_type,
                "page_type_confidence": c.page_type_confidence,
            }
        )

    prompt = PLP_SELECTION_PROMPT.format(data=dumps(payload, indent=2))
    raw = llm.generate(prompt)
    raw = _strip_markdown_code_blocks(raw)

    try:
        data = loads(raw)
    except Exception as exc:
        logger.error(f"[Select] Failed to parse JSON from LLM: {exc}. Raw: {raw[:400]}...")
        return []

    decisions = {item["keyword"]: item for item in data.get("selection", []) if "keyword" in item}

    selected: List[KeywordCandidate] = []
    for c in subset:
        info = decisions.get(c.keyword)
        if not info:
            continue
        if not info.get("keep"):
            continue

        role = (info.get("role") or "").strip()
        if role not in ("core", "support", "long_tail"):
            role = "support"
        c.selection_role = role

        c.selection_reason = (info.get("reason") or "").strip()

        #cluster_id = info.get("cluster_id")
      #  try:
        #    c.cluster_id = int(cluster_id) if cluster_id is not None else None
       # except (TypeError, ValueError):
         #   c.cluster_id = None

        c.cluster_label = (info.get("cluster_label") or "").strip() or None

        selected.append(c)

    logger.info(f"[Select] Selected {len(selected)} PLP keywords from {len(subset)} candidates")
    return selected


# -------------------------------------------------------------------
# Title & Description Recommendations
# -------------------------------------------------------------------

def suggest_page_title_and_h1(
    llm: VertexLLM,
    selected_keywords: List[KeywordCandidate],
    current_title: str,
    current_h1: str,
    max_options: int = 3,
) -> List[Dict[str, str]]:
    from json import dumps, loads

    core = [c.keyword for c in selected_keywords if c.selection_role == "core"]
    support = [c.keyword for c in selected_keywords if c.selection_role == "support"]

    payload = {
        "current_title": current_title,
        "current_h1": current_h1,
        "core_keywords": core[:3],
        "support_keywords": support[:7],
    }

    prompt = TITLE_H1_PROMPT.format(data=dumps(payload, indent=2), max_options=max_options)
    raw = llm.generate(prompt)
    raw = _strip_markdown_code_blocks(raw)
    data = loads(raw)

    return data.get("options", [])

TITLE_H1_PROMPT = """You are an SEO specialist for ecommerce PLPs.

You are given:
- The current HTML <title> and <h1> of a product listing page.
- A set of "core" and "support" keywords selected as the best PLP targets.

Your tasks:
1. Propose up to {max_options} alternative combinations of <title> and <h1>.
2. Each option should:
   - Strongly feature at least one core keyword.
   - Naturally incorporate 1–3 support keywords where appropriate.
   - Be readable and on-brand, not keyword spam.
3. Respect these constraints:
   - <title>: ideally 45–60 characters, never more than ~65.
   - <h1>: concise, human-readable, can be slightly longer than the title.

Return STRICTLY JSON in this format, with no extra text:

{{
  "options": [
    {{
      "title": "New HTML title here",
      "h1": "New H1 here",
      "primary_keywords": ["..."],
      "secondary_keywords": ["..."],
      "rationale": "Short explanation of why this is a good PLP fit."
    }}
  ]
}}

Here is the input data:

{data}
"""


# -------------------------------------------------------------------
# CSV output
# -------------------------------------------------------------------

def write_plp_csv(
    candidates: List[KeywordCandidate],
    file_path: str,
) -> None:
    if not candidates:
        logger.warning(f"[CSV] No candidates to write for {file_path}")
        return

    fieldnames = [
        "keyword",
        #"cluster_id",
        "clustered_term",  # from cluster_label
        "volume",
        "cpc",
        "competition",
       # "num_results",
        "current_rank",
        "best_page_type",
        "page_type_confidence",
        "page_type_status",
        "page_type_explanation",
        "selection_role",
        "selection_reason",
        "serp_top3_domains",
        "serp_top3_urls",
    ]
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in candidates:
            serp_domains = ",".join([s["domain"] for s in (c.serp_top3 or [])])
            serp_urls = ",".join([s["url"] for s in (c.serp_top3 or [])])

            row = asdict(c)
            row.pop("serp_top3", None)
            row.pop("cluster_label", None)

            row["clustered_term"] = c.cluster_label or ""
            row["serp_top3_domains"] = serp_domains
            row["serp_top3_urls"] = serp_urls

            # normalise Nones just for CSV
            row["page_type_status"] = c.page_type_status or ""
            row["page_type_explanation"] = c.page_type_explanation or ""

            writer.writerow(row)

    logger.info(f"[CSV] Wrote {len(candidates)} rows to {file_path}")


# -------------------------------------------------------------------
# High-level orchestration
# -------------------------------------------------------------------

def run_plp_pipeline(
    project_id: str,
    semrush_mcp: SemrushMCPClient,
    database: str,
    target_domain: str,
    page_title: str,
    page_h1: str,
    page_body: str,
    output_csv_path: str,
    debug: bool = False,
) -> None:
    global MCP_CALL_COUNT, LLM_CALL_COUNT
    MCP_CALL_COUNT = 0
    LLM_CALL_COUNT = 0

    if debug:
        logger.setLevel(logging.DEBUG)

    logger.info("[Pipeline] Starting PLP pipeline")

    generic_llm = VertexLLM(project_id=project_id)

    # Stage 1: seeds
    seeds = extract_seeds_from_page(
        llm=generic_llm,
        title=page_title,
        h1=page_h1,
        body_text=page_body,
    )
    if not seeds:
        logger.error("[Pipeline] No seeds extracted, aborting early to save calls")
        return

    # Stage 2: discovery
    candidates = discover_keywords_with_semrush(
        mcp=semrush_mcp,
        seeds=seeds,
        database=database,
        max_per_seed=30,
        max_seeds=5,
    )
    if not candidates:
        logger.warning("[Pipeline] No candidates discovered from keyword discovery phase")
        logger.warning("[Pipeline] This may be due to API errors or no matching keywords found")
        logger.warning("[Pipeline] Attempting to continue with seed keywords only...")

        if seeds:
            logger.info(f"[Pipeline] Using {len(seeds)} seed keywords as fallback candidates")
            fallback_candidates: List[KeywordCandidate] = []
            for seed in seeds:
                try:
                    overview_resp = mcp_call_with_logging(
                        semrush_mcp,
                        "get_keyword_overview",
                        {"keyword": seed.keyword, "database": database},
                    )
                    overview_text = _extract_first_text_content(overview_resp)
                    rows = _parse_semrush_csv_text(overview_text)
                    if rows:
                        row = rows[0]
                        try:
                            fallback_candidates.append(
                                KeywordCandidate(
                                    keyword=row.get("Keyword", seed.keyword),
                                    volume=int(row.get("Search Volume", 0)),
                                    cpc=float(row.get("CPC", 0.0)),
                                    competition=float(row.get("Competition", 0.0)),
                                   # num_results=int(row.get("Number of Results", 0)) if row.get("Number of Results") else None,
                                )
                            )
                            logger.info(f"[Pipeline] Fallback: Successfully created candidate for '{seed.keyword}'")
                        except (ValueError, KeyError) as exc:
                            logger.warning(f"[Pipeline] Could not parse overview data for seed '{seed.keyword}': {exc}")
                    else:
                        logger.warning(f"[Pipeline] No rows parsed from overview for seed '{seed.keyword}'. Text was: {overview_text[:200]}")
                except Exception as exc:
                    logger.warning(f"[Pipeline] Could not get overview for seed '{seed.keyword}': {exc}")
            if fallback_candidates:
                logger.info(f"[Pipeline] Fallback successful: {len(fallback_candidates)} candidates from seed keywords")
                candidates = fallback_candidates
            else:
                logger.error("[Pipeline] Fallback failed: could not get overview data for any seed keywords")
        else:
            logger.error("[Pipeline] No seed keywords available for fallback")

        if not candidates:
            logger.error("[Pipeline] No candidates available even after fallback, aborting")
            return

    # Stage 3: rankings
    enrich_with_current_rank(
        mcp=semrush_mcp,
        candidates=candidates,
        target_domain=target_domain,
        database=database,
        serp_limit=20,
        max_keywords=50,
    )

    # Stage 4: page types
    infer_page_types(
        llm=generic_llm,
        candidates=candidates,
        max_keywords=30,
    )

    # Stage 5: selection
    selected = select_plp_keywords(
        llm=generic_llm,
        candidates=candidates,
        max_candidates=60,
    )
   
    # Stage 6: title/H1 suggestions
    suggestions = suggest_page_title_and_h1(
        llm=generic_llm,
        selected_keywords=selected,
        current_title=page_title,
        current_h1=page_h1,
        max_options=3,
    )
    
    # Stage 7: CSV output
    write_plp_csv(selected, output_csv_path)

    logger.info(
        f"[Pipeline] Done. MCP calls: {MCP_CALL_COUNT}, LLM calls: {LLM_CALL_COUNT}, "
        f"final keywords: {len(selected)}"
    )
