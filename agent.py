import os
import json
import re
import hashlib
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

TODAY = datetime.now().strftime("%Y-%m-%d")

MODEL_NAME = "openai/gpt-oss-120b"

MEMORY_FILE = Path("memory.json")

llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0
)

search_tool = TavilySearch(
    max_results=2
)


# ============================================================
# MEMORY DATABASE
# ============================================================

def create_empty_memory():
    return {
        "version": "6.4",
        "created_at": datetime.now().isoformat(),
        "conversations": [],
        "research_history": []
    }


def load_memory():
    if not MEMORY_FILE.exists():
        data = create_empty_memory()
        save_memory(data)
        return data

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "conversations" not in data:
            data["conversations"] = []

        if "research_history" not in data:
            data["research_history"] = []

        data["version"] = "6.4"

        return data

    except Exception:
        print("Warning: memory file could not be read.")
        print("Creating a new memory file.")

        data = create_empty_memory()
        save_memory(data)

        return data


def save_memory(data):
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"Memory save error: {e}")


memory = load_memory()


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def clean_text(text):
    if text is None:
        return ""

    text = str(text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def safe_llm(prompt, keep_format=False):
    try:
        response = llm.invoke(prompt)

        text = response.content if hasattr(response, "content") else str(response)

        if keep_format:
            return str(text).strip()

        return clean_text(text)

    except Exception as e:
        print(f"LLM error: {e}")
        return ""


def normalize(text):
    return re.sub(r"[^a-z0-9\s]", " ", text.lower())


def tokenize(text):
    words = normalize(text).split()

    stopwords = {
        "what", "is", "are", "was", "were",
        "the", "a", "an", "and", "or",
        "to", "of", "in", "on", "for",
        "how", "why", "when", "where",
        "tell", "me", "about", "my",
        "do", "does", "did", "can",
        "could", "would", "should",
        "we", "i", "you", "your",
        "before", "previous", "discussed"
    }

    return {
        word
        for word in words
        if len(word) > 2 and word not in stopwords
    }


def domain_from_url(url):
    try:
        domain = urlparse(url).netloc.lower()

        if domain.startswith("www."):
            domain = domain[4:]

        return domain

    except Exception:
        return ""


# ============================================================
# MEMORY DETECTION
# ============================================================

def is_previous_question(question):
    q = question.lower()

    patterns = [
        "what was my previous question",
        "what was my last question",
        "what did i ask before",
        "what did i ask previously",
        "what was the previous question",
        "what was the last question"
    ]

    return any(pattern in q for pattern in patterns)


def is_first_question(question):
    q = question.lower()

    patterns = [
        "what was my first question",
        "what did i ask first",
        "what was the first thing i asked",
        "what was my first query"
    ]

    return any(pattern in q for pattern in patterns)


def is_previous_answer(question):
    q = question.lower()

    patterns = [
        "what was your previous answer",
        "what did you answer before",
        "what was your last answer",
        "repeat your previous answer",
        "what did you tell me before"
    ]

    return any(pattern in q for pattern in patterns)


def is_memory_request(question):
    q = question.lower()

    memory_terms = [
        "previous",
        "before",
        "earlier",
        "last question",
        "first question",
        "previous question",
        "previous answer",
        "what did we discuss",
        "what did we talk about",
        "remember",
        "memory",
        "discussed before",
        "talked about before"
    ]

    return any(term in q for term in memory_terms)


def contains_web_request(question):
    q = question.lower()

    web_terms = [
        "latest",
        "current",
        "today",
        "recent",
        "news",
        "2026",
        "2025",
        "new",
        "update",
        "updates",
        "search",
        "internet",
        "online",
        "research",
        "according to",
        "what is happening"
    ]

    return any(term in q for term in web_terms)


# ============================================================
# ROUTING
# ============================================================

def route_question(question):

    if is_previous_question(question):
        return "MEMORY"

    if is_first_question(question):
        return "MEMORY"

    if is_previous_answer(question):
        return "MEMORY"

    memory_request = is_memory_request(question)
    web_request = contains_web_request(question)

    if memory_request and web_request:
        return "HYBRID"

    if memory_request:
        return "HYBRID"

    return "WEB"


# ============================================================
# MEMORY SEARCH
# ============================================================

def get_relevant_memories(question, limit=5):

    conversations = memory.get("conversations", [])

    if not conversations:
        return []

    query_words = tokenize(question)

    scored = []

    for item in conversations:

        text = " ".join([
            item.get("user_question", ""),
            item.get("understood_question", ""),
            item.get("answer", "")
        ])

        memory_words = tokenize(text)

        overlap = len(query_words.intersection(memory_words))

        score = overlap

        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []

    for score, item in scored:

        if score > 0:
            results.append(item)

        if len(results) >= limit:
            break

    return results


def get_latest_memory():

    conversations = memory.get("conversations", [])

    if not conversations:
        return None

    return conversations[-1]


# ============================================================
# MEMORY ANSWERS
# ============================================================

def answer_memory_question(question):

    conversations = memory.get("conversations", [])

    if not conversations:
        return "I don't have any previous conversation stored yet."

    if is_previous_question(question):

        previous = conversations[-1]

        return (
            f'Your previous question was: '
            f'"{previous.get("user_question", "")}"'
        )

    if is_first_question(question):

        first = conversations[0]

        return (
            f'Your first question was: '
            f'"{first.get("user_question", "")}"'
        )

    if is_previous_answer(question):

        previous = conversations[-1]

        answer = previous.get("answer", "")

        return (
            "Your previous answer was:\n\n"
            + answer
        )

    relevant = get_relevant_memories(question)

    if not relevant:
        return "I could not find a relevant previous conversation."

    output = "Relevant previous conversation:\n\n"

    for index, item in enumerate(relevant, 1):

        output += (
            f"--- Memory {index} ---\n"
            f"Question: {item.get('user_question', '')}\n"
            f"Answer: {item.get('answer', '')[:1200]}\n\n"
        )

    return output


# ============================================================
# UNDERSTAND USER QUESTION
# ============================================================

def understand_question(question):

    prompt = f"""
You are the query-understanding component of an Internet Research Agent.

Today's date is {TODAY}.

User question:
{question}

Determine the user's actual research intent.

Return ONLY one clear sentence describing what the user wants to know.

Do not answer the question.
Do not invent information.
"""

    result = safe_llm(prompt)

    if not result:
        return question

    return result


# ============================================================
# EXTRACT TOPIC FROM MEMORY
# ============================================================

def extract_memory_topic(question, relevant_memories):

    if not relevant_memories:
        return ""

    memory_text = ""

    for item in relevant_memories:

        memory_text += (
            f"""
Previous question:
{item.get("user_question", "")}

Previous understanding:
{item.get("understood_question", "")}

Previous answer:
{item.get("answer", "")[:1500]}

"""
        )

    prompt = f"""
You are the topic extraction component of an intelligent research agent.

The user is asking:
{question}

Here is relevant previous conversation:
{memory_text}

Identify the MAIN TOPIC from the previous conversation that should be researched on the Internet.

Examples:

If memory discusses RAG and the user asks:
"Tell me what we discussed about RAG before"
return:
Retrieval-Augmented Generation (RAG)

If memory discusses Python agents and the user asks:
"What did we discuss about Python agents?"
return:
Python AI agents

Return ONLY the topic.
"""

    topic = safe_llm(prompt)

    return clean_text(topic)


# ============================================================
# SEARCH QUERY GENERATION
# ============================================================

def create_search_query(question, understood_question, topic=""):

    context = ""

    if topic:
        context = f"""
Relevant remembered topic:
{topic}

Use this remembered topic to understand what should be researched.
"""

    prompt = f"""
You are an expert search-query planner.

Today's date is {TODAY}.

Original user question:
{question}

Understood intent:
{understood_question}

{context}

Create ONE strong Internet search query.

Rules:
- Search for the actual information requested.
- Do not search the literal phrase "what did we discuss before".
- If a remembered topic is provided, search that topic.
- Prefer official documentation, research papers, company announcements,
  government sources, universities, and reputable publications.
- Include the current year when freshness matters.
- Do not add unnecessary words.
- Do not answer the question.

Return ONLY the search query.
"""

    query = safe_llm(prompt)

    if not query:
        if topic:
            return f"{topic} latest developments {TODAY}"

        return question

    query = query.strip('"').strip("'")

    return query


# ============================================================
# SEARCH
# ============================================================

def run_search(query):

    try:

        result = search_tool.invoke({
            "query": query
        })

        return extract_search_results(result)

    except Exception as e:

        print(f"Search error: {e}")

        return []


def extract_search_results(result):

    results = []

    if isinstance(result, dict):

        possible_lists = [
            result.get("results"),
            result.get("data"),
            result.get("items")
        ]

        for candidate in possible_lists:

            if isinstance(candidate, list):

                results.extend(candidate)

                break

    elif isinstance(result, list):

        results = result

    cleaned = []

    for item in results:

        if not isinstance(item, dict):
            continue

        title = clean_text(
            item.get("title")
            or item.get("name")
            or ""
        )

        url = clean_text(
            item.get("url")
            or item.get("link")
            or ""
        )

        content = clean_text(
            item.get("content")
            or item.get("snippet")
            or item.get("description")
            or ""
        )

        if not title and not url:
            continue

        cleaned.append({
            "title": title,
            "url": url,
            "content": content,
            "domain": domain_from_url(url),
            "published_date": clean_text(
                item.get("published_date")
                or item.get("published")
                or item.get("date")
                or ""
            )
        })

    return cleaned


# ============================================================
# SOURCE QUALITY
# ============================================================

PRIMARY_DOMAINS = {
    "openai.com",
    "anthropic.com",
    "deepmind.google",
    "ai.google",
    "google.com",
    "microsoft.com",
    "research.microsoft.com",
    "meta.com",
    "research.facebook.com",
    "ibm.com",
    "apple.com",
    "nvidia.com",
    "arxiv.org"
}

HIGH_AUTHORITY_DOMAINS = {
    "reuters.com",
    "bbc.com",
    "apnews.com",
    "nature.com",
    "science.org",
    "technologyreview.com",
    "wired.com"
}


def is_primary_domain(domain):

    if domain in PRIMARY_DOMAINS:
        return True

    if domain.endswith(".gov"):
        return True

    if domain.endswith(".edu"):
        return True

    return False


def is_high_authority(domain):

    if domain in HIGH_AUTHORITY_DOMAINS:
        return True

    return False


def freshness_score(source):

    date_text = source.get("published_date", "")

    if not date_text:
        text = " ".join([
            source.get("title", ""),
            source.get("content", ""),
            source.get("url", "")
        ])

        match = re.search(
            r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b",
            text
        )

        if match:
            date_text = match.group(0)

    if not date_text:
        return 0

    try:

        match = re.search(
            r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})",
            date_text
        )

        if not match:
            year_match = re.search(r"\b(20\d{2})\b", date_text)

            if not year_match:
                return 0

            year = int(year_match.group(1))

            if year >= datetime.now().year:
                return 5

            if year == datetime.now().year - 1:
                return 3

            return 1

        date_value = datetime(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3))
        )

        days_old = (datetime.now() - date_value).days

        if days_old <= 30:
            return 5

        if days_old <= 180:
            return 4

        if days_old <= 365:
            return 3

        if days_old <= 1095:
            return 2

        return 1

    except Exception:
        return 0


def authority_score(source):

    domain = source.get("domain", "")

    if is_primary_domain(domain):
        return 10

    if is_high_authority(domain):
        return 8

    if domain.endswith(".gov"):
        return 10

    if domain.endswith(".edu"):
        return 9

    return 4


def relevance_score(source, question):

    question_words = tokenize(question)

    source_text = " ".join([
        source.get("title", ""),
        source.get("content", "")
    ])

    source_words = tokenize(source_text)

    overlap = len(question_words.intersection(source_words))

    if overlap >= 8:
        return 10

    if overlap >= 5:
        return 8

    if overlap >= 3:
        return 6

    if overlap >= 1:
        return 4

    return 1


def score_source(source, question):

    authority = authority_score(source)
    relevance = relevance_score(source, question)
    freshness = freshness_score(source)

    total = authority + relevance + freshness

    return {
        **source,
        "authority_score": authority,
        "relevance_score": relevance,
        "freshness_score": freshness,
        "total_score": total,
        "primary": is_primary_domain(source.get("domain", ""))
    }


def rank_sources(sources, question):

    scored = [
        score_source(source, question)
        for source in sources
    ]

    scored.sort(
        key=lambda x: x["total_score"],
        reverse=True
    )

    return scored


# ============================================================
# SOURCE DIVERSITY
# ============================================================

def calculate_source_diversity(sources):

    domains = set()

    for source in sources:

        domain = source.get("domain", "")

        if domain:
            domains.add(domain)

    return len(domains)


def research_quality_score(sources):

    if not sources:
        return 0

    strong = 0
    primary = 0

    for source in sources:

        if source["authority_score"] >= 8:
            strong += 1

        if source["primary"]:
            primary += 1

    diversity = calculate_source_diversity(sources)

    score = (
        strong * 2
        + primary * 2
        + diversity
    )

    return score


# ============================================================
# FALLBACK SEARCH
# ============================================================

def fallback_search(question, topic=""):

    if topic:

        query = (
            f'"{topic}" '
            f'(official OR research OR announcement '
            f'OR documentation OR paper) '
            f'{TODAY}'
        )

    else:

        query = (
            f'"{question}" '
            f'(official OR "official blog" '
            f'OR "press release" '
            f'OR "research paper" '
            f'OR conference OR journal)'
        )

    print("\nFallback search:")
    print(query)

    return run_search(query)


# ============================================================
# EVIDENCE VERIFICATION
# ============================================================

def build_evidence_text(sources):

    text = ""

    for index, source in enumerate(sources, 1):

        text += (
            f"""
SOURCE {index}
Title: {source.get("title", "")}
URL: {source.get("url", "")}
Domain: {source.get("domain", "")}
Authority score: {source.get("authority_score", 0)}
Relevance score: {source.get("relevance_score", 0)}
Freshness score: {source.get("freshness_score", 0)}

Content:
{source.get("content", "")[:2500]}

--------------------------------------------------
"""
        )

    return text


def verify_evidence(question, sources):

    if not sources:

        return {
            "status": "NONE",
            "supported_sources": 0,
            "unsupported_risk": "HIGH",
            "conflict": False,
            "verification": "No sources were retrieved."
        }

    evidence = build_evidence_text(sources)

    prompt = f"""
You are the evidence-verification component of a research agent.

User question:
{question}

Retrieved sources:
{evidence}

Determine whether the retrieved sources actually provide evidence
for answering the user's question.

Important:
- Do NOT assume a source supports a claim merely because its title
  contains similar words.
- Do NOT treat search-result titles as proof.
- A source must contain meaningful evidence.
- Multiple pages from the same domain are NOT automatically independent evidence.
- If evidence is weak, say so.
- If sources disagree, identify the conflict.
- Do not invent evidence.

Return ONLY valid JSON in exactly this structure:

{{
  "status": "STRONG" or "MODERATE" or "WEAK" or "NONE",
  "supported_sources": number,
  "unsupported_risk": "LOW" or "MEDIUM" or "HIGH",
  "conflict": true or false,
  "verification": "short explanation"
}}
"""

    raw = safe_llm(prompt)

    if not raw:
        return {
            "status": "WEAK",
            "supported_sources": 0,
            "unsupported_risk": "HIGH",
            "conflict": False,
            "verification": "Evidence verification failed."
        }

    try:

        match = re.search(
            r"\{.*\}",
            raw,
            re.DOTALL
        )

        if not match:
            raise ValueError("No JSON found.")

        data = json.loads(match.group(0))

        return {
            "status": data.get("status", "WEAK"),
            "supported_sources": int(
                data.get("supported_sources", 0)
            ),
            "unsupported_risk": data.get(
                "unsupported_risk",
                "HIGH"
            ),
            "conflict": bool(
                data.get("conflict", False)
            ),
            "verification": data.get(
                "verification",
                ""
            )
        }

    except Exception:

        return {
            "status": "WEAK",
            "supported_sources": 0,
            "unsupported_risk": "HIGH",
            "conflict": False,
            "verification": "Could not reliably parse evidence verification."
        }


# ============================================================
# CONFIDENCE SCORE
# ============================================================

def calculate_confidence(
    sources,
    verification
):

    if not sources:
        return 0, "LOW"

    score = 0

    supported = verification.get(
        "supported_sources",
        0
    )

    status = verification.get(
        "status",
        "NONE"
    )

    unsupported_risk = verification.get(
        "unsupported_risk",
        "HIGH"
    )

    conflict = verification.get(
        "conflict",
        False
    )

    # Evidence support
    if supported >= 3:
        score += 4

    elif supported >= 2:
        score += 3

    elif supported >= 1:
        score += 2

    # Authority
    strong_sources = sum(
        1
        for source in sources
        if source["authority_score"] >= 8
    )

    if strong_sources >= 2:
        score += 2

    elif strong_sources == 1:
        score += 1

    # Primary sources
    primary_count = sum(
        1
        for source in sources
        if source["primary"]
    )

    if primary_count >= 2:
        score += 2

    elif primary_count == 1:
        score += 1

    # Diversity
    diversity = calculate_source_diversity(sources)

    if diversity >= 3:
        score += 2

    elif diversity >= 2:
        score += 1

    # Verification status
    if status == "STRONG":
        score += 2

    elif status == "MODERATE":
        score += 1

    # Penalties
    if unsupported_risk == "MEDIUM":
        score -= 2

    elif unsupported_risk == "HIGH":
        score -= 4

    if conflict:
        score -= 2

    score = max(0, min(10, score))

    if score >= 8:
        level = "HIGH"

    elif score >= 5:
        level = "MEDIUM"

    else:
        level = "LOW"

    return score, level


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(
    question,
    understood_question,
    sources,
    verification,
    confidence_score,
    confidence_level,
    relevant_memories=None,
    hybrid=False
):

    evidence = build_evidence_text(sources)

    memory_context = ""

    if relevant_memories:

        memory_context = "\nRELEVANT PREVIOUS CONVERSATION:\n"

        for item in relevant_memories:

            memory_context += (
                f"""
Previous question:
{item.get("user_question", "")}

Previous answer:
{item.get("answer", "")[:1500]}

"""
            )

    prompt = f"""
You are the final answer generator for a professional
AI Internet Research Agent.

Today's date:
{TODAY}

User question:
{question}

Understood question:
{understood_question}

Confidence:
{confidence_score}/10 ({confidence_level})

Evidence verification:
{json.dumps(verification, indent=2)}

{memory_context}

WEB SOURCES:
{evidence}

Your job is to provide the most accurate answer possible.

STRICT RULES:

1. Use retrieved evidence for factual claims.

2. Never invent:
   - products
   - companies
   - models
   - releases
   - dates
   - statistics
   - research findings
   - announcements

3. If the evidence is insufficient, clearly say:
   "The available sources do not provide enough reliable evidence
   to confirm this."

4. Do not turn speculation into fact.

5. Do not treat a weak source as authoritative.

6. Do not count duplicate pages from the same domain as independent
   confirmation.

7. If sources conflict, explain the conflict.

8. If the question is simple and factual, answer directly instead
   of unnecessarily refusing.

9. If previous conversation is provided, use it only as context.
   Do not claim that the Internet sources prove what the user
   previously discussed.

10. Use citations in this format:
    [Source 1]
    [Source 2]

11. Only cite sources that actually support the statement.

12. Do not create citations for sources that do not support the claim.

13. Keep the answer professional and useful.

Preferred structure:

## Answer

Direct answer.

## Key Points

Important points supported by evidence.

## Evidence Quality

Brief explanation of source reliability and confidence.

## Sources

[Source 1] ...
[Source 2] ...

If evidence is weak, keep the answer concise and clearly explain
the uncertainty.
"""

    answer = safe_llm(prompt, keep_format=True)

    if not answer:

        return (
            "I could not generate a reliable answer from the "
            "available evidence."
        )

    return answer


# ============================================================
# SAVE CONVERSATION
# ============================================================

def save_conversation(
    question,
    understood_question,
    route,
    answer,
    sources
):

    conversation_id = hashlib.md5(
        f"{datetime.now().isoformat()}-{question}".encode(
            "utf-8"
        )
    ).hexdigest()[:12]

    record = {
        "id": conversation_id,
        "timestamp": datetime.now().isoformat(),
        "user_question": question,
        "understood_question": understood_question,
        "route": route,
        "answer": answer,
        "sources": [
            {
                "title": source.get("title", ""),
                "url": source.get("url", ""),
                "domain": source.get("domain", "")
            }
            for source in sources[:5]
        ]
    }

    memory["conversations"].append(record)

    memory["conversations"] = (
        memory["conversations"][-100:]
    )

    save_memory(memory)


# ============================================================
# SAVE RESEARCH HISTORY
# ============================================================

def save_research_history(
    question,
    search_query,
    confidence,
    sources
):

    record = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "search_query": search_query,
        "confidence": confidence,
        "sources": [
            {
                "title": source.get("title", ""),
                "url": source.get("url", ""),
                "domain": source.get("domain", "")
            }
            for source in sources[:5]
        ]
    }

    memory["research_history"].append(record)

    memory["research_history"] = (
        memory["research_history"][-100:]
    )

    save_memory(memory)


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(sources):

    print("\nSources:")

    if not sources:
        print("  No sources found.")
        return

    for index, source in enumerate(sources, 1):

        print(
            f"\n  [{index}] {source.get('title', 'Untitled')}"
        )

        print(
            f"      {source.get('url', '')}"
        )

        print(
            f"      Domain: {source.get('domain', '')}"
        )

        print(
            f"      Authority: "
            f"{source.get('authority_score', 0)}/10 | "
            f"Relevance: "
            f"{source.get('relevance_score', 0)}/10 | "
            f"Freshness: "
            f"{source.get('freshness_score', 0)}/5"
        )


# ============================================================
# HYBRID CONTEXT
# ============================================================

def prepare_hybrid_context(question):

    relevant_memories = get_relevant_memories(
        question,
        limit=5
    )

    if not relevant_memories:
        return "", []

    topic = extract_memory_topic(
        question,
        relevant_memories
    )

    return topic, relevant_memories


# ============================================================
# MAIN RESEARCH PIPELINE (terminal mode)
# ============================================================

def research(question):

    route = route_question(question)

    print(f"\nRoute: {route}")

    if route == "MEMORY":

        answer = answer_memory_question(question)

        print("\n" + answer)

        return

    understood_question = understand_question(
        question
    )

    print(
        f"\nUnderstood question:\n"
        f"{understood_question}"
    )

    relevant_memories = []
    topic = ""

    if route == "HYBRID":

        topic, relevant_memories = (
            prepare_hybrid_context(question)
        )

        if relevant_memories:

            print(
                f"\nRelevant memories found: "
                f"{len(relevant_memories)}"
            )

        if topic:

            print(
                f"Remembered topic: {topic}"
            )

    search_query = create_search_query(
        question,
        understood_question,
        topic
    )

    print(
        f"\nSearch query:\n"
        f"{search_query}"
    )

    print("\nSearching the Internet...")

    raw_sources = run_search(search_query)

    sources = rank_sources(
        raw_sources,
        understood_question
    )

    quality = research_quality_score(
        sources
    )

    print(
        f"\nInitial sources found: "
        f"{len(sources)}"
    )

    print(
        f"Initial research quality score: "
        f"{quality}"
    )

    if quality < 5:

        fallback_sources = fallback_search(
            understood_question,
            topic
        )

        sources.extend(
            fallback_sources
        )

        unique_sources = {}

        for source in sources:

            url = source.get("url", "")

            if url:
                unique_sources[url] = source

        sources = list(
            unique_sources.values()
        )

        sources = rank_sources(
            sources,
            understood_question
        )

    sources = sources[:6]

    print("\nVerifying evidence...")

    verification = verify_evidence(
        understood_question,
        sources
    )

    print(
        f"\nVerification: "
        f"{verification.get('status', 'UNKNOWN')}"
    )

    print(
        f"Supported sources: "
        f"{verification.get('supported_sources', 0)}"
    )

    print(
        f"Unsupported-claim risk: "
        f"{verification.get('unsupported_risk', 'UNKNOWN')}"
    )

    print(
        f"Conflict detected: "
        f"{verification.get('conflict', False)}"
    )

    confidence_score, confidence_level = (
        calculate_confidence(
            sources,
            verification
        )
    )

    print(
        f"\nResearch confidence: "
        f"{confidence_score}/10 "
        f"({confidence_level})"
    )

    display_sources(sources)

    print("\nGenerating answer...")

    answer = generate_answer(
        question=question,
        understood_question=understood_question,
        sources=sources,
        verification=verification,
        confidence_score=confidence_score,
        confidence_level=confidence_level,
        relevant_memories=relevant_memories,
        hybrid=(route == "HYBRID")
    )

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)

    print(answer)

    print("\n" + "=" * 70)

    save_conversation(
        question,
        understood_question,
        route,
        answer,
        sources
    )

    save_research_history(
        question,
        search_query,
        confidence_score,
        sources
    )


# ============================================================
# STARTUP
# ============================================================

def print_startup():

    print("=" * 70)
    print("AI-POWERED INTELLIGENT INTERNET RESEARCH AGENT V6.4")
    print("=" * 70)

    print(f"Today's date: {TODAY}")

    print("\nPersistent memory:")
    print(f" - {MEMORY_FILE.resolve()}")

    print(
        f"\nStored conversations: "
        f"{len(memory.get('conversations', []))}"
    )

    print(
        f"Stored research records: "
        f"{len(memory.get('research_history', []))}"
    )

    print("\nType 'exit' to stop the agent.")
    print("=" * 70)


# ============================================================
# PROGRAM LOOP (only runs when you start agent.py directly)
# ============================================================

if __name__ == "__main__":

    print_startup()

    while True:

        try:

            question = input("\nYou: ").strip()

            if not question:
                continue

            if question.lower() in {"exit", "quit", "q"}:

                print("\nAgent stopped.")

                break

            research(question)

        except KeyboardInterrupt:

            print("\n\nAgent stopped.")

            break

        except Exception as e:

            print(f"\nUnexpected error: {e}")

            print(
                "The agent is still running. "
                "Try another question."
            )