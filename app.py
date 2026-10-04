"""VeriSearch AI - Streamlit frontend.
Backend file must be saved as agent.py in the same folder.
Run: streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="VeriSearch AI", page_icon="🔎", layout="wide")

import agent as core  # noqa: E402

# ---------------- Styling ----------------
st.markdown("""
<style>
.hero {padding:1.2rem 1.5rem;border-radius:14px;
  background:linear-gradient(135deg,#1e3a8a,#7c3aed);color:white;margin-bottom:1rem}
.hero h1 {margin:0;font-size:2rem}
.hero p {margin:.2rem 0 0;opacity:.9}
.badge {display:inline-block;padding:2px 10px;border-radius:999px;
  font-size:.75rem;font-weight:600;color:white;margin-right:6px}
.b-WEB{background:#2563eb}.b-MEMORY{background:#7c3aed}.b-HYBRID{background:#0d9488}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero"><h1>🔎 VeriSearch AI</h1>
<p>Verified Internet Research Agent — sources, evidence checks, confidence scores & memory</p></div>
""", unsafe_allow_html=True)


# ---------------- Pipeline ----------------
def run_pipeline(question, status):
    route = core.route_question(question)
    status.write(f"Route: **{route}**")

    if route == "MEMORY":
        return {"route": route, "answer": core.answer_memory_question(question),
                "sources": [], "understood": "", "query": "", "topic": "",
                "verification": None, "score": None, "level": None}

    understood = core.understand_question(question)
    status.write(f"Understood: {understood}")

    topic, memories = "", []
    if route == "HYBRID":
        topic, memories = core.prepare_hybrid_context(question)
        if topic:
            status.write(f"Remembered topic: {topic}")

    query = core.create_search_query(question, understood, topic)
    status.write(f"Searching: `{query}`")
    sources = core.rank_sources(core.run_search(query), understood)

    if core.research_quality_score(sources) < 5:
        status.write("Quality low, running fallback search...")
        sources.extend(core.fallback_search(understood, topic))
        unique = {s["url"]: s for s in sources if s.get("url")}
        sources = core.rank_sources(list(unique.values()), understood)
    sources = sources[:6]

    status.write("Verifying evidence...")
    verification = core.verify_evidence(understood, sources)
    score, level = core.calculate_confidence(sources, verification)

    status.write("Generating answer...")
    answer = core.generate_answer(
        question=question, understood_question=understood, sources=sources,
        verification=verification, confidence_score=score,
        confidence_level=level, relevant_memories=memories,
        hybrid=(route == "HYBRID"))

    core.save_conversation(question, understood, route, answer, sources)
    core.save_research_history(question, query, score, sources)
    return {"route": route, "answer": answer, "sources": sources,
            "understood": understood, "query": query, "topic": topic,
            "verification": verification, "score": score, "level": level}


def render_result(r, key):
    st.markdown(f'<span class="badge b-{r["route"]}">{r["route"]}</span>',
                unsafe_allow_html=True)

    if r["score"] is not None:
        c1, c2, c3, c4 = st.columns(4)
        v = r["verification"]
        c1.metric("Confidence", f'{r["score"]}/10', r["level"])
        c2.metric("Evidence", v["status"])
        c3.metric("Supported sources", v["supported_sources"])
        c4.metric("Unsupported risk", v["unsupported_risk"])
        if v["conflict"]:
            st.warning("Sources conflict with each other.")
        st.progress(r["score"] / 10)

    st.markdown(r["answer"])

    st.download_button("⬇️ Download answer (.md)", r["answer"],
                       file_name="verisearch_answer.md", key=f"dl_{key}")

    if r["understood"] or r["query"]:
        with st.expander("🧠 How the agent reasoned"):
            st.write(f"**Understood question:** {r['understood']}")
            st.write(f"**Search query:** `{r['query']}`")
            if r["topic"]:
                st.write(f"**Remembered topic:** {r['topic']}")
            if r["verification"]:
                st.write(f"**Verification note:** {r['verification']['verification']}")

    if r["sources"]:
        with st.expander(f"📚 Sources ({len(r['sources'])})", expanded=False):
            for i, s in enumerate(r["sources"], 1):
                with st.container(border=True):
                    star = " ⭐ primary" if s.get("primary") else ""
                    st.markdown(f"**[{i}] [{s['title'] or 'Untitled'}]({s['url']})**{star}")
                    st.caption(s["domain"])
                    a, b, c = st.columns(3)
                    a.metric("Authority", f'{s["authority_score"]}/10')
                    b.metric("Relevance", f'{s["relevance_score"]}/10')
                    c.metric("Freshness", f'{s["freshness_score"]}/5')
                    if s["content"]:
                        st.write(s["content"][:300] + "...")


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Dashboard")
    convs = core.memory.get("conversations", [])
    hist = core.memory.get("research_history", [])
    c1, c2 = st.columns(2)
    c1.metric("Conversations", len(convs))
    c2.metric("Research runs", len(hist))
    st.caption(f"Model: {core.MODEL_NAME}  \nDate: {core.TODAY}")

    st.subheader("💡 Try these")
    examples = ["Latest developments in AI agents",
                "What was my previous question?",
                "Tell me what we discussed before about AI"]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex

    if st.button("🗑️ Clear memory", use_container_width=True):
        core.memory["conversations"] = []
        core.memory["research_history"] = []
        core.save_memory(core.memory)
        st.session_state.messages = []
        st.rerun()

# ---------------- Chat ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.write(m["content"])
        else:
            render_result(m["result"], i)

question = st.chat_input("Ask anything...") or st.session_state.pop("pending", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.status("Researching...", expanded=True) as status:
            try:
                result = run_pipeline(question, status)
                status.update(label="Done", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Error", state="error")
                st.error(f"Something went wrong: {e}")
                st.stop()
        render_result(result, len(st.session_state.messages))
    st.session_state.messages.append({"role": "assistant", "result": result})
