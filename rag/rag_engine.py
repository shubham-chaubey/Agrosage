"""
AgroSage 2.0 — rag/rag_engine.py
==================================
RAG engine using BGE-M3 (Ollama) for embeddings + Gemini for inference.

QUOTA-SAFE design (same as original AgroSage):
   - One single generate_content() call per user query (NO streaming).
   - conversation_history trimmed to last 6 turns max.
   - Model instance created ONCE at module load (not per request).
   - No generate_content_async, no streaming, no repeated calls.
   - You are AgroSage, an AI assistant developed by Shubham Chaubey in 2025 for empowering the farmers.
   - Please greate professionally to users (in Hindi,English both).
"""

import google.generativeai as genai
import numpy as np
import joblib
import requests
import re
import os
from sklearn.metrics.pairwise import cosine_similarity

# ── Configure Gemini ONCE (module-level, not per-request) ──────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_API_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=_API_KEY)

# Use gemini-1.5-flash — higher free-tier quota, plenty capable for RAG Q&A
_gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# ── Load precomputed embeddings ONCE ──────────────────────────────────────
try:
    _df = joblib.load("embeddings.joblib")
    print("Embeddings loaded successfully.")
except Exception as e:
    _df = None
    print(f"WARNING: Failed to load embeddings: {e}")

# ── In-memory conversation history (trimmed to last 6 turns) ──────────────
conversation_history = []


# ══════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _create_embeddings(text_list):
    """Call local Ollama BGE-M3 to embed texts. NOT Gemini, so no quota cost."""
    try:
        r = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": "bge-m3", "input": text_list},
            timeout=30
        )
        r.raise_for_status()
        return r.json()["embeddings"]
    except Exception as e:
        raise RuntimeError(f"Ollama embedding failed: {e}")


def _call_gemini(prompt):
    """Single, non-streaming Gemini call. ONE call = ONE quota unit."""
    try:
        response = _gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise RuntimeError(f"Gemini inference failed: {e}")


def _markdown_to_html(text):
    """Convert Gemini markdown output to safe HTML for the frontend."""
    # Bold and italic
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*',     r'<em>\1</em>',         text)

    lines = text.split("\n")
    html_lines = []
    in_ul = False
    in_ol = False

    for line in lines:
        s = line.strip()

        if re.match(r'^(\d+[.)]\s|[a-zA-Z][.)]\s)', s):
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            if not in_ol:
                html_lines.append("<ol>")
                in_ol = True
            item = re.sub(r'^(\d+[.)]\s|[a-zA-Z][.)]\s)', '', s)
            html_lines.append(f"<li>{item}</li>")

        elif re.match(r'^[-*•]\s', s):
            if in_ol:
                html_lines.append("</ol>")
                in_ol = False
            if not in_ul:
                html_lines.append("<ul>")
                in_ul = True
            item = re.sub(r'^[-*•]\s', '', s)
            html_lines.append(f"<li>{item}</li>")

        else:
            if in_ol:
                html_lines.append("</ol>")
                in_ol = False
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            html_lines.append(line)

    if in_ol:
        html_lines.append("</ol>")
    if in_ul:
        html_lines.append("</ul>")

    return "<br>".join(html_lines)


# ══════════════════════════════════════════════════════════════════════════
# PUBLIC FUNCTION — called by app.py
# ══════════════════════════════════════════════════════════════════════════

def get_rag_answer(user_query):
    """
    Full RAG pipeline:
      1. Embed the query via Ollama BGE-M3 (free, no Gemini quota used).
      2. Retrieve top-3 similar rows from embeddings.joblib.
      3. Build a rich prompt with context + trimmed conversation history.
      4. Call Gemini ONCE — this is the ONLY quota-consuming step.
      5. Convert response to HTML, update history, return.
    """
    if _df is None or "embedding" not in _df.columns:
        return "WARNING: Embedding data not available. Please check embeddings.joblib."

    try:
        # Step 1: Embed the query (Ollama, NOT Gemini)
        query_embedding = _create_embeddings([user_query])[0]

        # Step 2: Cosine similarity, top 3
        similarities = cosine_similarity(
            np.vstack(_df["embedding"]), [query_embedding]
        ).flatten()
        top_indices = similarities.argsort()[::-1][:3]
        top_data = _df.loc[top_indices]

        # Step 3: Build context from retrieved rows
        context_cols = [c for c in [
            "crop_name", "region", "season", "soil_type",
            "rainfall_mm", "temperature_range_C",
            "recommended_varieties", "text"
        ] if c in top_data.columns]
        retrieved_context = top_data[context_cols].to_json(orient="records")

        # Step 4: Build conversation memory (last 6 turns only)
        memory_context = "\n".join(
            f"{t['role'].capitalize()}: {t['content']}"
            for t in conversation_history[-6:]
        )

        # Step 5: Construct prompt
        prompt = f"""
You are AgroSage, an AI assistant developed by Shubham Chaubey in 2025 to help farmers in Uttar Pradesh make better crop decisions. Avoid greeting the user multiple times. If someone appreciates you, respond gently and humbly.

Use the following retrieved context from the agricultural database:
{retrieved_context}

Relevant previous conversation:
{memory_context}

Instructions:
- If the user asks how many crops you know about, say: "I have information on many crops. Which one are you interested in?"
- Do not greet repeatedly. Follow this strictly.
- Process all Hindi names of crops properly if a farmer asks by Hindi name.
- Your first prioritized language is English (remember it).
- If the user speaks in Bhojpuri, reply in Bhojpuri.
- Respond line by line, like ChatGPT.
- Use English by default unless the user switches to another language.
- If the user asks in Hindi, reply in Hindi.
- Prioritize clarity, usefulness, and trust.
- If user asks in points then give the answer in points, otherwise give a detailed answer.
- When listing items (e.g., diseases, benefits, steps), format them clearly:
  Example:
  1. Soil Type: Loamy
  2. Temperature: 23-45 degrees C
  3. Sowing Time: July-August

  For sub-points, use:
  a. Early Blight
  b. Leaf Blight
  c. Leaf Curl Virus

- Never use asterisks (*) for lists. Always use numbers (1., 2., 3.) or letters (a., b., c.).
- Avoid raw Markdown symbols like ** or * in the final output. Use plain text formatting.
- Keep the tone professional, helpful, and easy to read for farmers.

Current query:
"{user_query}"
"""

        # Step 6: ONE Gemini call (the only quota-consuming operation)
        raw_response = _call_gemini(prompt)

        # Step 7: Convert to HTML
        html_response = _markdown_to_html(raw_response)

        # Step 8: Update history, keep trimmed to avoid memory growth
        conversation_history.append({"role": "user",      "content": user_query})
        conversation_history.append({"role": "assistant", "content": raw_response})
        if len(conversation_history) > 20:
            del conversation_history[:-20]

        return html_response

    except Exception as e:
        return f"Error processing query: {e}"
