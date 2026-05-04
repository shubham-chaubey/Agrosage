"""
AgroSage 2.0 — rag/rag_engine.py
==================================
Bulletproof Cloud-Ready RAG engine with Ultimate Auto-Detect.
"""

import os
import re
import numpy as np
import joblib
import google.generativeai as genai
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# ── Load Environment Variables ──────────────────────────────────────────────
load_dotenv()
_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not _API_KEY:
    print("WARNING: GEMINI_API_KEY is not set in environment variables.")

genai.configure(api_key=_API_KEY)

# ============================================================================
# BRAHMASTRA: ULTIMATE AUTO-DETECT FOR CHAT MODEL
# ============================================================================
_chat_model_name = None

try:
    # 1. Google se pucho kaunse models available hain
    _available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    
    if _available_models:
        # 2. Flash model dhoondne ki koshish karo (sabse fast hota hai)
        for name in _available_models:
            if "flash" in name.lower():
                _chat_model_name = name
                break
        
        # 3. Agar flash na mile, toh jo bhi pehla available hai wahi le lo
        if not _chat_model_name:
            _chat_model_name = _available_models[-1]
            
        # 4. SABSE ZAROORI: 'models/' word ko hata do warna 404 error aata hai
        _chat_model_name = _chat_model_name.replace("models/", "")
        print(f"✅ Chat Model Locked: {_chat_model_name}")
    else:
        print("❌ ERROR: Google is not allowing any chat models for this API key.")
        _chat_model_name = "gemini-1.5-flash" # Fallback just in case
except Exception as e:
    print(f"Auto-detect failed: {e}")
    _chat_model_name = "gemini-1.5-flash"

# Model Initialize karein
_gemini_model = genai.GenerativeModel(_chat_model_name)

# Embedding model wahi hai jo script ne theek se chalaya tha
_embed_model_name = "models/gemini-embedding-2" 


# ── Load precomputed embeddings ONCE ──────────────────────────────────────
try:
    _df = joblib.load("embeddings.joblib")
    print("✅ Embeddings loaded successfully.")
except Exception as e:
    _df = None
    print(f"WARNING: Failed to load embeddings: {e}")

# ── In-memory conversation history (trimmed to last 6 turns) ──────────────
conversation_history = []


# ══════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _create_embeddings(text):
    """Call Gemini API using the exact latest embedding model."""
    try:
        result = genai.embed_content(
            model=_embed_model_name,
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        raise RuntimeError(f"Gemini embedding failed: {e}")


def _call_gemini(prompt):
    """Single, non-streaming Gemini call."""
    try:
        response = _gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise RuntimeError(f"Gemini inference failed with model {_chat_model_name}: {e}")


def _markdown_to_html(text):
    """Convert Gemini markdown output to safe HTML for the frontend."""
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*',     r'<em>\1</em>',         text)

    lines = text.split("\n")
    html_lines = []
    in_ul = False
    in_ol = False

    for line in lines:
        s = line.strip()
        if not s:
            continue

        if re.match(r'^(\d+[.)]\s|[a-zA-Z][.)]\s)', s):
            if in_ul: html_lines.append("</ul>"); in_ul = False
            if not in_ol: html_lines.append("<ol>"); in_ol = True
            item = re.sub(r'^(\d+[.)]\s|[a-zA-Z][.)]\s)', '', s)
            html_lines.append(f"<li>{item}</li>")

        elif re.match(r'^[-*•]\s', s):
            if in_ol: html_lines.append("</ol>"); in_ol = False
            if not in_ul: html_lines.append("<ul>"); in_ul = True
            item = re.sub(r'^[-*•]\s', '', s)
            html_lines.append(f"<li>{item}</li>")

        else:
            if in_ol: html_lines.append("</ol>"); in_ol = False
            if in_ul: html_lines.append("</ul>"); in_ul = False
            html_lines.append(line)

    if in_ol: html_lines.append("</ol>")
    if in_ul: html_lines.append("</ul>")

    return "<br>".join(html_lines)


# ══════════════════════════════════════════════════════════════════════════
# PUBLIC FUNCTION — called by app.py
# ══════════════════════════════════════════════════════════════════════════

def get_rag_answer(user_query):
    if _df is None or "embedding" not in _df.columns:
        return "WARNING: Embedding data not available. Please check embeddings.joblib."

    try:
        query_embedding = _create_embeddings(user_query)

        sample_db_embedding = _df["embedding"].iloc[0]
        if len(query_embedding) != len(sample_db_embedding):
            return (f"ERROR: Dimension mismatch! DB dimensions: {len(sample_db_embedding)}, "
                    f"Query dimensions: {len(query_embedding)}. Please update embeddings.joblib.")

        similarities = cosine_similarity(
            np.vstack(_df["embedding"]), [query_embedding]
        ).flatten()
        top_indices = similarities.argsort()[::-1][:3]
        top_data = _df.loc[top_indices]

        context_cols = [c for c in [
            "crop_name", "region", "season", "soil_type",
            "rainfall_mm", "temperature_range_C",
            "recommended_varieties", "text"
        ] if c in top_data.columns]
        retrieved_context = top_data[context_cols].to_json(orient="records")

        memory_context = "\n".join(
            f"{t['role'].capitalize()}: {t['content']}"
            for t in conversation_history[-6:]
        )

        prompt = f"""
You are AgroSage, an AI assistant developed by Shubham Chaubey in 2025 to help farmers in Uttar Pradesh make better crop decisions. Avoid greeting the user multiple times. If someone appreciates you, respond gently and humbly.

Use the following retrieved context from the agricultural database:
{retrieved_context}

Relevant previous conversation:
{memory_context}

Instructions:
- If the user asks how many crops you know about, say: "I have information on many crops. Which one are you interested in?"
- Process all Hindi names of crops properly.
- Your first prioritized language is English. If the user asks in Hindi, reply in Hindi. If Bhojpuri, reply in Bhojpuri.
- Respond line by line, like ChatGPT.
- Never use asterisks (*) for lists. Always use numbers (1., 2., 3.) or letters (a., b., c.).
- Avoid raw Markdown symbols like ** or * in the final output. Use plain text formatting.

Current query:
"{user_query}"
"""

        raw_response = _call_gemini(prompt)
        html_response = _markdown_to_html(raw_response)

        conversation_history.append({"role": "user",      "content": user_query})
        conversation_history.append({"role": "assistant", "content": raw_response})
        if len(conversation_history) > 20:
            del conversation_history[:-20]

        return html_response

    except Exception as e:
        return f"Error processing query: {e}"