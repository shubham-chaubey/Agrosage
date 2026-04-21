"""
AgroSage 2.0 — app.py (Fixed)
- Adds missing profile_update endpoint so templates using url_for('profile_update') work.
- Keeps single POST RAG behavior (no streaming).
- Assumes auth blueprint, rag.rag_engine.get_rag_answer, model.pkl, embeddings.joblib exist.
"""

from flask import (
    Flask, render_template, request, jsonify,
    redirect, url_for, flash, session
)
import joblib
import pandas as pd

from rag.rag_engine import get_rag_answer
from auth import (
    auth_bp, init_db, register_teardown,
    login_required, get_user_by_id, get_all_users,
    update_profile, save_query
)

# ── App setup ─────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'agrosage_secret_2025'

# Register auth blueprint and teardown
app.register_blueprint(auth_bp)
register_teardown(app)

# Initialise DB on startup
with app.app_context():
    init_db()

# ── Load ML model once ────────────────────────────────────────────────────
try:
    model = joblib.load("model.pkl")
    print("Crop recommendation model loaded.")
except Exception as e:
    model = None
    print(f"WARNING: Could not load model: {e}")

# ── Crop translation dict ─────────────────────────────────────────────────
crop_translation = {
    "orange":       "संतरा",
    "banana":       "केला",
    "cotton":       "कपास",
    "maize":        "मक्का",
    "chickpea":     "चना",
    "rice":         "धान",
    "blackgram":    "उड़द",
    "watermelon":   "तरबूज",
    "pomegranate":  "अनार",
    "mothbeans":    "मोट",
    "grapes":       "अंगूर",
    "mango":        "आम",
    "apple":        "सेब",
    "jute":         "जूट",
    "coffee":       "कॉफी",
    "coconut":      "नारियल",
    "papaya":       "पपीता",
    "kidneybeans":  "राजमा",
    "pigeonpeas":   "अरहर",
    "mungbean":     "मूँग",
    "lentil":       "मसूर",
    "muskmelon":    "खरबूजा",
}


# ══════════════════════════════════════════════════════════════════════════
# MAIN ROUTES
# ══════════════════════════════════════════════════════════════════════════

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/features')
def features():
    return render_template('feature.html')


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        flash("Thank you for reaching out! We'll get back to you soon.", "success")
        return redirect(url_for('contact'))
    return render_template('contact.html')


# ── Query (RAG) ─────────────────────────────────────────────────────────
@app.route('/query', methods=['GET', 'POST'])
def query():
    """
    GET  → render the chat page
    POST → run RAG, return JSON  { question, answer }

    One POST = One RAG call. No streaming.
    """
    if request.method == 'POST':
        q = request.form.get('question', '').strip()
        if not q:
            return jsonify({'question': '', 'answer': 'Please enter a question.'}), 400

        try:
            ans = get_rag_answer(q)
        except Exception as e:
            ans = f"Error: {e}"

        # Save to DB if user is logged in
        user_id = session.get('user_id')
        if user_id:
            try:
                save_query(user_id, q, ans)
            except Exception:
                pass  # Non-critical — don't break the response

        return jsonify({'question': q, 'answer': ans})

    return render_template('query.html')


# ── Crop Recommendation ──────────────────────────────────────────────────
@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    if request.method == 'POST':
        f = request.form
        try:
            X = pd.DataFrame([{
                "N":           float(f.get("nitrogen")),
                "P":           float(f.get("phosphorus")),
                "K":           float(f.get("potassium")),
                "temperature": float(f.get("temperature")),
                "humidity":    float(f.get("humidity")),
                "ph":          float(f.get("ph")),
                "rainfall":    float(f.get("rainfall")),
            }])
            crop           = model.predict(X)[0] if model else None
            normalized     = crop.strip().lower() if crop else None
            hindi_crop     = crop_translation.get(normalized, crop)
            conf           = round(max(model.predict_proba(X)[0]) * 100, 2) \
                             if (model and hasattr(model, "predict_proba")) else None
        except Exception as e:
            return render_template('recommendation.html', error=f"Error: {e}")

        return render_template(
            'recommendation.html',
            crop=crop, hindi_crop=hindi_crop, confidence=conf
        )
    return render_template('recommendation.html')


# ── Profile (view + update) ──────────────────────────────────────────────
@app.route('/profile', methods=['GET'])
@login_required
def profile():
    """
    Renders profile page. The profile update form in templates may post to
    the separate /profile/update endpoint (profile_update) — see below.
    """
    user_id = session.get('user_id')
    user    = get_user_by_id(user_id)
    return render_template('profile.html', user=user)


@app.route('/profile/update', methods=['POST'])
@login_required
def profile_update():
    """
    Separate endpoint for profile updates. Templates that call
    url_for('profile_update') will now resolve correctly.
    """
    user_id = session.get('user_id')
    fullname = request.form.get('fullname', '').strip()
    location = request.form.get('location', '').strip()

    if fullname or location:
        try:
            update_profile(user_id, fullname, location)
            # update session display name if provided
            if fullname:
                session['fullname'] = fullname
            flash("Profile updated successfully.", "success")
        except Exception as e:
            flash(f"Could not update profile: {e}", "error")
    else:
        flash("No changes submitted.", "info")

    return redirect(url_for('profile'))


# ── Admin ────────────────────────────────────────────────────────────────
@app.route('/admin')
@login_required
def admin():
    # Simple guard: only user with id=1 is admin
    if session.get('user_id') != 1:
        flash("Admin access only.", "error")
        return redirect(url_for('home'))
    users = get_all_users()
    return render_template('admin.html', users=users)


# ── Error handlers ────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return render_template('error.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html'), 500


# ══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    app.run(debug=True)
