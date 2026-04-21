from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from rag.rag_engine import get_rag_answer
import joblib
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# ✅ Load ML model once at startup (model.pkl is in root)
MODEL_FILE = "model.pkl"
try:
    model = joblib.load(MODEL_FILE)
    print("Crop recommendation model loaded successfully.")
except Exception as e:
    model = None
    print(f"⚠️ Could not load model: {e}")

# English → Hindi crop translation dictionary (lowercase keys for safe lookup)
crop_translation = {
    "orange": "संतरा",
    "banana": "केला",
    "cotton": "कपास",
    "maize": "मक्का",
    "chickpea": "चना",
    "rice": "धान",
    "blackgram": "उड़द",
    "watermelon": "तरबूज",
    "pomegranate": "अनार",
    "mothbeans": "मोट",
    "grapes": "अंगूर",
    "mango": "आम",
    "apple": "सेब",
    "jute": "जूट",
    "coffee": "कॉफी",
    "coconut": "नारियल",
    "papaya": "पपीता",
    "kidneybeans": "राजमा",
    "pigeonpeas": "अरहर",
    "mungbean": "मूँग",
    "lentil": "मसूर",
    "muskmelon": "खरबूजा"
}

@app.route('/')
def home(): return render_template('index.html')

@app.route('/about')
def about(): return render_template('about.html')

@app.route('/contact', methods=['GET','POST'])
def contact():
    if request.method == 'POST':
        flash("✅ Thank you for reaching out! We'll get back to you soon.", "success")
        return redirect(url_for('contact'))
    return render_template('contact.html')

@app.route('/query', methods=['GET','POST'])
def query():
    if request.method == 'POST':
        q = request.form.get('question')
        try: ans = get_rag_answer(q)
        except Exception as e: ans = f"⚠️ Error: {e}"
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'question': q, 'answer': ans})
        return render_template('query.html', question=q, answer=ans)
    return render_template('query.html')

@app.route('/recommendation', methods=['GET','POST'])
def recommendation():
    if request.method == 'POST':
        f = request.form
        try:
            X = pd.DataFrame([{
                "N": float(f.get("nitrogen")),
                "P": float(f.get("phosphorus")),
                "K": float(f.get("potassium")),
                "temperature": float(f.get("temperature")),
                "humidity": float(f.get("humidity")),
                "ph": float(f.get("ph")),
                "rainfall": float(f.get("rainfall"))
            }])
            crop = model.predict(X)[0] if model else None

            # Normalize for dictionary lookup: strip spaces, lowercase
            normalized_crop = crop.strip().lower() if crop else None

            # Translate to Hindi if available; fallback to original crop
            hindi_crop = crop_translation.get(normalized_crop, crop)

            conf = round(max(model.predict_proba(X)[0])*100,2) if hasattr(model,"predict_proba") else None
        except Exception as e:
            return render_template('recommendation.html', error=f"⚠️ {e}")
        return render_template('recommendation.html', crop=crop, hindi_crop=hindi_crop, confidence=conf)
    return render_template('recommendation.html')

@app.errorhandler(404)
def not_found(e): return render_template('error.html'), 404

@app.route("/features")
def features(): return render_template("feature.html")
@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/logout')
def logout():
    # Example: clear session and redirect
    session.clear()
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
