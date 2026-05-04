import os
import time
import pandas as pd
import joblib
import google.generativeai as genai
from dotenv import load_dotenv

# 1. API Key Setup
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("❌ ERROR: .env file mein GEMINI_API_KEY nahi mili!")
    exit()

genai.configure(api_key=API_KEY)

# 2. AUTO-DETECT AVAILABLE MODEL
print("🔍 Checking available embedding models...")
best_model = None
for m in genai.list_models():
    if 'embedContent' in m.supported_generation_methods:
        best_model = m.name
        # Naye gemini models ko priority dete hain
        if "gemini-embedding" in m.name or "text-embedding-004" in m.name:
            break

if not best_model:
    print("❌ ERROR: Aapke API key par koi embedding model available nahi hai!")
    exit()

print(f"✅ Auto-selected model: {best_model}")

# 3. File Paths
JSON_FILE_PATH = "data.json"
OUTPUT_FILE = "embeddings.joblib"

def load_data():
    if not os.path.exists(JSON_FILE_PATH):
        print(f"❌ ERROR: {JSON_FILE_PATH} file nahi mili!")
        exit()
    
    print(f"✅ Loading data from {JSON_FILE_PATH}...")
    df = pd.read_json(JSON_FILE_PATH)
    return df

def create_embeddings(df):
    embeddings = []
    total_rows = len(df)
    
    print(f"⏳ Generating embeddings for {total_rows} rows...")
    print("Isme thoda waqt lag sakta hai, kripya intezaar karein...\n")

    for index, row in df.iterrows():
        row_text = " ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
        
        success = False
        retries = 0
        
        # SMART RETRY LOOP: Agar 429 aaye toh 60 sec wait karke wapas try karega
        while not success and retries < 3:
            try:
                result = genai.embed_content(
                    model=best_model,
                    content=row_text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
                print(f"[{index + 1}/{total_rows}] Embedded ✅")
                success = True
                time.sleep(2) # Normal 2 second ka pause limit se bachne ke liye
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    print(f"⚠️ Limit hit at row {index + 1}. Code 60 seconds ke liye pause le raha hai...")
                    time.sleep(60) # 1 minute ka cooldown
                    retries += 1
                else:
                    print(f"❌ Failed at row {index + 1} with error: {error_msg}")
                    break # Agar koi aur error hai toh loop tod do
        
        if not success:
            embeddings.append(None)

    return embeddings

if __name__ == "__main__":
    df = load_data()
    df['embedding'] = create_embeddings(df)
    
    # Drop rows that failed
    df = df.dropna(subset=['embedding'])
    
    print(f"\n💾 Saving {len(df)} embeddings to {OUTPUT_FILE}...")
    joblib.dump(df, OUTPUT_FILE)
    print(f"🎉 BOOM! Aapki nayi embeddings.joblib taiyaar hai ({best_model} ke sath)!")