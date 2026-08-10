import os
import time
import json
import hashlib
import re
import requests
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timezone

# ============ CONFIGURATION & CHANNELS ============
DEFAULT_CHANNELS = [
    "bbcafaanoromoo",
    "fana_afaan_oromoo",
    "obn_afaan_oromoo",
    "voaafaanoromoo",
    "tikvahafaanoromoo",
    "OromiaCultureAndTourism",
    "AfaanOromooLearning",
    "tikvahethiopia",
    "fana_brodcasting",
    "ethiopianhistory1",
    "ethio_culture_arts"
]

BATCH_SIZE = 15

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FIREBASE_SERVICE_ACCOUNT = os.getenv("FIREBASE_SERVICE_ACCOUNT")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

total_scanned_all_time = 0
total_saved_all_time = 0
total_runs_all_time = 0

# ============ INITIALIZATION ============

db = None
if FIREBASE_SERVICE_ACCOUNT:
    try:
        raw_json = FIREBASE_SERVICE_ACCOUNT.strip()
        if raw_json.startswith("'") and raw_json.endswith("'"):
            raw_json = raw_json[1:-1]
        service_account_info = json.loads(raw_json)
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("[+] Firebase Admin SDK initialized successfully!")
    except Exception as e:
        print(f"[!] Firebase Initialization Error: {e}")
else:
    print("[!] FIREBASE_SERVICE_ACCOUNT secret is missing!")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("[!] GEMINI_API_KEY secret is missing!")

# ============ DYNAMIC CHANNELS ============

def get_active_channels():
    channels = set(DEFAULT_CHANNELS)
    global db
    if db:
        try:
            docs = db.collection("custom_channels").stream()
            for doc in docs:
                data = doc.to_dict()
                ch_name = data.get("channel_username")
                if ch_name:
                    channels.add(ch_name.strip().replace("@", ""))
        except Exception as e:
            print(f"[!] Custom channels fetch warning: {e}")
    return list(channels)

# ============ SCRAPING REAL DATA ============

def scrape_telegram_web(channel):
    url = f"https://t.me/s/{channel}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            return []
            
        html = response.text
        pattern = r'<div class="tgme_widget_message_text js-message_text"[^>]*>(.*?)</div>'
        matches = re.findall(pattern, html, re.DOTALL)
        
        extracted_posts = []
        for match in matches:
            clean_text = re.sub(r'<[^>]+>', '', match).strip()
            clean_text = re.sub(r'\s+', ' ', clean_text)
            if len(clean_text) > 70:
                extracted_posts.append({
                    "channel": channel,
                    "text": clean_text,
                    "hash": hashlib.md5(clean_text.encode('utf-8')).hexdigest()
                })
        return extracted_posts
    except Exception as e:
        print(f"[!] Error scraping channel {channel}: {e}")
        return []

def deduplicate_posts(posts_list):
    seen_hashes = set()
    unique_posts = []
    for item in posts_list:
        if item["hash"] not in seen_hashes:
            seen_hashes.add(item["hash"])
            unique_posts.append(item["text"])
    return unique_posts

# ============ GEMINI AI (MODEL UPDATED TO 2.5-FLASH) ============

def process_batch_with_gemini(text_batch):
    if not GEMINI_API_KEY:
        print("[!] GEMINI_API_KEY missing!")
        return []

    prompt = f"""You are an expert NLP Dataset Editor for Afaan Oromo and Amharic culture, history, language, and news.

STRICT FILTERING RULES:
1. PRIVACY & SECURITY: Completely DISCARD any text containing personal sensitive information or national security secrets.
2. QUALITY & DUPLICATES: Filter out spam, promotional ads, repeated headlines, or non-informative text.
3. HUMANIZATION: Clean and humanize the Afaan Oromo text ("text_oromo").
4. ACCURATE TRANSLATION: Generate clean, contextually accurate Amharic text ("text_amharic").
5. CATEGORIZATION: Classify into topics ("culture", "history", "language", "folklore", "lifestyle", "news").
6. QUALITY RATING: Assign a score from 0.0 to 10.0.

OUTPUT FORMAT:
Return ONLY a strictly valid JSON Array of objects. No markdown outside JSON.

JSON Schema:
[
  {{
    "text_oromo": "...",
    "text_amharic": "...",
    "topic": "culture",
    "quality_score": 9.2
  }}
]

Input batch for processing:
{json.dumps(text_batch, ensure_ascii=False)}
"""

    try:
        # Gemini 2.5 Flash Model
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.2}
        )
        
        raw_text = response.text.strip()
        raw_text = re.sub(r'^```(?:json)?\s*', '', raw_text, flags=re.IGNORECASE)
        raw_text = re.sub(r'\s*```$', '', raw_text, flags=re.IGNORECASE).strip()
        
        if not raw_text:
            return []
            
        if not raw_text.startswith('['):
            json_match = re.search(r'\[\s*\{.*?\}\s*\]', raw_text, re.DOTALL)
            if json_match:
                raw_text = json_match.group(0)
        
        parsed_json = json.loads(raw_text)
        if isinstance(parsed_json, list):
            return parsed_json
        return []
    except Exception as e:
        print(f"[!] Gemini processing error: {e}")
        return []

# ============ FIRESTORE STORAGE ============

def save_to_firestore(item):
    global db
    if not db:
        print("[!] Firestore DB not initialized!")
        return False

    if not item.get("text_oromo") or not item.get("text_amharic"):
        return False

    try:
        doc_ref = db.collection("cleaned_dataset").document()
        doc_ref.set({
            "text_oromo": str(item.get("text_oromo", "")),
            "text_amharic": str(item.get("text_amharic", "")),
            "topic": str(item.get("topic", "news")),
            "quality_score": float(item.get("quality_score", 8.0)),
            "created_at": firestore.SERVER_TIMESTAMP
        })
        return True
    except Exception as e:
        print(f"[!] Firestore Save Error (Create database in Firebase Console): {e}")
        return False

# ============ TELEGRAM REPORTING ============

def send_telegram_report(run_scanned, run_unique, run_saved, duration, total_channels):
    global total_scanned_all_time, total_saved_all_time, total_runs_all_time
    
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[!] Telegram credentials missing!")
        return

    message = (
        f"📊 *NATA AI Real Dataset Pipeline Report*\n\n"
        f"🌐 *Channels Scanned:* *{total_channels}*\n"
        f"⏱ *Current Run Duration:* *{duration}s*\n"
        f"• Total Web Scanned: *{run_scanned}*\n"
        f"• Unique Filtered: *{run_unique}*\n"
        f"• Cleaned & Saved to Firestore: *{run_saved}*\n\n"
        f"📈 *Cumulative Totals:* \n"
        f"• All-Time Saved JSON: *{total_saved_all_time}*\n"
        f"• Total Runs Completed: *{total_runs_all_time}*\n\n"
        f"🚀 *Status: 100% Pipeline Active*"
    )
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        res = requests.post(url, json=payload, timeout=10)
        print(f"[+] Telegram Report Sent Status Code: {res.status_code}")
    except Exception as e:
        print(f"[!] Telegram report failed: {e}")

# ============ MAIN PIPELINE RUNNER ============

def run_pipeline():
    global total_scanned_all_time, total_saved_all_time, total_runs_all_time
    
    start_time = time.time()
    channels_to_scan = get_active_channels()
    
    all_raw_posts = []
    for channel in channels_to_scan:
        posts = scrape_telegram_web(channel)
        all_raw_posts.extend(posts)
        time.sleep(1)
        
    run_scanned = len(all_raw_posts)
    unique_texts = deduplicate_posts(all_raw_posts)
    run_unique = len(unique_texts)
    
    run_saved = 0
    if unique_texts:
        chunks = [unique_texts[i:i + BATCH_SIZE] for i in range(0, len(unique_texts), BATCH_SIZE)]
        for batch in chunks:
            processed_data = process_batch_with_gemini(batch)
            for item in processed_data:
                if save_to_firestore(item):
                    run_saved += 1
            time.sleep(2)
            
    duration = int(time.time() - start_time)
    
    total_scanned_all_time += run_scanned
    total_saved_all_time += run_saved
    total_runs_all_time += 1
    
    send_telegram_report(run_scanned, run_unique, run_saved, duration, len(channels_to_scan))

if __name__ == "__main__":
    print("[=== NATA AI Dataset Worker Execution Started ===]")
    run_pipeline()       
