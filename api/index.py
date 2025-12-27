from flask import Flask, request, jsonify
import google.generativeai as genai
import pandas as pd
import os
import re

app = Flask(__name__)

# ★ここにAPIキーを入れてください★
GEMINI_API_KEY = "AIzaSyAYi0aaKT_mXFKIMESr-sZ_sFSTCNSk5Bw"
genai.configure(api_key=GEMINI_API_KEY)

# CSVデータの読み込み
CSV_PATH = os.path.join(os.path.dirname(__file__), 'item_data.csv')
try:
    df = pd.read_csv(CSV_PATH, encoding='utf-8')
except:
    try:
        df = pd.read_csv(CSV_PATH, encoding='shift_jis')
    except:
        df = pd.DataFrame() # 読み込み失敗時は空にする

# 画像URL抽出（簡易版）
def extract_image_url(desc):
    if not isinstance(desc, str): return ""
    # 説明文の中からjpg画像を探す
    match = re.search(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+(?:\.jpg|\.png)', desc)
    if match: return match.group(0)
    return ""

if not df.empty and 'extracted_image' not in df.columns:
    df['extracted_image'] = df['PC用メイン商品説明文'].apply(extract_image_url)

# --- 1. 通常チャット用 (/api/chat) ---
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_msg = data.get('message', '')
    history = data.get('history', [])
    
    # ここに今まで設定したシステムプロンプト（アルパカくんの性格など）を入れる
    system_prompt = """
    あなたはアルパカPCのマスコット「アルパカくん」です。
    （中略...今までの長いプロンプトをここに貼り付けてください）
    """
    
    model = genai.GenerativeModel("gemini-2.5-flash")
    # 履歴の変換処理などは省略（実際はhistoryをGemini形式に変換して渡す）
    response = model.generate_content([system_prompt, user_msg])
    
    return jsonify({"reply": response.text})

# --- 2. 商品提案用 (/api/recommend) ---
@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_msg = data.get('message', '')
    
    # 簡易検索ロジック
    candidates = df.copy()
    keywords = user_msg.replace('円', '').replace('以下', '').split()
    
    def calc_score(row):
        text = str(row['商品名']) + str(row['PC用メイン商品説明文'])
        score = 0
        for kw in keywords:
            if kw in text: score += 1
        return score

    if not candidates.empty:
        candidates['score'] = candidates.apply(calc_score, axis=1)
        top_candidates = candidates.sort_values('score', ascending=False).head(5)
    else:
        top_candidates = pd.DataFrame()

    products_info = ""
    for _, row in top_candidates.iterrows():
        products_info += f"""
        - 商品名: {row['商品名']}
        - 価格: {row['販売価格']}円
        - 画像: {row['extracted_image']}
        - URL: {row['商品ページURL']}
        ------------------------
        """

    rec_prompt = f"""
    あなたはパソコン選びコンシェルジュです。以下の在庫リストから、
    ユーザーの要望「{user_msg}」に合う商品を1つ選び、HTML形式（画像付き）で紹介してください。
    
    【在庫リスト】
    {products_info}
    
    【ルール】
    1. 商品画像がある場合は <img src="URL" class="product-img"> で表示。
    2. 商品名は太字。
    3. 在庫がない場合は正直に言う。
    """
    
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(rec_prompt)
    
    return jsonify({"reply": response.text})

# Vercel用のおまじない
if __name__ == '__main__':
    app.run(debug=True)