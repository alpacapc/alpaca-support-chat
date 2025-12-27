from flask import Flask, request, jsonify
import google.generativeai as genai
import pandas as pd
import os
import re

app = Flask(__name__)

# osモジュールを使って、サーバーの設定からキーを読み込む
GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# CSVデータの読み込み
CSV_PATH = os.path.join(os.path.dirname(__file__), 'item_data.csv')
try:
    df = pd.read_csv(CSV_PATH, encoding='utf-8')
except:
    try:
        df = pd.read_csv(CSV_PATH, encoding='shift_jis')
    except:
        df = pd.DataFrame()

# 画像URL抽出（指定のルール適用）
def extract_image_url(html_content):
    if not isinstance(html_content, str): return ""
    pattern = r'https://image\.rakuten\.co\.jp/alpacapc/cabinet/item_new2?/[^"]+\.jpg'
    match = re.search(pattern, html_content)
    if match: return match.group(0)
    return ""

if not df.empty:
    df['extracted_image'] = df['PC用メイン商品説明文'].apply(extract_image_url)

# --- 1. 通常チャット用 (/api/chat) ---
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_msg = data.get('message', '')
    history = data.get('history', [])
    
    # ここには【サポート用】のプロンプトが入ります（省略なしで設定してください）
    system_prompt = """
    あなたはアルパカPCのマスコット「アルパカくん」です。
    （中略...サポート用のプロンプト）
    """
    
    model = genai.GenerativeModel("gemini-2.5-flash")
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
        - 画像URL: {row['extracted_image']}
        - 商品URL: {row['商品ページURL']}
        - スペックなど: {str(row['PC用メイン商品説明文'])[:200]}...
        ------------------------
        """

    # ★ここをご指摘の内容に合わせて修正しました★
    rec_prompt = f"""
あなたは中古パソコンショップ「アルパカPC」の「パソコン選びコンシェルジュ」の**「アルパカちゃん」**です。
    ユーザーの要望「{user_msg}」に対して、在庫リストから最適な商品を提案してください。

    【在庫リスト】
    {products_info}

    【最重要：役割の分担（サポートへの誘導）】
    あなたは「商品の提案（販売）」のみを行う専門家です。
    **「電源が入らない」「Wi-Fiが繋がらない」「修理したい」といった、トラブル解決やサポートに関する話題は絶対に扱わないでください。**

    もしユーザーの入力がトラブル相談やサポート依頼だった場合は、
    商品は提案せず、以下の定型文で「サポート担当のチャットボット（index.html）」へ誘導してください。
    
    ▼サポート誘導時の回答例
    「申し訳ありません！ボクは商品選びの専門なので、トラブル解決や修理のご相談については、サポート担当のアルパカくんにお繋ぎしますね。
    <br><br>
    👉 <a href="index.html" style="color:#FF9800; font-weight:bold;">サポートチャットへ移動する</a>」

    【最重要：販売員としての行動指針（誠実さ）】
    1. **過剰な演出・嘘の禁止**
       - スペックを盛って話さないでください。「Celeronで爆速」や「グラボなしで最新ゲーム快適」といった嘘は厳禁です。
       - できないことは正直に「それは厳しいです」と伝えてください。

    2. **「ぼったくり」の禁止（予算への誠実さ）**
       - お客様の用途に対して過剰なスペックの商品は案内しないでください。
       - 例：「ネットが見たいだけ」なら、予算5万円と言われても、2万円台で十分な商品があればそちらを優先し、「これならご予算の半分で済みますよ」と提案してください。

    3. **予算とスペックのバランス**
       - 予算内で用途を満たすものがあれば、それを最優先で提案してください。
       - もし予算内で見つからない場合、「ご予算を少し超えてしまいますが…」と前置きした上で、条件を満たす商品を控えめに提案してください（押し売りは禁止）。
       - 該当する商品が在庫リストに全くない場合は、正直に「申し訳ありません、現在ご案内できる在庫がございません」と伝えてください。

    【出力形式のルール】
    - 商品画像がある場合は `<img src="画像URL" class="product-img">` で表示してください。
    - 商品名は太字にしてください。
    - おすすめ理由を、お客様の用途に合わせて具体的に添えてください。
    """
    
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(rec_prompt)
    
    return jsonify({"reply": response.text})

if __name__ == '__main__':
    app.run(debug=True)