from flask import Flask, request, jsonify
import google.generativeai as genai
import pandas as pd
import os
import re

app = Flask(__name__)

GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# CSV読み込み
CSV_PATH = os.path.join(os.path.dirname(__file__), 'item_data.csv')
try:
    df = pd.read_csv(CSV_PATH, encoding='utf-8')
except:
    try:
        df = pd.read_csv(CSV_PATH, encoding='shift_jis')
    except:
        df = pd.DataFrame()

# 画像URL抽出
def extract_image_url(html_content):
    if not isinstance(html_content, str): return ""
    pattern = r'https://image\.rakuten\.co\.jp/alpacapc/cabinet/item_new2?/[^"]+\.jpg'
    match = re.search(pattern, html_content)
    if match: return match.group(0)
    return ""

if not df.empty:
    df['extracted_image'] = df['PC用メイン商品説明文'].apply(extract_image_url)
    df['数量'] = pd.to_numeric(df['数量'], errors='coerce').fillna(0)
    df = df[df['数量'] > 0]

# --- 1. 通常チャット用 ---
@app.route('/api/chat', methods=['POST'])
def chat():
    # （ここは変更なし。以前のサポート用コードのままにしてください）
    return jsonify({"reply": "サポート機能"}) 

# --- 2. 商品提案用 (/api/recommend) ---
@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_msg = data.get('message', '')
    history = data.get('history', []) # 会話履歴も受け取る
    
    # 全ての会話履歴を結合して、ユーザーの要望（ノートかデスクか等）を判断する
    full_context = " ".join([h['content'] for h in history if h['role'] == 'user']) + " " + user_msg
    
    # --- 1. 絞り込みロジック (Python側で候補を減らす) ---
    candidates = df.copy()
    
    # 「ノート」指定ならデスクトップを除外
    if 'ノート' in full_context and 'デスクトップ' not in user_msg:
        candidates = candidates[candidates['商品名'].str.contains('ノート', na=False) | candidates['カテゴリーパス'].str.contains('ノート', na=False)]
    
    # 「デスク」指定ならノートを除外
    elif 'デスクトップ' in full_context or 'デスク' in full_context:
         candidates = candidates[candidates['商品名'].str.contains('デスク', na=False) | candidates['カテゴリーパス'].str.contains('デスク', na=False)]

    # キーワードスコアリング
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
        - スペック: {str(row['PC用メイン商品説明文'])[:200]}...
        ------------------------
        """

    # --- 2. AIへの指令 (ヒアリングの順番を定義) ---
    rec_prompt = f"""
    あなたは中古パソコンショップ「アルパカPC」のコンシェルジュです。
    ユーザーの要望「{user_msg}」に対し、以下のステップで接客を行ってください。

    【現在の会話状況】
    ユーザーの過去の発言: {full_context}

    【接客ステップ（この順番を必ず守ってください）】
    
    **Step 1: 用途の確認**
    まだユーザーが「何に使いたいか」を言っていない場合は、商品を提案せず、まず用途を聞いてください。
    （※初期ボタンで選択済みの場合はStep 2へ）

    **Step 2: 形状（ノート/デスク）の確認**
    ユーザーが「ノートパソコン」か「デスクトップ」かを指定していない場合は、以下の選択肢を出して聞いてください。
    タグ：[CHOICES:ノートパソコンがいい,デスクトップがいい,どちらでも良い]

    **Step 3: デスクトップの場合のセット確認**
    もし「デスクトップ」が選ばれた場合、かつ「モニターセット」の希望が不明な場合は、こう聞いてください。
    「モニターやキーボード・マウスのセットは必要ですか？」
    タグ：[CHOICES:セットが必要(すぐ使いたい),本体のみでOK]

    **Step 4: 予算の確認**
    用途と形状が決まっているが、予算が不明な場合は、以下の選択肢を出して聞いてください。
    タグ：[CHOICES:2万円以下,3万円以下,4万円以下,5万円以下,8万円以下,10万円以下,特に決まっていない]

    **Step 5: 商品の提案**
    上記の全て（用途・形状・予算）がある程度わかったら、以下の在庫リストから最適な商品を提案してください。

    【在庫リスト（現在庫あり）】
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

    【提案時のルール】
    - リンクは <a href="商品URL" target="_blank" class="product-link">この商品を見る ＞</a> の形式で。
    - 画像は <img src="画像URL" class="product-img"> で表示。
    - ぼったくり禁止。安く済むなら安いものを。
    - おすすめ理由を、お客様の用途に合わせて具体的に添えてください。
    - 商品名は太字にしてください。
    - 在庫がない場合は正直に言う。
    - サポート相談が来たら index.html へ誘導。
    """
    
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(rec_prompt)
    
    return jsonify({"reply": response.text})

if __name__ == '__main__':
    app.run(debug=True)