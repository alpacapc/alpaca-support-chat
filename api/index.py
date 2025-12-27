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

# 画像URL抽出（フォールバック機能付き）
def extract_image_url(row):
    html_content = str(row['PC用メイン商品説明文'])
    pattern_makeshop = r'https://makeshop-multi-images\.akamaized\.net/alpacapc/shopimages/[^"\s]+'
    match = re.search(pattern_makeshop, html_content)
    if match: return match.group(0)
    pattern_rakuten = r'https://image\.rakuten\.co\.jp/alpacapc/cabinet/item_new2?/[^"\s]+\.jpg'
    match_old = re.search(pattern_rakuten, html_content)
    if match_old: return match_old.group(0)
    code = str(row['独自商品コード'])
    if code and code != 'nan':
         return f"https://makeshop-multi-images.akamaized.net/alpacapc/shopimages/{code}.jpg"
    return ""

if not df.empty:
    df['extracted_image'] = df.apply(extract_image_url, axis=1)
    df['数量'] = pd.to_numeric(df['数量'], errors='coerce').fillna(0)
    df['販売価格'] = pd.to_numeric(df['販売価格'], errors='coerce').fillna(0)
    df = df[df['数量'] > 0]
    df['full_text'] = df['商品名'].astype(str) + " " + df['PC用メイン商品説明文'].astype(str)

@app.route('/api/chat', methods=['POST'])
def chat():
    return jsonify({"reply": "サポート機能"}) 

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_msg = data.get('message', '')
    history = data.get('history', [])
    
    full_context = " ".join([h['content'] for h in history if h['role'] == 'user']) + " " + user_msg
    candidates = df.copy()

    # --- 1. 用途（重いか軽いか）の判定 ---
    heavy_keywords = ['ゲーム', 'Apex', 'VALORANT', '原神', 'マインクラフト', 'マイクラ', 'フォートナイト', 'PUBG', 'スト6', 'FF14', 'ゲーミング', '動画', '編集', 'イラスト']
    is_heavy_task = any(k.lower() in full_context.lower() for k in heavy_keywords)

    # --- 2. 検索・ソートロジック ---
    if 'ノート' in full_context and 'デスクトップ' not in user_msg:
        candidates = candidates[candidates['full_text'].str.contains('ノート', na=False)]
    elif 'デスクトップ' in full_context or 'デスク' in full_context:
         candidates = candidates[candidates['full_text'].str.contains('デスク', na=False)]

    # GPUが必要そうな用途なら「高い順」で高性能機を拾う
    if is_heavy_task:
        gpu_keywords = ['GTX', 'RTX', 'GeForce', 'Radeon', 'グラフィック', 'GPU']
        gpu_candidates = candidates[candidates['full_text'].str.contains('|'.join(gpu_keywords), case=False, na=False)]
        
        if not gpu_candidates.empty:
            candidates = gpu_candidates.sort_values('販売価格', ascending=False)
        else:
            candidates = candidates.sort_values('販売価格', ascending=False)
    else:
        # 事務用などは「安い順」でコスパ機を拾う
        candidates = candidates.sort_values('販売価格', ascending=True)

    # 上位100件をAIに渡す
    top_candidates = candidates.head(100)

    # --- 3. 商品リスト生成 ---
    products_info = ""
    for _, row in top_candidates.iterrows():
        products_info += f"ID:{row['システム商品コード']} | 商品:{row['商品名']} | 価格:{row['販売価格']} | URL:{row['商品ページURL']} | img:{row['extracted_image']} | Spec:{str(row['PC用メイン商品説明文'])[:200]}\n"

    # --- 4. プロンプト（ご指定の内容を完全反映） ---
    rec_prompt = f"""
    あなたは中古パソコンショップ「アルパカPC」のコンシェルジュの「アルパカちゃん」です。
    ユーザーの要望「{user_msg}」に対し、以下のステップで提供された【在庫リスト】の中から**最も適した1～3台**を選んで提案してください。

    【現在の会話状況】
    ユーザーの過去の発言: {full_context}

    【AI判断の特別ルール：ゲームのスペック判定】
    ユーザーが「{user_msg}」というゲームを希望した場合、以下の **2段階判定** を行ってください。

    1. **第一段階：【必要動作環境（Minimum Specs）】を満たしているか？**
       - そのゲームが「起動して、設定を下げれば遊べる」ラインです。
       - これさえ満たしていれば、候補として除外せず、**「設定調整でプレイ可能です」** と添えて提案してください。
       - （例：ApexならGT1030やGTX650等でも「低設定なら可」として提案する）

    2. **第二段階：【推奨動作環境（Recommended Specs）】を満たしているか？**
       - そのゲームが「高画質で快適にヌルヌル動く」ラインです。
       - これを満たす商品（RTX4090等）があれば、**「最高画質で快適です！」** と強く勧めてください。

    **★重要：判断ミスによる機会損失を防ぐこと**
    - 「推奨スペック（理想）」に届かないからといって、「在庫がない」と断るのは禁止です。
    - 「必要スペック（最低限）」さえ満たしていれば、予算を抑えたいお客様にとっては最良の選択肢になり得ます。

    【接客ステップ（状況に応じてスキップ可）】
    
    **Step 1: 用途の確認**
    まだユーザーが「何に使いたいか」を言っていない場合は、商品を提案せず、まず用途を聞いてください。
    （※初期ボタンで選択済みの場合はStep 2へ）

    **Step 1.5: ゲームタイトルの詳細確認**
    もしユーザーが「ゲーム」を選んだ場合、または「ゲームがしたい」と言った場合は、
    **必ず「プレイしたい具体的なゲーム名」を聞いてください。**
    タグ：[CHOICES:マインクラフト,Apex / VALORANT,原神,その他(重いゲーム)]
    
    **Step 2: 商品の提案（最優先）**
    - **Step 1.5でゲーム名を聞いた後、または用途が決まっている場合、すぐに【在庫リスト】から商品を提案してください。**
    - 提案する商品が見つかった場合は、**形状（ノート/デスク）の確認ステップはスキップしてください。**
    - 提案した後に、「こちらのデスクトップでよろしいですか？ モニターセットは必要ですか？」と続けてください。

    **Step 3: 形状・予算確認（商品が決まらない場合のみ）**
    - もし適切な商品が見当たらない場合や、ユーザーが迷っている場合のみ、形状や予算を聞いてください。
    - タグ：[CHOICES:ノートパソコンがいい,デスクトップがいい,どちらでも良い]
    - タグ：[CHOICES:2万円以下,3万円以下,4万円以下,5万円以下,8万円以下,10万円以下,特に決まっていない]

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
       - スペックを盛って話さないでください。
       - できないことは正直に「それは厳しいです」と伝えてください。

    2. **「ぼったくり」の禁止**
       - お客様の用途に対して過剰なスペックの商品は案内しないでください。

    3. **予算とスペックのバランス**
       - 予算内で用途を満たすものがあれば、それを最優先で提案してください。
       - もし予算内で見つからない場合、「ご予算を少し超えてしまいますが…」と前置きした上で提案してください。

    【提案時のルール】
    - リンクは <a href="商品URL" target="_blank" class="product-link">この商品を見る ＞</a> の形式で。
    - 画像は <img src="画像URL" class="product-img"> で表示。
    - ぼったくり禁止。安く済むなら安いものを。
    - おすすめ理由を、お客様の用途に合わせて具体的に添えてください。
    - 商品名は太字にしてください。
    - 在庫がない場合は正直に言う。
    """
    
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(rec_prompt)
    
    return jsonify({"reply": response.text})

if __name__ == '__main__':
    app.run(debug=True)