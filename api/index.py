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
    pattern_makeshop = r'https://makeshop-multi-images\.akamaized\.net/alpacapc/shopimages/[^"\s]+'
    match = re.search(pattern_makeshop, html_content)
    if match: return match.group(0)
    pattern_rakuten = r'https://image\.rakuten\.co\.jp/alpacapc/cabinet/item_new2?/[^"\s]+\.jpg'
    match_old = re.search(pattern_rakuten, html_content)
    if match_old: return match_old.group(0)
    return ""

if not df.empty:
    df['extracted_image'] = df['PC用メイン商品説明文'].apply(extract_image_url)
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

    # --- 1. ゲーム関連の話題かどうか判定 ---
    game_keywords = ['ゲーム', 'Apex', 'VALORANT', '原神', 'マインクラフト', 'マイクラ', 'フォートナイト', 'PUBG', 'スト6', 'FF14', 'ゲーミング']
    is_gaming_intent = any(k.lower() in full_context.lower() for k in game_keywords)

    if is_gaming_intent:
        # GPUキーワードが含まれる商品を抽出
        gpu_keywords = ['GTX', 'RTX', 'GeForce', 'Radeon', 'グラフィック', 'GPU']
        gaming_candidates = candidates[candidates['full_text'].str.contains('|'.join(gpu_keywords), case=False, na=False)]
        
        if not gaming_candidates.empty:
            # ★リミッター解除！★
            # ゲーミングPCの在庫は80件程度なので、15件に絞らず「最大100件（ほぼ全件）」をAIに渡します。
            # これにより、AIは価格の高い順だけでなく「予算に合うもの」を全商品から探せるようになります。
            gaming_candidates = gaming_candidates.sort_values('販売価格', ascending=False)
            top_candidates = gaming_candidates.head(100) 
        else:
            candidates = candidates.sort_values('販売価格', ascending=False)
            top_candidates = candidates.head(50) # GPU無しでも多めに渡す
            
    else:
        # --- 2. 通常検索 ---
        if 'ノート' in full_context and 'デスクトップ' not in user_msg:
            candidates = candidates[candidates['full_text'].str.contains('ノート', na=False)]
        elif 'デスクトップ' in full_context or 'デスク' in full_context:
             candidates = candidates[candidates['full_text'].str.contains('デスク', na=False)]

        keywords = user_msg.replace('円', '').replace('以下', '').split()
        def calc_score(row):
            text = str(row['full_text'])
            score = 0
            for kw in keywords:
                if kw in text: score += 1
            return score

        candidates['score'] = candidates.apply(calc_score, axis=1)
        # 通常検索も多めに50件渡す
        top_candidates = candidates.sort_values(['score', '販売価格'], ascending=[False, False]).head(50)

    # --- AIへの受け渡し ---
    # 商品情報を少しコンパクトにして、大量のデータを渡せるようにします
    products_info = ""
    for _, row in top_candidates.iterrows():
        products_info += f"""
ID:{row['システム商品コード']} | 商品:{row['商品名']} | 価格:{row['販売価格']} | URL:{row['商品ページURL']} | img:{row['extracted_image']} | Spec:{str(row['PC用メイン商品説明文'])[:200]}
"""

    rec_prompt = f"""
    あなたは中古パソコンショップ「アルパカPC」のコンシェルジュの「アルパカちゃん」です。
    ユーザーの要望「{user_msg}」に対し、以下のステップで提供された【在庫リスト】の中から**最も適した1～3台**を選んで提案してください。


    【現在の会話状況】
    ユーザーの過去の発言: {full_context}

    【AI判断の特別ルール：ゲームについて】
    ユーザーが「{user_msg}」というゲームをプレイしたい場合：
    1. あなたの知識にある「そのゲームの推奨スペック」と、上記の「在庫リストのスペック詳細」を比較してください。
    2. もし「GTX」や「RTX」などが搭載されていて、快適に動くと判断できるなら、自信を持って提案してください。
    3. もし在庫リストの商品がすべてスペック不足（内蔵GPUのみ等）なら、無理に勧めず「申し訳ありません、そのゲームを快適に動かすためのグラフィックボード搭載モデルが、現在在庫切れです」と正直に答えてください。

    【AI判断の特別ルール】
    1. **全件比較**: 上記リストは、高いものから安いものまで幅広く含まれています。**上から順に見るのではなく、リスト全体を見て、お客様の予算と用途に「最もバランスが良いもの」**を選んでください。
    2. **ゲーム用途**: 「Apex」等の場合、推奨スペック（GPU性能）を満たすものの中で、予算内で買える最良のものを提案してください。
       - 予算指定がない場合は、快適に動くラインで「コスパの良いもの」を優先してください。
       - 在庫リストにGTX/RTX搭載機があるなら、必ずそれを優先してください。

    【接客ステップ（この順番を必ず守ってください）】
    
    **Step 1: 用途の確認**
    まだユーザーが「何に使いたいか」を言っていない場合は、商品を提案せず、まず用途を聞いてください。
    （※初期ボタンで選択済みの場合はStep 2へ）

    **Step 1.5: ゲームタイトルの詳細確認（重要）**
    もしユーザーが「ゲーム」を選んだ場合、または「ゲームがしたい」と言った場合は、
    **必ず「プレイしたい具体的なゲーム名」を聞いてください。**
    その際、以下の選択肢タグを出力してください。
    タグ：[CHOICES:マインクラフト,Apex / VALORANT,原神,その他(重いゲーム)]
    
    **Step 1.6: 自由入力されたゲームのスペック判定（AI判断）**
    もしユーザーが具体的なゲーム名（例：「スト6」「FF14」など）を入力した場合は、
    **あなたの知識でそのゲームの推奨スペック（特にGPUの有無）を判断してください。**
    
    * **判定ルール:**
        * 在庫リストの商品（多くはIntel HD/UHD Graphics等の内蔵GPU）で動く軽いゲーム（マイクラ、ブラウザゲーム等）なら提案OK。
        * **専用グラフィックボード（GTX/RTX等）が必須のゲーム（Apex, 原神, FF14等）の場合**、在庫リストにdGPU搭載機がなければ、**「当店の在庫はビジネス向けが中心のため、そのゲームを快適に動かすのは難しいです」と正直に伝えてください。** 無理に売るのは禁止です。

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