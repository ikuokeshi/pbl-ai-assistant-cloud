import streamlit as st
import openai
from openai import AzureOpenAI
import io
import os
from dotenv import load_dotenv
import base64
from PIL import Image
import hashlib
import pandas as pd
import numpy as np  # 統計処理用

# 環境変数を読み込み（.envファイルがある場合）
load_dotenv()

# ================================================
# 1. ページの基本設定
# ================================================
st.set_page_config(
    page_title="PBL AI アシスタント",
    page_icon="🤖",
    layout="wide"
)

# ================================================
# 2. タイトルと説明
# ================================================
st.title("🤖 PBL AI アシスタント")
st.markdown("---")
st.markdown("### 使い方")
st.markdown("""
1. 左サイドバーでAzure OpenAIの設定を入力
2. 参考資料をアップロード（任意）
3. 質問や相談内容を入力
4. 「AIに相談する」ボタンをクリック
""")
st.markdown("---")

# ================================================
# 4. Azure OpenAI設定の読み込み
# ================================================

# ================================================
# 3. ユーティリティ関数
# ================================================

def safe_key(text):
    """文字列から安全なStreamlit keyを生成"""
    return hashlib.md5(text.encode()).hexdigest()[:8]

# ================================================
# プロンプト改善コーチング機能
# ================================================
def evaluate_prompt_quality(prompt):
    """プロンプトの品質を評価（0-100点）- 改善版"""
    score = 30  # 基本点
    
    # 長さチェック（より詳細）
    words = prompt.split()
    if len(words) >= 8:
        score += 10
    if len(words) >= 15:
        score += 10
    if len(words) >= 25:
        score += 5
    
    # 具体性チェック（より幅広く）
    analysis_words = ['分析', '比較', '傾向', '相関', '予測', '特徴', '要因', '関係', '変動', 'パターン']
    if any(word in prompt for word in analysis_words):
        score += 15
    
    # 質問の明確性（「どのように」などがなくても、詳細な指示があれば評価）
    deep_words = ['なぜ', 'どのように', 'いつ', 'どこで', 'なに', 'どんな', 'どの程度']
    if any(word in prompt for word in deep_words):
        score += 10
    
    # データへの言及
    if any(word in prompt for word in ['データ', 'CSV', 'ファイル', '情報']):
        score += 10
    
    # 出力形式の指定（より柔軟に）
    format_words = ['3つ', '4つ', '5つ', 'ポイント', 'ランキング', 'グラフ', '表', '観点', '項目']
    if any(word in prompt for word in format_words):
        score += 10
    
    # 具体的な指標への言及
    metric_words = ['平均', '最大', '最小', '標準偏差', '変動', '範囲', '数値']
    if any(word in prompt for word in metric_words):
        score += 10
    
    # 詳細な説明要求
    detail_words = ['詳しく', '具体的', '詳細', '根拠', '理由']
    if any(word in prompt for word in detail_words):
        score += 5
    
    return min(score, 100)

def suggest_prompt_improvements(prompt, score):
    """プロンプト改善提案を生成 - 改善版"""
    suggestions = []
    
    if score < 60:
        suggestions.append("🎯 **構造化**: 「以下の3つの観点で分析してください：1. ○○ 2. ○○ 3. ○○」")
    
    if not any(word in prompt for word in ['3つ', '4つ', '5つ', 'ポイント', '観点']):
        suggestions.append("📊 **形式指定**: 「3つのポイントで」「4つの観点で」など具体的な構造を指定")
    
    if not any(word in prompt for word in ['なぜ', 'どのように', '要因', '理由']):
        suggestions.append("❓ **深い分析**: 「なぜそうなるのか」「どのような要因で」など原因分析を追加")
    
    if not any(word in prompt for word in ['提案', 'アドバイス', '洞察', '示唆']):
        suggestions.append("💡 **実用性**: 「実用的な洞察も提案してください」を追加")
    
    if '。' not in prompt or prompt.count('。') < 2:
        suggestions.append("📝 **文章構造**: 複数の文に分けて、より詳細な指示を記述")
    
    return suggestions

def show_prompt_coaching(prompt, question_type):
    """プロンプトコーチング表示（データ分析専用）"""
    # データ分析・統計の場合のみコーチング機能を表示
    if question_type == "データ分析・統計" and prompt.strip():
        score = evaluate_prompt_quality(prompt)
        
        # スコア表示
        col1, col2 = st.columns([1, 3])
        with col1:
            if score >= 80:
                st.success(f"🎉 プロンプト品質: {score}点")
            elif score >= 60:
                st.info(f"👍 プロンプト品質: {score}点")
            else:
                st.warning(f"💡 プロンプト品質: {score}点")
        
        with col2:
            if score < 70:
                suggestions = suggest_prompt_improvements(prompt, score)
                if suggestions:
                    with st.expander("💡 データ分析のプロンプト改善ヒント"):
                        for suggestion in suggestions:
                            st.write(suggestion)
    # データ分析・統計以外では何も表示しない

def load_azure_config():
    """Azure OpenAI設定を読み込む（Cloud対応版）"""
    config = {}
    
    # Streamlit Secrets から読み込み（優先）
    try:
        if hasattr(st, 'secrets') and 'azure_openai' in st.secrets:
            config["azure_endpoint"] = st.secrets["azure_openai"]["endpoint"]
            config["api_key"] = st.secrets["azure_openai"]["api_key"]
            config["deployment_name"] = st.secrets["azure_openai"]["deployment_name"]
            config["api_version"] = st.secrets["azure_openai"]["api_version"]
            
            if config["azure_endpoint"] and config["api_key"]:
                st.sidebar.success("✅ Streamlit Secretsから設定を読み込みました")
                return config
    except Exception:
        pass
    
    # 環境変数から読み込み
    try:
        config["azure_endpoint"] = os.getenv("AZURE_OPENAI_ENDPOINT")
        config["api_key"] = os.getenv("AZURE_OPENAI_API_KEY")
        config["deployment_name"] = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-deployment")
        config["api_version"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        if config["azure_endpoint"] and config["api_key"]:
            st.sidebar.success("✅ 環境変数から設定を読み込みました")
            return config
    except Exception:
        pass
    
    return None

# 設定を読み込み
config = load_azure_config()

st.sidebar.header("🔧 Azure OpenAI 設定")

if config:
    # 設定ファイルから読み込めた場合
    st.sidebar.info("設定ファイルから自動読み込み中...")
    azure_endpoint = config["azure_endpoint"]
    api_key = config["api_key"] 
    deployment_name = config["deployment_name"]
    api_version = config["api_version"]
    
    # 設定の確認表示（APIキーは隠す）
    st.sidebar.text(f"エンドポイント: {azure_endpoint[:30]}...")
    st.sidebar.text(f"APIキー: {'*' * 20}")
    st.sidebar.text(f"デプロイメント: {deployment_name}")
    
else:
    # 設定ファイルが見つからない場合は手動入力
    st.sidebar.warning("⚠️ 設定ファイルが見つかりません。手動で入力してください。")
    
    # Azure OpenAIの接続情報を入力
    azure_endpoint = st.sidebar.text_input(
        "Azure OpenAI エンドポイント",
        placeholder="https://your-resource.openai.azure.com/",
        help="Azure PortalのAzure OpenAIリソースから取得"
    )

    api_key = st.sidebar.text_input(
        "APIキー",
        type="password",
        placeholder="あなたのAPIキーを入力",
        help="Azure PortalのAzure OpenAIリソース → キーとエンドポイント"
    )

    deployment_name = st.sidebar.text_input(
        "デプロイメント名",
        value="gpt-4o-deployment",
        help="Azure OpenAI Studioで作成したデプロイメント名"
    )

    # API バージョン（通常は変更不要）
    api_version = st.sidebar.selectbox(
        "APIバージョン",
        ["2024-02-15-preview", "2023-12-01-preview"],
        index=0,
        help="通常は最新版で問題ありません"
    )

# サイドバーの学習ガイド部分を修正
st.sidebar.markdown("---")

# データ分析・統計選択時のみ学習ガイドを表示
current_question_type = st.session_state.get('current_question_type', '')
if current_question_type == "データ分析・統計":
    st.sidebar.header("🎓 データ分析学習ガイド")
    
    # セッション状態で分析回数を記録
    if 'analysis_count' not in st.session_state:
        st.session_state.analysis_count = 0
    
    # デバッグ表示を削除
    # st.sidebar.write(f"**デバッグ**: 分析回数 = {st.session_state.analysis_count}")
    
    # 学習レベル表示
    level = min(st.session_state.analysis_count // 2 + 1, 5)
    progress = (st.session_state.analysis_count % 2) / 2.0 if level < 5 else 1.0
    st.sidebar.write(f"**現在のレベル**: {level}/5")
    st.sidebar.progress(progress)
    
    # レベル別ガイド
    level_guide = {
        1: "🌱 初心者: 基本的なデータ分析を体験",
        2: "🌿 初級: 具体的な質問でより詳しい分析", 
        3: "🌳 中級: 複数の観点から多角的分析",
        4: "🌲 上級: 予測や提案を含む高度な分析",
        5: "🏆 エキスパート: データサイエンティストレベル"
    }
    
    st.sidebar.info(level_guide[level])

else:
    # データ分析・統計以外の場合は一般的なガイドを表示
    st.sidebar.header("💡 AI活用ガイド")
    st.sidebar.info("選択した分野に応じて、具体的で明確な質問をしてください。")
    
# ================================================
# 5. Azure OpenAI クライアントの初期化
# ================================================
def create_azure_client():
    """Azure OpenAI クライアントを作成する関数"""
    if not azure_endpoint or not api_key:
        return None
    
    try:
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version
        )
        return client
    except Exception as e:
        st.sidebar.error(f"接続エラー: {str(e)}")
        return None

# ================================================
# 6. ファイルアップロード機能（複数画像対応）
# ================================================
st.header("📁 参考資料のアップロード")

# セッション状態の初期化
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}
if 'uploaded_content' not in st.session_state:
    st.session_state.uploaded_content = ""

# 複数ファイルアップロード
uploaded_files = st.file_uploader(
    "参考資料（複数ファイル対応）",
    type=['txt', 'pdf', 'docx', 'csv', 'jpg', 'jpeg', 'png'],
    accept_multiple_files=True,
    help="テキストファイル、CSV、PDF、Word、画像ファイルに対応。複数選択可能"
)

# アップロードされたファイルを処理
# ファイル処理部分にCSVの処理を追加
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_key = uploaded_file.name
        if file_key not in st.session_state.uploaded_files:
            try:
                file_type = uploaded_file.type
                
                if file_type == "text/plain":
                    # テキストファイルの場合（既存）
                    content = str(uploaded_file.read(), "utf-8")
                    st.session_state.uploaded_files[file_key] = {
                        'type': 'text',
                        'content': content,
                        'file_object': None
                    }
                    st.session_state.uploaded_content += f"\n\n=== {file_key} ===\n{content}"
                
                elif file_type == "text/csv" or file_key.endswith('.csv'):
                # CSVファイルの場合（AI理解強化版）
                    df = pd.read_csv(uploaded_file)
    
                    # より詳細で分析しやすい形式で情報を作成
                    csv_info = f"""
                === CSVファイル: {file_key} ===

                【データ概要】
                - 行数: {len(df)}行のデータ
                - 列数: {len(df.columns)}列
                - 列名: {', '.join(df.columns)}

                【実際のデータ内容】
                {df.to_string(index=False)}

                【数値データの統計情報】
                {df.describe().to_string()}

                このデータを詳細に分析してください。各数値について具体的に言及し、
                傾向やパターンを特定してください。
                """
                    
                    st.session_state.uploaded_files[file_key] = {
                        'type': 'csv',
                        'content': csv_info,
                        'dataframe': df,
                        'file_object': None
                    }
                    st.session_state.uploaded_content += csv_info
                    
                elif file_type in ["image/jpeg", "image/jpg", "image/png"]:
                    # 画像ファイルの場合（既存）
                    file_bytes = uploaded_file.read()
                    uploaded_file.seek(0)
                    
                    st.session_state.uploaded_files[file_key] = {
                        'type': 'image',
                        'content': f"[画像ファイル: {file_key}]",
                        'file_object': uploaded_file,
                        'file_bytes': file_bytes
                    }
                    
                else:
                    # その他のファイル（既存）
                    st.session_state.uploaded_files[file_key] = {
                        'type': 'other',
                        'content': f"[ファイル: {file_key}]",
                        'file_object': None
                    }
                    
            except Exception as e:
                st.error(f"ファイル読み込みエラー（{file_key}）: {str(e)}")

# アップロード済みファイルの表示
if st.session_state.uploaded_files:
    st.subheader("📄 アップロード済みファイル")
    
    cols = st.columns([3, 1])
    with cols[0]:
        st.info(f"📎 {len(st.session_state.uploaded_files)}個のファイル")
    with cols[1]:
        if st.button("🗑️ 全ファイル削除", type="secondary", key="delete_all_files"):
            st.session_state.uploaded_files = {}
            st.session_state.uploaded_content = ""
            st.rerun()
    
    # ファイル一覧表示
    for file_key, file_info in st.session_state.uploaded_files.items():
        with st.expander(f"📎 {file_key} ({file_info['type']})"):
        
            if file_info['type'] == 'csv':
                # CSVの場合は特別な表示
                df = file_info['dataframe']
                st.write(f"**データ概要**: {len(df)}行 × {len(df.columns)}列")
                st.write(f"**列名**: {', '.join(df.columns)}")
            
                # データのプレビュー
                st.write("**データプレビュー（最初の5行）**:")
                st.dataframe(df.head())
            
            # 基本統計
            elif file_info['type'] == 'csv':
                df = file_info['dataframe']
                st.write(f"**データ概要**: {len(df)}行 × {len(df.columns)}列")
                st.write(f"**列名**: {', '.join(df.columns)}")
    
                # データのプレビュー
                st.write("**データプレビュー（最初の5行）**:")
                st.dataframe(df.head())
    
                # 基本統計（CSVの場合のみ）
                if len(df.select_dtypes(include=[np.number]).columns) > 0:
                    st.write("**基本統計**:")
                    st.dataframe(df.describe())
            
            if file_info['type'] == 'image':
                # 画像の場合は表示
                image = Image.open(io.BytesIO(file_info['file_bytes']))
                st.image(image, caption=file_key, use_column_width=True)
                st.caption(f"画像サイズ: {image.size[0]} x {image.size[1]} ピクセル")
                st.info(f"💡 質問で画像を参照するには: 「{file_key}の画像を見て...」")
                
            elif file_info['type'] == 'text':
                # テキストの場合は内容表示
                st.text_area("内容", file_info['content'], height=150, disabled=True)
                
            # 個別削除ボタン
            if st.button(f"🗑️ {file_key}を削除", key=f"delete_file_{safe_key(file_key)}"):
                del st.session_state.uploaded_files[file_key]
                # テキストコンテンツからも削除
                st.session_state.uploaded_content = ""
                for k, v in st.session_state.uploaded_files.items():
                    if v['type'] == 'text':
                        st.session_state.uploaded_content += f"\n\n=== {k} ===\n{v['content']}"
                st.rerun()

# ファイル参照ガイド
if st.session_state.uploaded_files:
    image_files = [k for k, v in st.session_state.uploaded_files.items() if v['type'] == 'image']
    if image_files:
        st.info(f"🎯 **画像参照のコツ**: テキストファイル内で「スポット名({image_files[0]})」形式で画像を参照")

# ================================================
# 7. メイン機能：質問入力とAI応答（画像入力対応）
# ================================================
st.header("💬 AI に相談・質問")

# 質問の種類を選択
question_type = st.selectbox(
    "分析の種類を選んでください",
    [
        "データ分析・統計",
        "一般的な質問",
        "観光プラン作成",
        "数学の問題解決",
        "就活・自己PR作成",
        "その他"
    ]
)

# セッション状態に保存
st.session_state.current_question_type = question_type

# デバッグ表示を削除
# st.write(f"🔧 選択中: {question_type}")
# st.write(f"🔧 保存確認: {st.session_state.get('current_question_type', '保存失敗')}")



# 質問入力エリア
col1, col2 = st.columns([3, 1])

with col1:
    # ユーザーの質問を入力
    user_question = st.text_area(
        "質問や相談内容を入力してください",
        placeholder="例：このCSVデータから売上の季節的傾向を分析し、来月の予測を3つのシナリオで提示してください",
        height=100
    )
    
    # プロンプトコーチング機能（データ分析専用）
    show_prompt_coaching(user_question, question_type)

# データ分析の場合のみ効果的な質問例を表示
if question_type == "データ分析・統計":
    if st.checkbox("💡 データ分析での効果的な質問例を見る"):
        st.info("""
        **🎯 データ分析での効果的な質問例:**
        
        ❌ 悪い例: 「データを分析して」
        ✅ 良い例: 「この売上データから季節による傾向を分析し、来月の予測を3つのシナリオで教えて」
        
        ❌ 悪い例: 「何か分かることある？」  
        ✅ 良い例: 「このスポーツデータから、パフォーマンス向上のための改善点を数値根拠とともに5つ提案して」
        
        ❌ 悪い例: 「グラフにして」
        ✅ 良い例: 「この気温データから異常な日を特定し、その要因を数値で分析して具体的な理由を教えて」
        """)

with col2:
    # 質問に添付する画像
    question_image = st.file_uploader(
        "質問に画像を添付",
        type=['jpg', 'jpeg', 'png'],
        help="質問に関連する画像を1枚アップロード可能",
        key="question_image"
    )

# 質問に添付した画像の表示
if question_image is not None:
    st.subheader("📷 質問に添付した画像")
    image = Image.open(question_image)
    st.image(image, caption=question_image.name, width=300)
    st.info(f"💡 この画像について質問文で「この画像を見て...」と参照できます")

# ================================================
# 8. プロンプト作成関数（複数画像・参照対応版）
# ================================================
def encode_image_to_base64_from_bytes(image_bytes):
    """画像バイトデータをbase64エンコードする"""
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        return base64_image
    except Exception as e:
        st.error(f"画像エンコードエラー: {str(e)}")
        return None

def encode_image_to_base64(image_file):
    """画像ファイルをbase64エンコードする"""
    try:
        image_file.seek(0)
        image_bytes = image_file.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        return base64_image
    except Exception as e:
        st.error(f"画像エンコードエラー: {str(e)}")
        return None

def extract_image_references_from_text(text_content):
    """テキスト内の画像参照を抽出する（改善版）"""
    import re
    
    # パターン1：「スポット名(画像ファイル名.jpg)」- 基本形
    pattern1 = r'([^(「]+?)\(([^)]*\.(?:jpg|jpeg|png|JPG|JPEG|PNG))\)'
    # パターン2：「スポット名(画像ファイル名.jpg)」- 鍵括弧付き
    pattern2 = r'「([^」(]+?)\(([^)]*\.(?:jpg|jpeg|png|JPG|JPEG|PNG))\)[^」]*」'
    # パターン3：「スポット名(画像ファイル名.jpg)」- より柔軟なパターン
    pattern3 = r'「([^」]+?)\(([^)]*\.(?:jpg|jpeg|png|JPG|JPEG|PNG))\)'
    
    image_references = {}
    
    # 各パターンで抽出を試行
    for pattern in [pattern1, pattern2, pattern3]:
        matches = re.findall(pattern, text_content)
        for spot_name, filename in matches:
            # スポット名のクリーニング
            spot_name = spot_name.strip().rstrip('」').lstrip('「').rstrip('・').strip()
            spot_name = spot_name.replace('「', '').replace('」', '')  # 鍵括弧を除去
            filename = filename.strip()
            
            if spot_name and filename and len(spot_name) > 1:  # 最低限の長さチェック
                image_references[spot_name] = filename
    
    return image_references

def find_mentioned_spots_in_response(ai_response, image_references):
    """AI回答内で言及されたスポットを特定し、対応する画像を返す（改善版）"""
    mentioned_images = []
    
    for spot_name, filename in image_references.items():
        # より柔軟なマッチング（部分一致も含む）
        spot_words = spot_name.split()
        found = False
        
        # 完全一致チェック
        if spot_name in ai_response:
            found = True
        else:
            # 部分一致チェック（3文字以上の単語）
            for word in spot_words:
                if len(word) >= 3 and word in ai_response:
                    found = True
                    break
        
        if found:
            mentioned_images.append({
                'spot_name': spot_name,
                'filename': filename
            })
    
    return mentioned_images

def create_enhanced_prompt(question_type, user_question, uploaded_files, question_image=None):
    """
    複数画像対応のプロンプト作成関数（テキスト内画像参照対応）
    """
    
    base_prompt = f"""あなたは親切で知識豊富なAIアシスタントです。
ユーザーの質問に対して、わかりやすく丁寧に回答してください。

"""
    
    # 質問の種類に応じた専門的な指示を追加
    if question_type == "観光プラン作成":
        base_prompt += """特に観光プラン作成の専門家として、以下の点を含めて回答してください：
- 具体的な観光地やスポット
- 移動時間や交通手段
- 予算の目安
- おすすめの食事場所
- 提供された参考資料の詳細情報を活用した提案

"""
    elif question_type == "データ分析・統計":
        base_prompt += """データ分析の専門家として、以下の構造で**簡潔に**回答してください：

【データ概要】（3行以内）
- データの基本情報と統計

【主要な発見】（5つ以内）
- 重要なパターンや傾向

【異常値・注目点】（3つ以内）
- 特筆すべきデータポイント

【結論と提案】（3つのポイント）
- 実用的な洞察と推奨事項

**重要**: 各セクションは簡潔に、全体で2000文字以内で完結してください。
"""

    
    elif question_type == "数学の問題解決":
        base_prompt += """数学の専門家として、以下の点を含めて回答してください：
- 公式や定理の説明
- 段階的な解法手順
- 関連する概念の説明
- 類似問題へのヒント

"""
    
    elif question_type == "就活・自己PR作成":
        base_prompt += """就活アドバイザーとして、以下の点を含めて回答してください：
- 具体的で説得力のある表現
- 企業が求める人材像への適合
- 強みの効果的なアピール方法
- 改善点やアドバイス

"""
    
    # テキスト資料がある場合は追加
    text_content = ""
    all_image_references = {}
    
    for filename, file_info in uploaded_files.items():
        if file_info['type'] == 'text':
            text_content += f"\n\n=== {filename} ===\n{file_info['content']}"
            # テキスト内の画像参照を抽出
            image_refs = extract_image_references_from_text(file_info['content'])
            all_image_references.update(image_refs)
            
        elif file_info['type'] == 'csv':
            # CSVデータを明確にプロンプトに含める
            text_content += f"\n\n=== CSVデータ分析: {filename} ===\n{file_info['content']}"
    
    if text_content:
        base_prompt += f"""
参考資料：
{text_content}

上記のデータを使用して、具体的な数値と統計情報を基に詳細な分析を行ってください。
"""
    
    # ユーザーの質問を追加
    base_prompt += f"""
ユーザーの質問：
{user_question}

提供されたデータの数値を具体的に引用して分析してください。
"""
    
    return base_prompt, all_image_references

# ================================================
# 9. AI応答の生成（複数画像対応版）
# ================================================
def get_ai_response_enhanced(client, prompt, deployment_name, question_image=None):
    """テキスト内画像参照対応のAI応答生成関数"""
    try:
        # メッセージコンテンツを構築
        message_content = [{"type": "text", "text": prompt}]
        
        # 質問に添付された画像を追加（従来機能維持）
        if question_image is not None:
            base64_image = encode_image_to_base64(question_image)
            if base64_image:
                image_format = question_image.type.split('/')[-1]
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_format};base64,{base64_image}"
                    }
                })
        
        # API呼び出し（シンプル化）
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {
                    "role": "user",
                    "content": message_content
                }
            ],
            temperature=0.7,
            max_tokens=3000
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"

# ================================================
# 10. 実行ボタンとAI応答の表示
# ================================================
# ================================================
# 10. 実行ボタンとAI応答の表示
# ================================================

# セッション状態でAI応答を管理（重複実行防止）
if 'ai_response_data' not in st.session_state:
    st.session_state.ai_response_data = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🤖 AIに相談する", type="primary", use_container_width=True, key="main_ai_button"):
        # 重複実行防止
        if not st.session_state.processing:
            # 入力チェック
            if not user_question.strip():
                st.error("質問を入力してください")
            elif not azure_endpoint or not api_key:
                st.error("サイドバーでAzure OpenAIの設定を入力してください")
            else:
                # 処理開始フラグ
                st.session_state.processing = True
                
                # Azure OpenAI クライアントを作成
                client = create_azure_client()
                
                if client:
                    # プロンプトを作成（画像参照情報も取得）
                    prompt, image_references = create_enhanced_prompt(
                        question_type, user_question, st.session_state.uploaded_files, question_image
                    )
                    
                    # AI応答を生成（ローディング表示付き）
                    with st.spinner("🤔 AIが考えています..."):
                        ai_response = get_ai_response_enhanced(
                            client, prompt, deployment_name, question_image
                        )
                    
                    # 結果をセッション状態に保存
                    st.session_state.ai_response_data = {
                        'ai_response': ai_response,
                        'image_references': image_references,
                        'question_image': question_image,
                        'prompt': prompt,
                        'counted': False  # ← カウント済みフラグを追加
                    }
                
                # 処理完了フラグ
                st.session_state.processing = False

# AI応答の表示（セッション状態から）
if st.session_state.ai_response_data is not None:
    data = st.session_state.ai_response_data
    ai_response = data['ai_response']
    image_references = data['image_references']
    question_image = data['question_image']
    prompt = data['prompt']
    
    # =================================================
    # AI応答表示（1回のみ）
    # =================================================
    st.markdown("---")
    st.header("🤖 AIからの回答")
    st.markdown(ai_response)
    
    # =================================================
    # 質問添付画像の表示
    # =================================================
    if question_image is not None:
        st.subheader("📷 質問に添付された画像")
        image = Image.open(question_image)
        st.image(image, caption=question_image.name, width=300)
    
    # =================================================
    # スポット画像の表示
    # =================================================
    if image_references:
        mentioned_spots = []
        for spot_name, filename in image_references.items():
            # スポット検索
            spot_words = spot_name.split()
            found = False
            
            if spot_name in ai_response:
                found = True
            else:
                for word in spot_words:
                    if len(word) >= 3 and word in ai_response:
                        found = True
                        break
            
            if found:
                mentioned_spots.append({
                    'spot_name': spot_name,
                    'filename': filename
                })
        
        # 言及されたスポットの画像のみ表示
        if mentioned_spots:
            st.subheader("📷 回答で言及されたスポットの画像")
            
            for spot_info in mentioned_spots:
                spot_name = spot_info['spot_name']
                filename = spot_info['filename']
                
                if filename in st.session_state.uploaded_files:
                    file_info = st.session_state.uploaded_files[filename]
                    if file_info['type'] == 'image':
                        st.write(f"**{spot_name}**")
                        image = Image.open(io.BytesIO(file_info['file_bytes']))
                        st.image(image, caption=f"📁 {filename}", width=400)
                else:
                    st.error(f"画像ファイル「{filename}」が見つかりません")
            
            # スポット一覧表示
            spot_names = [spot['spot_name'] for spot in mentioned_spots]
            st.success(f"📍 {len(mentioned_spots)}箇所のスポット画像を表示: {', '.join(spot_names)}")
        else:
            st.info("💡 AI回答でテキスト内の画像参照スポットが言及されませんでした。")
    
    # =================================================
    # 分析回数カウント（データ分析・統計のみ）
    # =================================================
    current_question_type = st.session_state.get('current_question_type', '')
    if current_question_type == "データ分析・統計":
        st.session_state.analysis_count += 1
        # デバッグ表示は削除

    # =================================================
    # デバッグ情報
    # =================================================
    with st.expander("🔍 詳細情報（デバッグ用）"):
        if image_references:
            st.write("📄 **抽出された画像参照情報:**")
            for spot, filename in image_references.items():
                st.write(f"  • 「{spot}」→ {filename}")
        
        st.write("🤖 **送信したプロンプト:**")
        st.text(prompt)
        
        if question_image:
            st.info("📷 質問に画像が添付されました")
        if image_references:
            st.info(f"📁 テキスト内で参照可能な画像: {len(image_references)}枚")
    
    # 結果をクリアするボタン
    if st.button("🔄 新しい質問をする", key="clear_response"):
        st.session_state.ai_response_data = None
        st.rerun()


# ================================================
# 11. 使用上の注意とヒント
# ================================================
st.markdown("---")
st.header("💡 使用のヒント")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **良い質問の仕方：**
    - 具体的で明確な質問をする
    - 背景情報や条件を含める
    - 期待する回答の形式を指定する
    
    **新しい画像参照方法：**
    - テキスト内で「スポット名(画像ファイル名.jpg)」
    - AIが回答でそのスポットを言及すると画像表示
    - より自然な資料作成が可能
    """)

with col2:
    st.markdown("""
    **テキスト内画像参照の例：**
    ```
    【東尋坊の絶景ポイント】
    ・雄島夕陽ベンチ(東尋坊夕日.jpg)：
      地元民専用の撮影スポット
    
    【恐竜博物館の体験】  
    ・化石発掘体験(化石発掘.jpg)：
      隠し予約枠あり
    ```
    
    **デモでのインパクト：**
    - AI回答で言及されたスポットの画像が自動表示
    - 自然な文書形式での画像管理
    """)

# 使用可能な画像ファイル一覧
if st.session_state.uploaded_files:
    image_files = [k for k, v in st.session_state.uploaded_files.items() if v['type'] == 'image']
    if image_files:
        st.markdown("### 🎯 現在利用可能な画像ファイル")
        cols = st.columns(min(len(image_files), 4))
        for i, filename in enumerate(image_files):
            with cols[i % 4]:
                st.code(filename, language=None)
        st.info("💡 テキスト内で「スポット名(ファイル名)」形式で参照すると、AI回答でそのスポットが言及された際に画像が表示されます")

# ================================================
# 12. フッター
# ================================================
st.markdown("---")
st.markdown("**PBL Project** | Powered by Azure OpenAI GPT-4o + Streamlit | 📷 Smart Image Reference System")
