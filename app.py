"""
YOLO物体検出アプリケーション
StreamlitベースのWebアプリケーション
"""

import streamlit as st
import os
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from yolo_model_manager import YOLOModelManager
from yolo_visualization import YOLOv8VisualizationManager
from ui_components import (
    YOLOUIComponents,
    ImageUploadComponent,
    DetectionResultComponent,
    ModelSettingsComponent
)

# ページ設定
st.set_page_config(
    page_title="YOLO物体検出アプリ",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #5a6fd8 0%, #6a4190 100%);
    }
</style>
""", unsafe_allow_html=True)

class YOLOApp:
    """YOLO物体検出アプリケーションのメインクラス"""
    
    def __init__(self):
        """アプリケーションの初期化"""
        self.model_manager = YOLOModelManager()
        self.ui_components = YOLOUIComponents()
        self.image_upload = ImageUploadComponent()
        self.detection_result = DetectionResultComponent()
        self.model_settings = ModelSettingsComponent()
        
        # セッション状態の初期化
        if 'detection_results' not in st.session_state:
            st.session_state.detection_results = None
        if 'uploaded_image' not in st.session_state:
            st.session_state.uploaded_image = None
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
    
    def run(self):
        """アプリケーションを実行"""
        # ヘッダー
        self._display_header()
        
        # サイドバー
        self._display_sidebar()
        
        # メインコンテンツ
        self._display_main_content()
    
    def _display_header(self):
        """ヘッダーを表示"""
        st.markdown("""
        <div class="main-header">
            <h1>🎯 YOLO物体検出アプリケーション</h1>
            <p>リアルタイム物体検出と可視化ツール</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _display_sidebar(self):
        """サイドバーを表示"""
        with st.sidebar:
            st.header("⚙️ 設定")
            
            # モデル設定
            self.model_settings.display_model_settings()
            
            # 検出設定
            st.subheader("🔍 検出設定")
            confidence_threshold = st.slider(
                "信頼度閾値",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="検出結果の信頼度閾値を設定"
            )
            
            nms_threshold = st.slider(
                "NMS閾値",
                min_value=0.0,
                max_value=1.0,
                value=0.4,
                step=0.05,
                help="Non-Maximum Suppressionの閾値を設定"
            )
            
            # 可視化設定
            st.subheader("🎨 可視化設定")
            show_labels = st.checkbox("ラベルを表示", value=True)
            show_confidence = st.checkbox("信頼度を表示", value=True)
            show_boxes = st.checkbox("バウンディングボックスを表示", value=True)
            
            # 設定をセッション状態に保存
            st.session_state.confidence_threshold = confidence_threshold
            st.session_state.nms_threshold = nms_threshold
            st.session_state.show_labels = show_labels
            st.session_state.show_confidence = show_confidence
            st.session_state.show_boxes = show_boxes
    
    def _display_main_content(self):
        """メインコンテンツを表示"""
        # タブの作成
        tab1, tab2, tab3, tab4 = st.tabs([
            "📷 物体検出", 
            "📊 可視化", 
            "📈 統計", 
            "ℹ️ 情報"
        ])
        
        with tab1:
            self._display_detection_tab()
        
        with tab2:
            self._display_visualization_tab()
        
        with tab3:
            self._display_statistics_tab()
        
        with tab4:
            self._display_info_tab()
    
    def _display_detection_tab(self):
        """物体検出タブを表示"""
        st.header("📷 物体検出")
        
        # 画像アップロード
        uploaded_file = self.image_upload.display_image_upload()
        
        if uploaded_file is not None:
            # 画像をセッション状態に保存
            st.session_state.uploaded_image = uploaded_file
            
            # 検出実行ボタン
            if st.button("🔍 検出を実行", type="primary"):
                with st.spinner("検出中..."):
                    try:
                        # モデルがロードされていない場合はロード
                        if not st.session_state.model_loaded:
                            self.model_manager.load_model()
                            st.session_state.model_loaded = True
                        
                        # 検出実行
                        results = self.model_manager.detect_objects(
                            uploaded_file,
                            confidence=st.session_state.confidence_threshold,
                            nms_threshold=st.session_state.nms_threshold
                        )
                        
                        # 結果をセッション状態に保存
                        st.session_state.detection_results = results
                        
                        st.success("検出完了！")
                        
                    except Exception as e:
                        st.error(f"検出中にエラーが発生しました: {str(e)}")
            
            # 検出結果の表示
            if st.session_state.detection_results is not None:
                self.detection_result.display_detection_results(
                    st.session_state.uploaded_image,
                    st.session_state.detection_results,
                    show_labels=st.session_state.show_labels,
                    show_confidence=st.session_state.show_confidence,
                    show_boxes=st.session_state.show_boxes
                )
    
    def _display_visualization_tab(self):
        """可視化タブを表示"""
        st.header("📊 YOLOv8可視化")
        
        # YOLOv8の可視化を表示
        YOLOv8VisualizationManager.display_yolo_visualization()
    
    def _display_statistics_tab(self):
        """統計タブを表示"""
        st.header("📈 検出統計")
        
        if st.session_state.detection_results is not None:
            # 検出結果の統計情報を表示
            self._display_detection_statistics()
        else:
            st.info("検出結果がありません。まず物体検出を実行してください。")
    
    def _display_info_tab(self):
        """情報タブを表示"""
        st.header("ℹ️ アプリケーション情報")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚀 機能")
            st.markdown("""
            - **リアルタイム物体検出**: YOLOv8を使用した高速検出
            - **マルチクラス検出**: 80種類のCOCOクラスに対応
            - **可視化機能**: 検出結果の詳細な可視化
            - **設定可能**: 信頼度閾値やNMS閾値の調整
            - **統計情報**: 検出結果の統計分析
            """)
        
        with col2:
            st.subheader("🔧 技術仕様")
            st.markdown("""
            - **フレームワーク**: Streamlit
            - **モデル**: YOLOv8 (Ultralytics)
            - **バックエンド**: PyTorch
            - **可視化**: OpenCV, Matplotlib
            - **対応形式**: JPG, PNG, JPEG
            """)
        
        st.subheader("📖 使用方法")
        st.markdown("""
        1. **画像アップロード**: 検出したい画像をアップロード
        2. **設定調整**: サイドバーで検出パラメータを調整
        3. **検出実行**: 「検出を実行」ボタンをクリック
        4. **結果確認**: 検出結果と統計情報を確認
        5. **可視化**: タブを切り替えて詳細な可視化を確認
        """)
    
    def _display_detection_statistics(self):
        """検出統計を表示"""
        results = st.session_state.detection_results
        
        if results is None:
            return
        
        # 基本統計
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("検出物体数", len(results.boxes) if results.boxes is not None else 0)
        
        with col2:
            if results.boxes is not None and len(results.boxes) > 0:
                avg_confidence = float(results.boxes.conf.mean())
                st.metric("平均信頼度", f"{avg_confidence:.3f}")
            else:
                st.metric("平均信頼度", "N/A")
        
        with col3:
            if results.boxes is not None and len(results.boxes) > 0:
                max_confidence = float(results.boxes.conf.max())
                st.metric("最大信頼度", f"{max_confidence:.3f}")
            else:
                st.metric("最大信頼度", "N/A")
        
        with col4:
            if results.boxes is not None and len(results.boxes) > 0:
                min_confidence = float(results.boxes.conf.min())
                st.metric("最小信頼度", f"{min_confidence:.3f}")
            else:
                st.metric("最小信頼度", "N/A")
        
        # クラス別統計
        if results.boxes is not None and len(results.boxes) > 0:
            st.subheader("📊 クラス別検出統計")
            
            # クラス名と検出数を取得
            class_counts = {}
            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = results.names[class_id]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # クラス別統計を表示
            for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                st.markdown(f"**{class_name}**: {count}個検出")

def main():
    """メイン関数"""
    try:
        app = YOLOApp()
        app.run()
    except Exception as e:
        st.error(f"アプリケーションの実行中にエラーが発生しました: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main() 