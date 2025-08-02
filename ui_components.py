"""
YOLOv8アプリケーションのUIコンポーネント
サイドバー、入力、結果表示のUI要素を提供
"""

import streamlit as st
from typing import Optional, Tuple
from PIL import Image
import requests
from io import BytesIO


class SidebarManager:
    """サイドバーの管理を担当するクラス"""
    
    @staticmethod
    def create_model_selection() -> str:
        """
        モデル選択のUIを作成します
        
        Returns:
            選択されたモデル名
        """
        st.sidebar.header("🔧 設定")
        
        from yolo_model_manager import YOLOModelLoader
        available_models = YOLOModelLoader.get_available_models()
        
        selected_model = st.sidebar.selectbox(
            "YOLOv8モデルを選択",
            available_models,
            index=0,
            help="使用するYOLOv8モデルを選択してください（nanoが推奨）"
        )
        
        return selected_model
    
    @staticmethod
    def create_detection_settings() -> Tuple[float, float]:
        """
        検出設定のUIを作成します
        
        Returns:
            (信頼度閾値, IoU閾値)のタプル
        """
        st.sidebar.subheader("検出設定")
        
        confidence_threshold = st.sidebar.slider(
            "信頼度閾値",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="検出結果の信頼度閾値を設定（低い値ほど多くの物体を検出）"
        )
        
        iou_threshold = st.sidebar.slider(
            "IoU閾値",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.05,
            help="Non-Maximum SuppressionのIoU閾値を設定"
        )
        
        return confidence_threshold, iou_threshold
    
    @staticmethod
    def create_debug_settings() -> bool:
        """
        デバッグ設定のUIを作成します
        
        Returns:
            デバッグモードが有効かどうか
        """
        st.sidebar.subheader("🐛 デバッグ設定")
        
        debug_mode = st.sidebar.checkbox(
            "デバッグモードを有効化",
            value=True,
            help="詳細なデバッグ情報とYOLOv8の仕組みを表示"
        )
        
        # デバッグモードをセッション状態に保存
        st.session_state.debug_mode = debug_mode
        
        return debug_mode
    
    @staticmethod
    def create_visualization_options() -> str:
        """
        可視化オプションのUIを作成します
        
        Returns:
            選択された可視化タイプ
        """
        st.sidebar.subheader("📊 可視化オプション")
        
        show_yolo_diagram = st.sidebar.checkbox(
            "YOLOv8仕組み図を表示",
            value=False,
            help="YOLOv8のアーキテクチャと処理フローを可視化"
        )
        
        if show_yolo_diagram:
            st.session_state.show_yolo_diagram = True
        else:
            st.session_state.show_yolo_diagram = False
        
        return "yolo_diagram" if show_yolo_diagram else "none"


class InputManager:
    """入力管理を担当するクラス"""
    
    @staticmethod
    def create_image_input_section() -> Optional[Image.Image]:
        """
        画像入力セクションのUIを作成します
        
        Returns:
            読み込まれた画像、失敗時はNone
        """
        st.header("📷 画像入力")
        
        input_method = st.selectbox(
            "画像入力方法を選択",
            ["サンプル画像を使用", "画像をアップロード", "URLから画像を取得", "カメラで撮影"]
        )
        
        image = None
        
        if input_method == "サンプル画像を使用":
            image = InputManager._load_sample_image()
        elif input_method == "画像をアップロード":
            image = InputManager._load_uploaded_image()
        elif input_method == "URLから画像を取得":
            image = InputManager._load_url_image()
        elif input_method == "カメラで撮影":
            image = InputManager._load_camera_image()
        
        if image:
            st.success("画像が正常に読み込まれました")
            st.image(image, caption="入力画像", use_container_width=True)
            
            # 画像情報の表示
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("幅", f"{image.width}px")
            with col2:
                st.metric("高さ", f"{image.height}px")
            with col3:
                st.metric("サイズ", f"{image.width * image.height:,}px²")
            
            return image
        else:
            st.warning("画像を入力してください")
            return None
    
    @staticmethod
    def _load_sample_image() -> Optional[Image.Image]:
        """サンプル画像を読み込みます"""
        import requests
        from io import BytesIO
        
        # サンプル画像の選択
        sample_images = {
            "オフィスシーン": "https://images.unsplash.com/photo-1497366216548-37526070297c?w=800&h=600&fit=crop",
            "キッチンシーン": "https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=800&h=600&fit=crop",
            "リビングルーム": "https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=800&h=600&fit=crop",
            "街並み": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800&h=600&fit=crop",
            "自然風景": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&h=600&fit=crop",
            "人物写真": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800&h=600&fit=crop",
            "車の写真": "https://images.unsplash.com/photo-1549317661-bd32c8ce0db2?w=800&h=600&fit=crop"
        }
        
        selected_image = st.selectbox(
            "サンプル画像を選択",
            list(sample_images.keys())
        )
        
        if selected_image:
            try:
                url = sample_images[selected_image]
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                return image
            except Exception as e:
                st.error(f"サンプル画像の読み込みに失敗しました: {e}")
                return None
        
        return None
    
    @staticmethod
    def _load_uploaded_image() -> Optional[Image.Image]:
        """アップロードされた画像を読み込みます"""
        uploaded_file = st.file_uploader(
            "画像ファイルをアップロード",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="PNG, JPG, JPEG, BMP, TIFF形式の画像をアップロードできます"
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                return image
            except Exception as e:
                st.error(f"画像の読み込みに失敗しました: {e}")
                return None
        
        return None
    
    @staticmethod
    def _load_url_image() -> Optional[Image.Image]:
        """URLから画像を読み込みます"""
        url = st.text_input(
            "画像URLを入力",
            placeholder="https://example.com/image.jpg",
            help="画像のURLを入力してください"
        )
        
        if url:
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                return image
            except Exception as e:
                st.error(f"URLからの画像読み込みに失敗しました: {e}")
                return None
        
        return None
    
    @staticmethod
    def _load_camera_image() -> Optional[Image.Image]:
        """カメラで撮影した画像を読み込みます"""
        try:
            camera_input = st.camera_input("カメラで撮影")
            if camera_input is not None:
                return Image.open(camera_input)
            return None
        except Exception as e:
            st.error(f"カメラからの画像読み込みに失敗しました: {e}")
            return None


class ResultsManager:
    """結果表示を担当するクラス"""
    
    @staticmethod
    def display_detection_results(
        detection_result: dict,
        debug_mode: bool = False
    ) -> None:
        """
        検出結果を表示します
        
        Args:
            detection_result: 検出結果
            debug_mode: デバッグモード
        """
        if not detection_result:
            st.warning("検出結果がありません")
            return
        
        st.subheader("🎯 検出結果")
        
        # 基本情報
        boxes = detection_result['boxes']
        scores = detection_result['scores']
        class_names = detection_result['class_names']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("検出物体数", len(boxes))
        with col2:
            st.metric("平均信頼度", f"{sum(scores)/len(scores):.3f}")
        with col3:
            if detection_result.get('inference_time'):
                st.metric("推論時間", f"{detection_result['inference_time']:.1f}ms")
        
        # 検出詳細
        if debug_mode:
            st.write("**📋 検出詳細:**")
            for i, (box, score, class_name) in enumerate(zip(boxes, scores, class_names)):
                x1, y1, x2, y2 = box
                st.write(f"{i+1}. **{class_name}** (信頼度: {score:.3f}) - 座標: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
        
        # クラス別統計
        unique_classes, counts = {}, {}
        for class_name in class_names:
            if class_name in unique_classes:
                counts[class_name] += 1
            else:
                unique_classes[class_name] = len(unique_classes)
                counts[class_name] = 1
        
        st.write("**🏷️ クラス別検出数:**")
        for class_name, count in counts.items():
            st.write(f"  - {class_name}: {count}個")
    
    @staticmethod
    def create_download_section(visualized_image: Image.Image) -> None:
        """
        ダウンロードセクションを作成します
        
        Args:
            visualized_image: 可視化された画像
        """
        st.subheader("💾 結果のダウンロード")
        
        # 画像をバイトに変換
        import io
        img_buffer = io.BytesIO()
        visualized_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        st.download_button(
            label="検出結果画像をダウンロード",
            data=img_buffer.getvalue(),
            file_name="yolo_detection_result.png",
            mime="image/png",
            help="検出結果が描画された画像をPNG形式でダウンロード"
        )
    
    @staticmethod
    def display_yolo_info() -> None:
        """YOLOv8の情報を表示します"""
        st.subheader("🚀 YOLOv8について")
        
        st.markdown("""
        **YOLOv8（You Only Look Once version 8）**は、Ultralyticsが開発した最新の物体検出モデルです。
        
        **主な特徴:**
        - 🚀 **Anchor-Free設計**: アンカーボックスが不要でシンプル
        - 🔗 **CSPDarknetバックボーン**: 効率的な特徴抽出
        - 📊 **PANet (FPN + PAN)**: マルチスケール特徴融合
        - ⚡ **高速推論**: リアルタイム物体検出
        - 📱 **エッジデバイス対応**: 軽量モデルでモバイル対応
        - 🎯 **高精度**: COCOデータセットで高いmAPを達成
        
        **技術仕様:**
        - 入力サイズ: 640x640ピクセル
        - アーキテクチャ: CSPDarknet + PANet + Anchor-Free Head
        - 損失関数: BCE + CIoU + DFL
        - 最適化器: AdamW
        - データセット: COCO (80クラス)
        
        **モデルサイズ:**
        - **YOLOv8n**: 3.2Mパラメータ (軽量・高速)
        - **YOLOv8s**: 11.2Mパラメータ (バランス型)
        - **YOLOv8m**: 25.9Mパラメータ (高精度)
        - **YOLOv8l**: 43.7Mパラメータ (最高精度)
        - **YOLOv8x**: 68.2Mパラメータ (最大精度)
        """)
        
        # 参考リンク
        with st.expander("📚 参考資料"):
            st.markdown("""
            **公式ドキュメント:**
            - [Ultralytics YOLOv8](https://docs.ultralytics.com/)
            - [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
            - [YOLOv8 Paper](https://arxiv.org/abs/2304.00501)
            
            **技術詳細:**
            - [YOLOv8 Architecture](https://docs.ultralytics.com/models/yolov8/)
            - [YOLOv8 Training](https://docs.ultralytics.com/modes/train/)
            - [YOLOv8 Inference](https://docs.ultralytics.com/modes/predict/)
            """) 