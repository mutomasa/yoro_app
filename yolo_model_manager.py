"""
YOLOv8モデルの管理を担当するモジュール
モデルの読み込み、推論、後処理機能を提供
"""

import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import time


class YOLOModelLoader:
    """YOLOv8モデルの読み込みを担当するクラス"""
    
    DEFAULT_MODEL_NAME = "yolov8n.pt"
    
    @staticmethod
    @st.cache_resource
    def load_model(model_name: str = DEFAULT_MODEL_NAME) -> Optional[YOLO]:
        """
        YOLOv8モデルを読み込みます
        
        Args:
            model_name: モデル名
            
        Returns:
            読み込まれたモデル、失敗時はNone
        """
        try:
            with st.spinner(f"YOLOv8モデル '{model_name}' を読み込み中..."):
                model = YOLO(model_name)
                return model
        except Exception as e:
            st.error(f"モデルの読み込みに失敗しました: {e}")
            return None
    
    @staticmethod
    def get_available_models() -> List[str]:
        """
        利用可能なモデルのリストを取得します
        
        Returns:
            利用可能なモデル名のリスト
        """
        return [
            "yolov8n.pt",  # nano
            "yolov8s.pt",  # small
            "yolov8m.pt",  # medium
            "yolov8l.pt",  # large
            "yolov8x.pt"   # xlarge
        ]
    
    @staticmethod
    def get_model_info(model_name: str) -> Dict[str, Any]:
        """
        モデルの詳細情報を取得します
        
        Args:
            model_name: モデル名
            
        Returns:
            モデルの詳細情報
        """
        model_info = {
            "yolov8n.pt": {
                "name": "YOLOv8 Nano",
                "parameters": "3.2M",
                "size": "6.3MB",
                "mAP50": "0.637",
                "mAP50-95": "0.454",
                "speed_gpu": "8.7ms",
                "speed_cpu": "23.4ms",
                "description": "軽量で高速、エッジデバイス向け"
            },
            "yolov8s.pt": {
                "name": "YOLOv8 Small",
                "parameters": "11.2M",
                "size": "22.6MB",
                "mAP50": "0.718",
                "mAP50-95": "0.554",
                "speed_gpu": "12.9ms",
                "speed_cpu": "35.2ms",
                "description": "バランス型、一般的な用途"
            },
            "yolov8m.pt": {
                "name": "YOLOv8 Medium",
                "parameters": "25.9M",
                "size": "52.2MB",
                "mAP50": "0.764",
                "mAP50-95": "0.628",
                "speed_gpu": "22.6ms",
                "speed_cpu": "61.8ms",
                "description": "高精度、サーバー向け"
            },
            "yolov8l.pt": {
                "name": "YOLOv8 Large",
                "parameters": "43.7M",
                "size": "87.7MB",
                "mAP50": "0.792",
                "mAP50-95": "0.671",
                "speed_gpu": "31.2ms",
                "speed_cpu": "85.4ms",
                "description": "最高精度、研究用途"
            },
            "yolov8x.pt": {
                "name": "YOLOv8 XLarge",
                "parameters": "68.2M",
                "size": "136.6MB",
                "mAP50": "0.814",
                "mAP50-95": "0.699",
                "speed_gpu": "35.7ms",
                "speed_cpu": "98.1ms",
                "description": "最大精度、特殊用途"
            }
        }
        
        return model_info.get(model_name, {})
    
    @staticmethod
    def get_coco_classes() -> List[str]:
        """
        COCOデータセットのクラス名を取得します
        
        Returns:
            COCOクラス名のリスト
        """
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]


class YOLODetectionProcessor:
    """YOLOv8検出処理を担当するクラス"""
    
    @staticmethod
    def run_inference(
        model: YOLO,
        image: Image.Image,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        debug_mode: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        YOLOv8推論を実行します
        
        Args:
            model: YOLOv8モデル
            image: 入力画像
            confidence_threshold: 信頼度閾値
            iou_threshold: IoU閾値
            debug_mode: デバッグモード
            
        Returns:
            推論結果、失敗時はNone
        """
        try:
            if debug_mode:
                st.write("**🚀 YOLOv8推論実行中...**")
                start_time = time.time()
            
            # 推論実行
            results = model(
                image,
                conf=confidence_threshold,
                iou=iou_threshold,
                verbose=False
            )
            
            if debug_mode:
                end_time = time.time()
                inference_time = (end_time - start_time) * 1000  # ms
                st.write(f"✅ 推論完了 (処理時間: {inference_time:.2f}ms)")
            
            # 結果の解析
            if results and len(results) > 0:
                result = results[0]  # 最初の結果を使用
                
                # 検出結果の取得
                boxes = result.boxes
                if boxes is not None:
                    # バウンディングボックス、信頼度、クラスIDを取得
                    bbox_data = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                    confidence_scores = boxes.conf.cpu().numpy()
                    class_ids = boxes.cls.cpu().numpy().astype(int)
                    
                    # クラス名を取得
                    coco_classes = YOLOModelLoader.get_coco_classes()
                    class_names = [coco_classes[class_id] for class_id in class_ids]
                    
                    detection_result = {
                        'boxes': bbox_data.tolist(),
                        'scores': confidence_scores.tolist(),
                        'class_ids': class_ids.tolist(),
                        'class_names': class_names,
                        'inference_time': inference_time if debug_mode else None
                    }
                    
                    if debug_mode:
                        st.write(f"**📊 検出結果:**")
                        st.write(f"- 検出された物体数: {len(bbox_data)}")
                        st.write(f"- 平均信頼度: {np.mean(confidence_scores):.3f}")
                        st.write(f"- 最高信頼度: {np.max(confidence_scores):.3f}")
                        
                        # クラス別の検出数
                        unique_classes, counts = np.unique(class_names, return_counts=True)
                        st.write("**🏷️ クラス別検出数:**")
                        for class_name, count in zip(unique_classes, counts):
                            st.write(f"  - {class_name}: {count}個")
                    
                    return detection_result
                else:
                    if debug_mode:
                        st.warning("検出された物体がありません")
                    return None
            else:
                if debug_mode:
                    st.warning("推論結果が空です")
                return None
                
        except Exception as e:
            st.error(f"推論の実行に失敗しました: {e}")
            if debug_mode:
                st.write(f"エラーの詳細: {str(e)}")
            return None
    
    @staticmethod
    def create_visualization(
        image: Image.Image,
        detection_result: Dict[str, Any],
        debug_mode: bool = False
    ) -> Optional[Image.Image]:
        """
        検出結果を可視化します
        
        Args:
            image: 元画像
            detection_result: 検出結果
            debug_mode: デバッグモード
            
        Returns:
            可視化された画像、失敗時はNone
        """
        try:
            if debug_mode:
                st.write("**🎨 可視化処理中...**")
            
            # 画像のコピーを作成
            vis_image = image.copy()
            
            # 検出結果を描画
            boxes = detection_result['boxes']
            scores = detection_result['scores']
            class_names = detection_result['class_names']
            
            # 色の定義
            colors = [
                '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF',
                '#00FFFF', '#FFA500', '#800080', '#008000', '#800000'
            ]
            
            from PIL import ImageDraw, ImageFont
            
            # フォントの設定
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            draw = ImageDraw.Draw(vis_image)
            
            for i, (box, score, class_name) in enumerate(zip(boxes, scores, class_names)):
                x1, y1, x2, y2 = box
                
                # 色の選択
                color = colors[i % len(colors)]
                
                # バウンディングボックスを描画
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # ラベルテキスト
                label_text = f"{class_name}: {score:.2f}"
                
                # ラベルの背景を描画
                bbox = draw.textbbox((x1, y1-20), label_text, font=font)
                draw.rectangle(bbox, fill=color)
                
                # ラベルテキストを描画
                draw.text((x1, y1-20), label_text, fill='white', font=font)
            
            if debug_mode:
                st.write("✅ 可視化完了")
            
            return vis_image
            
        except Exception as e:
            st.error(f"可視化の作成に失敗しました: {e}")
            if debug_mode:
                st.write(f"エラーの詳細: {str(e)}")
            return None


class YOLODebugInfo:
    """YOLOv8のデバッグ情報を管理するクラス"""
    
    @staticmethod
    def display_model_info(model_name: str) -> None:
        """
        モデル情報を表示します
        
        Args:
            model_name: モデル名
        """
        model_info = YOLOModelLoader.get_model_info(model_name)
        
        if model_info:
            st.subheader(f"📋 {model_info['name']} モデル情報")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("パラメータ数", model_info['parameters'])
                st.metric("モデルサイズ", model_info['size'])
                st.metric("mAP@0.5", model_info['mAP50'])
            
            with col2:
                st.metric("mAP@0.5:0.95", model_info['mAP50-95'])
                st.metric("GPU推論速度", model_info['speed_gpu'])
                st.metric("CPU推論速度", model_info['speed_cpu'])
            
            st.info(f"**説明:** {model_info['description']}")
    
    @staticmethod
    def display_system_info() -> None:
        """システム情報を表示します"""
        st.subheader("💻 システム情報")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Python バージョン:** {torch.version.python}")
            st.write(f"**PyTorch バージョン:** {torch.__version__}")
            st.write(f"**CUDA 利用可能:** {torch.cuda.is_available()}")
        
        with col2:
            if torch.cuda.is_available():
                st.write(f"**CUDA バージョン:** {torch.version.cuda}")
                st.write(f"**GPU デバイス:** {torch.cuda.get_device_name(0)}")
                st.write(f"**GPU メモリ:** {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            else:
                st.write("**GPU:** 利用不可")
                st.write("**推論:** CPUモード")
                st.write("**注意:** GPU使用を推奨")
    
    @staticmethod
    def display_detection_stats(detection_result: Dict[str, Any]) -> None:
        """
        検出統計情報を表示します
        
        Args:
            detection_result: 検出結果
        """
        if not detection_result:
            return
        
        st.subheader("📊 検出統計")
        
        boxes = detection_result['boxes']
        scores = detection_result['scores']
        class_names = detection_result['class_names']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("検出物体数", len(boxes))
            st.metric("平均信頼度", f"{np.mean(scores):.3f}")
        
        with col2:
            st.metric("最高信頼度", f"{np.max(scores):.3f}")
            st.metric("最低信頼度", f"{np.min(scores):.3f}")
        
        with col3:
            st.metric("ユニーククラス数", len(set(class_names)))
            if detection_result.get('inference_time'):
                st.metric("推論時間", f"{detection_result['inference_time']:.1f}ms")
        
        # クラス別の検出数
        unique_classes, counts = np.unique(class_names, return_counts=True)
        
        st.write("**🏷️ クラス別検出数:**")
        for class_name, count in zip(unique_classes, counts):
            st.write(f"  - {class_name}: {count}個")
        
        # 信頼度分布
        if len(scores) > 0:
            st.write("**📈 信頼度分布:**")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('信頼度')
            ax.set_ylabel('頻度')
            ax.set_title('検出信頼度の分布')
            st.pyplot(fig)


class YOLOModelManager:
    """YOLOv8モデルの管理を担当するクラス"""
    
    def __init__(self):
        """モデルマネージャーの初期化"""
        self.model = None
        self.model_name = "yolov8n.pt"
    
    def load_model(self, model_name: str = "yolov8n.pt") -> bool:
        """
        モデルを読み込みます
        
        Args:
            model_name: モデル名
            
        Returns:
            読み込み成功時はTrue
        """
        try:
            self.model = YOLOModelLoader.load_model(model_name)
            self.model_name = model_name
            return self.model is not None
        except Exception as e:
            st.error(f"モデルの読み込みに失敗しました: {e}")
            return False
    
    def detect_objects(
        self, 
        image, 
        confidence: float = 0.5, 
        nms_threshold: float = 0.4
    ):
        """
        物体検出を実行します
        
        Args:
            image: 入力画像
            confidence: 信頼度閾値
            nms_threshold: NMS閾値
            
        Returns:
            検出結果
        """
        if self.model is None:
            st.error("モデルが読み込まれていません")
            return None
        
        try:
            # 検出実行
            results = self.model(
                image,
                conf=confidence,
                iou=nms_threshold,
                verbose=False
            )
            return results[0] if results else None
        except Exception as e:
            st.error(f"検出中にエラーが発生しました: {e}")
            return None
    
    def get_available_models(self) -> List[str]:
        """利用可能なモデルのリストを取得"""
        return YOLOModelLoader.get_available_models()
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """モデルの詳細情報を取得"""
        return YOLOModelLoader.get_model_info(model_name) 