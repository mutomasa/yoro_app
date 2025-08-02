"""
YOLOv8の処理フロー可視化モジュール
画像→バックボーン→Neck→Head→検出の流れを視覚的に表現
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from typing import List, Tuple, Optional, Dict, Any
import seaborn as sns


class YOLOv8Visualizer:
    """YOLOv8の処理フローを可視化するクラス"""
    
    def __init__(self):
        """可視化クラスの初期化"""
        # 日本語フォントの設定
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
    
    @staticmethod
    def create_processing_flow_diagram() -> plt.Figure:
        """
        YOLOv8の処理フロー図を作成します
        
        Returns:
            処理フロー図のmatplotlib Figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 8)
        ax.axis('off')
        
        # 背景色
        ax.set_facecolor('#f8f9fa')
        
        # ステップ1: 画像入力
        YOLOv8Visualizer._draw_image_input_step(ax, 1.5, 6)
        
        # ステップ2: バックボーン
        YOLOv8Visualizer._draw_backbone_step(ax, 3.5, 6)
        
        # ステップ3: Neck (FPN)
        YOLOv8Visualizer._draw_neck_step(ax, 5.5, 6)
        
        # ステップ4: Head
        YOLOv8Visualizer._draw_head_step(ax, 7.5, 6)
        
        # ステップ5: 検出結果
        YOLOv8Visualizer._draw_detection_step(ax, 9.5, 6)
        
        # 矢印で接続
        YOLOv8Visualizer._draw_arrows(ax)
        
        # タイトル
        ax.text(6, 7.5, '🚀 YOLOv8 処理フロー', 
                fontsize=20, fontweight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        return fig
    
    @staticmethod
    def _draw_image_input_step(ax, x: float, y: float):
        """画像入力ステップを描画"""
        # 画像の枠
        rect = patches.Rectangle((x-0.8, y-0.6), 1.6, 1.2, 
                               linewidth=2, edgecolor='#007bff', facecolor='#e3f2fd')
        ax.add_patch(rect)
        
        # 画像アイコン
        ax.text(x, y+0.1, '📷', fontsize=24, ha='center')
        
        # ラベル
        ax.text(x, y-0.8, '画像入力', fontsize=12, fontweight='bold', ha='center')
        ax.text(x, y-1.0, '640x640', fontsize=10, ha='center', color='#666')
    
    @staticmethod
    def _draw_backbone_step(ax, x: float, y: float):
        """バックボーンステップを描画"""
        # バックボーンの枠
        rect = patches.Rectangle((x-0.8, y-0.6), 1.6, 1.2, 
                               linewidth=2, edgecolor='#28a745', facecolor='#e8f5e8')
        ax.add_patch(rect)
        
        # バックボーンアイコン
        ax.text(x, y+0.1, '🔗', fontsize=24, ha='center')
        
        # ラベル
        ax.text(x, y-0.8, 'バックボーン', fontsize=12, fontweight='bold', ha='center')
        ax.text(x, y-1.0, 'CSPDarknet', fontsize=10, ha='center', color='#666')
    
    @staticmethod
    def _draw_neck_step(ax, x: float, y: float):
        """Neckステップを描画"""
        # Neckの枠
        rect = patches.Rectangle((x-0.8, y-0.6), 1.6, 1.2, 
                               linewidth=2, edgecolor='#ffc107', facecolor='#fff8e1')
        ax.add_patch(rect)
        
        # Neckアイコン
        ax.text(x, y+0.1, '🔗', fontsize=24, ha='center')
        
        # ラベル
        ax.text(x, y-0.8, 'Neck (FPN)', fontsize=12, fontweight='bold', ha='center')
        ax.text(x, y-1.0, 'PANet', fontsize=10, ha='center', color='#666')
    
    @staticmethod
    def _draw_head_step(ax, x: float, y: float):
        """Headステップを描画"""
        # Headの枠
        rect = patches.Rectangle((x-0.8, y-0.6), 1.6, 1.2, 
                               linewidth=2, edgecolor='#fd7e14', facecolor='#fff3e0')
        ax.add_patch(rect)
        
        # Headアイコン
        ax.text(x, y+0.1, '🎯', fontsize=24, ha='center')
        
        # ラベル
        ax.text(x, y-0.8, 'Head', fontsize=12, fontweight='bold', ha='center')
        ax.text(x, y-1.0, 'Anchor-Free', fontsize=10, ha='center', color='#666')
    
    @staticmethod
    def _draw_detection_step(ax, x: float, y: float):
        """検出結果ステップを描画"""
        # 検出結果の枠
        rect = patches.Rectangle((x-0.8, y-0.6), 1.6, 1.2, 
                               linewidth=2, edgecolor='#dc3545', facecolor='#fce4ec')
        ax.add_patch(rect)
        
        # 検出アイコン
        ax.text(x, y+0.1, '📍', fontsize=24, ha='center')
        
        # ラベル
        ax.text(x, y-0.8, '検出結果', fontsize=12, fontweight='bold', ha='center')
        ax.text(x, y-1.0, 'バウンディングボックス', fontsize=10, ha='center', color='#666')
    
    @staticmethod
    def _draw_arrows(ax):
        """矢印を描画"""
        # ステップ間の矢印
        arrow_props = dict(arrowstyle='->', lw=2, color='#333')
        
        # 画像→バックボーン
        ax.annotate('', xy=(2.7, 6), xytext=(2.3, 6), arrowprops=arrow_props)
        ax.text(2.5, 6.3, '特徴抽出', fontsize=9, ha='center', 
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # バックボーン→Neck
        ax.annotate('', xy=(4.7, 6), xytext=(4.3, 6), arrowprops=arrow_props)
        ax.text(4.5, 6.3, 'マルチスケール', fontsize=9, ha='center',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # Neck→Head
        ax.annotate('', xy=(6.7, 6), xytext=(6.3, 6), arrowprops=arrow_props)
        ax.text(6.5, 6.3, '予測', fontsize=9, ha='center',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # Head→検出
        ax.annotate('', xy=(8.7, 6), xytext=(8.3, 6), arrowprops=arrow_props)
        ax.text(8.5, 6.3, '後処理', fontsize=9, ha='center',
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    @staticmethod
    def create_architecture_diagram() -> plt.Figure:
        """
        YOLOv8のアーキテクチャ図を作成します
        
        Returns:
            アーキテクチャ図のmatplotlib Figure
        """
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        # 背景
        ax.set_facecolor('#f8f9fa')
        
        # タイトル
        ax.text(8, 11.5, '🚀 YOLOv8 アーキテクチャ', 
                fontsize=20, fontweight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
        
        # 入力層
        YOLOv8Visualizer._draw_input_layer(ax, 8, 10)
        
        # バックボーン層
        YOLOv8Visualizer._draw_backbone_layers(ax, 8, 8.5)
        
        # Neck層
        YOLOv8Visualizer._draw_neck_layers(ax, 8, 6.5)
        
        # Head層
        YOLOv8Visualizer._draw_head_layers(ax, 8, 4.5)
        
        # 出力層
        YOLOv8Visualizer._draw_output_layer(ax, 8, 2.5)
        
        # 技術仕様
        YOLOv8Visualizer._draw_technical_specs(ax, 1, 1)
        
        # 性能指標
        YOLOv8Visualizer._draw_performance_metrics(ax, 12, 1)
        
        return fig
    
    @staticmethod
    def _draw_input_layer(ax, x: float, y: float):
        """入力層を描画"""
        rect = patches.Rectangle((x-2, y-0.5), 4, 1, linewidth=2, edgecolor='#007bff', facecolor='#e3f2fd')
        ax.add_patch(rect)
        ax.text(x, y, '入力画像 (640x640x3)', fontsize=12, fontweight='bold', ha='center')
    
    @staticmethod
    def _draw_backbone_layers(ax, x: float, y: float):
        """バックボーン層を描画"""
        layers = [
            ('C2f', 0.5),
            ('C2f', 1.0),
            ('C2f', 1.5),
            ('C2f', 2.0),
            ('SPPF', 2.5)
        ]
        
        for i, (layer_name, offset) in enumerate(layers):
            rect = patches.Rectangle((x-1.5, y-offset), 3, 0.4, linewidth=1, edgecolor='#28a745', facecolor='#e8f5e8')
            ax.add_patch(rect)
            ax.text(x, y-offset+0.2, layer_name, fontsize=10, ha='center')
            
            if i < len(layers) - 1:
                ax.annotate('', xy=(x, y-offset-0.2), xytext=(x, y-offset+0.4), 
                           arrowprops=dict(arrowstyle='->', lw=1))
        
        ax.text(x-3, y+0.5, 'バックボーン\n(CSPDarknet)', fontsize=12, fontweight='bold', ha='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='#e8f5e8', alpha=0.8))
    
    @staticmethod
    def _draw_neck_layers(ax, x: float, y: float):
        """Neck層を描画"""
        # 上向きのFPN
        for i in range(3):
            rect = patches.Rectangle((x-1.5, y-i*0.4), 3, 0.3, linewidth=1, edgecolor='#ffc107', facecolor='#fff8e1')
            ax.add_patch(rect)
            ax.text(x, y-i*0.4+0.15, f'P{i+1}', fontsize=10, ha='center')
        
        # 下向きのPAN
        for i in range(2):
            rect = patches.Rectangle((x-1.5, y-1.2-i*0.4), 3, 0.3, linewidth=1, edgecolor='#ffc107', facecolor='#fff8e1')
            ax.add_patch(rect)
            ax.text(x, y-1.2-i*0.4+0.15, f'N{i+1}', fontsize=10, ha='center')
        
        ax.text(x-3, y+0.5, 'Neck\n(PANet)', fontsize=12, fontweight='bold', ha='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='#fff8e1', alpha=0.8))
    
    @staticmethod
    def _draw_head_layers(ax, x: float, y: float):
        """Head層を描画"""
        heads = ['Detection Head', 'Classification Head', 'Regression Head']
        colors = ['#fd7e14', '#fd7e14', '#fd7e14']
        
        for i, (head_name, color) in enumerate(zip(heads, colors)):
            rect = patches.Rectangle((x-1.5, y-i*0.4), 3, 0.3, linewidth=1, edgecolor=color, facecolor='#fff3e0')
            ax.add_patch(rect)
            ax.text(x, y-i*0.4+0.15, head_name, fontsize=10, ha='center')
        
        ax.text(x-3, y+0.5, 'Head\n(Anchor-Free)', fontsize=12, fontweight='bold', ha='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='#fff3e0', alpha=0.8))
    
    @staticmethod
    def _draw_output_layer(ax, x: float, y: float):
        """出力層を描画"""
        rect = patches.Rectangle((x-2, y-0.5), 4, 1, linewidth=2, edgecolor='#dc3545', facecolor='#fce4ec')
        ax.add_patch(rect)
        ax.text(x, y, '検出結果 (バウンディングボックス + クラス)', fontsize=12, fontweight='bold', ha='center')
    
    @staticmethod
    def _draw_technical_specs(ax, x: float, y: float):
        """技術仕様を描画"""
        specs = [
            '🔧 技術仕様:',
            '• パラメータ数: 3.2M (nano)',
            '• 入力サイズ: 640x640',
            '• アーキテクチャ: CSPDarknet',
            '• Neck: PANet (FPN + PAN)',
            '• Head: Anchor-Free',
            '• 損失関数: BCE + CIoU',
            '• 最適化器: AdamW',
            '• 学習率: 0.01'
        ]
        
        for i, spec in enumerate(specs):
            ax.text(x, y-i*0.4, spec, fontsize=10, ha='left',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='#e9ecef', alpha=0.8))
    
    @staticmethod
    def _draw_performance_metrics(ax, x: float, y: float):
        """性能指標を描画"""
        metrics = [
            '📊 性能指標 (COCO):',
            '• mAP@0.5: 0.637 (nano)',
            '• mAP@0.5:0.95: 0.454 (nano)',
            '• 推論速度: 8.7ms (GPU)',
            '• 推論速度: 23.4ms (CPU)',
            '',
            '🏆 特徴:',
            '• リアルタイム検出',
            '• 高精度',
            '• 軽量モデル',
            '• マルチスケール対応'
        ]
        
        for i, metric in enumerate(metrics):
            ax.text(x, y-i*0.4, metric, fontsize=10, ha='left',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='#e9ecef', alpha=0.8))
    
    @staticmethod
    def create_detailed_comparison() -> plt.Figure:
        """
        YOLOv8の詳細比較図を作成します
        
        Returns:
            詳細比較図のmatplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🚀 YOLOv8 詳細比較', fontsize=20, fontweight='bold')
        
        # サブプロット1: モデルサイズ比較
        ax1 = axes[0, 0]
        YOLOv8Visualizer._draw_model_size_comparison(ax1)
        
        # サブプロット2: 精度比較
        ax2 = axes[0, 1]
        YOLOv8Visualizer._draw_accuracy_comparison(ax2)
        
        # サブプロット3: 速度比較
        ax3 = axes[1, 0]
        YOLOv8Visualizer._draw_speed_comparison(ax3)
        
        # サブプロット4: 特徴比較
        ax4 = axes[1, 1]
        YOLOv8Visualizer._draw_feature_comparison(ax4)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def _draw_model_size_comparison(ax):
        """モデルサイズ比較を描画"""
        ax.set_title('📏 モデルサイズ比較', fontsize=14, fontweight='bold')
        
        models = ['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l', 'YOLOv8x']
        params = [3.2, 11.2, 25.9, 43.7, 68.2]  # 百万パラメータ
        
        bars = ax.bar(models, params, color=['#007bff', '#28a745', '#ffc107', '#fd7e14', '#dc3545'])
        ax.set_ylabel('パラメータ数 (M)')
        ax.set_xlabel('モデル')
        
        # バーの上に値を表示
        for bar, param in zip(bars, params):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   f'{param}M', ha='center', va='bottom', fontweight='bold')
    
    @staticmethod
    def _draw_accuracy_comparison(ax):
        """精度比較を描画"""
        ax.set_title('🎯 精度比較 (mAP@0.5)', fontsize=14, fontweight='bold')
        
        models = ['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l', 'YOLOv8x']
        mAP = [0.637, 0.718, 0.764, 0.792, 0.814]
        
        bars = ax.bar(models, mAP, color=['#007bff', '#28a745', '#ffc107', '#fd7e14', '#dc3545'])
        ax.set_ylabel('mAP@0.5')
        ax.set_xlabel('モデル')
        ax.set_ylim(0, 1)
        
        # バーの上に値を表示
        for bar, map_val in zip(bars, mAP):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{map_val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    @staticmethod
    def _draw_speed_comparison(ax):
        """速度比較を描画"""
        ax.set_title('⚡ 推論速度比較 (GPU)', fontsize=14, fontweight='bold')
        
        models = ['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l', 'YOLOv8x']
        fps = [115, 85, 52, 39, 28]  # FPS
        
        bars = ax.bar(models, fps, color=['#007bff', '#28a745', '#ffc107', '#fd7e14', '#dc3545'])
        ax.set_ylabel('FPS')
        ax.set_xlabel('モデル')
        
        # バーの上に値を表示
        for bar, fps_val in zip(bars, fps):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{fps_val}', ha='center', va='bottom', fontweight='bold')
    
    @staticmethod
    def _draw_feature_comparison(ax):
        """特徴比較を描画"""
        ax.set_title('🔧 主要特徴', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        features = [
            '🚀 Anchor-Free設計',
            '🔗 CSPDarknetバックボーン',
            '📊 PANet (FPN + PAN)',
            '🎯 マルチスケール検出',
            '⚡ 高速推論',
            '📱 エッジデバイス対応',
            '🔄 自動学習率調整',
            '📈 高精度検出'
        ]
        
        for i, feature in enumerate(features):
            y_pos = 9 - i * 1.1
            ax.text(1, y_pos, feature, fontsize=11, ha='left',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='#e9ecef', alpha=0.8))


class YOLOv8VisualizationManager:
    """YOLOv8の可視化を管理するクラス"""
    
    @staticmethod
    def display_yolo_visualization() -> None:
        """YOLOv8の可視化を表示します"""
        st.header("🚀 YOLOv8 可視化")
        
        # 可視化タイプの選択
        viz_type = st.selectbox(
            "可視化タイプを選択",
            ["処理フロー図", "アーキテクチャ図", "詳細比較図"],
            help="YOLOv8の異なる側面を可視化"
        )
        
        if viz_type == "処理フロー図":
            fig = YOLOv8Visualizer.create_processing_flow_diagram()
            st.pyplot(fig)
            
            st.markdown("""
            **処理フロー図の説明:**
            - 📷 **画像入力**: 640x640のRGB画像を入力
            - 🔗 **バックボーン**: CSPDarknetによる特徴抽出
            - 🔗 **Neck (FPN)**: PANetによるマルチスケール特徴融合
            - 🎯 **Head**: Anchor-Free設計による予測
            - 📍 **検出結果**: バウンディングボックスとクラス予測
            """)
            
        elif viz_type == "アーキテクチャ図":
            fig = YOLOv8Visualizer.create_architecture_diagram()
            st.pyplot(fig)
            
            st.markdown("""
            **アーキテクチャ図の説明:**
            
            **🔧 技術仕様:**
            - パラメータ数: 3.2M (nano) 〜 68.2M (xlarge)
            - 入力サイズ: 640x640ピクセル
            - アーキテクチャ: CSPDarknet + PANet + Anchor-Free Head
            
            **📊 性能指標:**
            - mAP@0.5: 0.637 (nano) 〜 0.814 (xlarge)
            - 推論速度: 8.7ms (GPU) 〜 35.7ms (GPU)
            - リアルタイム性能: 28-115 FPS
            """)
            
        elif viz_type == "詳細比較図":
            fig = YOLOv8Visualizer.create_detailed_comparison()
            st.pyplot(fig)
            
            st.markdown("""
            **詳細比較図の説明:**
            
            この図は、YOLOv8の各モデルサイズの性能を比較しています。
            
            **モデル選択のガイド:**
            - **YOLOv8n**: 軽量で高速、エッジデバイス向け
            - **YOLOv8s**: バランス型、一般的な用途
            - **YOLOv8m**: 高精度、サーバー向け
            - **YOLOv8l**: 最高精度、研究用途
            - **YOLOv8x**: 最大精度、特殊用途
            """)
        
        # 技術的な詳細情報
        with st.expander("🔧 技術的な詳細"):
            st.markdown("""
            **YOLOv8のアーキテクチャ詳細:**
            
            **1. CSPDarknet バックボーン**
            - CSP (Cross Stage Partial) 接続
            - 効率的な特徴抽出
            - 軽量で高性能
            
            **2. PANet (Path Aggregation Network)**
            - FPN (Feature Pyramid Network) + PAN
            - マルチスケール特徴融合
            - 小物体検出性能向上
            
            **3. Anchor-Free Head**
            - アンカーボックス不要
            - 直接的な座標予測
            - シンプルで効率的
            
            **4. 損失関数**
            - BCE (Binary Cross Entropy): 分類損失
            - CIoU (Complete IoU): 回帰損失
            - DFL (Distribution Focal Loss): 分布学習
            
            **5. 最適化**
            - AdamW 最適化器
            - 自動学習率調整
            - 余弦アニーリング
            """)
        
        # パフォーマンス情報
        with st.expander("⚡ パフォーマンス情報"):
            st.markdown("""
            **処理時間の目安 (RTX 3080):**
            
            | モデル | パラメータ数 | mAP@0.5 | FPS | メモリ使用量 |
            |--------|-------------|---------|-----|-------------|
            | YOLOv8n | 3.2M | 0.637 | 115 | ~1GB |
            | YOLOv8s | 11.2M | 0.718 | 85 | ~2GB |
            | YOLOv8m | 25.9M | 0.764 | 52 | ~3GB |
            | YOLOv8l | 43.7M | 0.792 | 39 | ~4GB |
            | YOLOv8x | 68.2M | 0.814 | 28 | ~6GB |
            
            **精度と速度のトレードオフ:**
            - 軽量モデル: 高速だが精度が低い
            - 大規模モデル: 高精度だが低速
            - 用途に応じて適切なモデルを選択
            
            **エッジデバイス対応:**
            - YOLOv8n: モバイル、IoTデバイス
            - YOLOv8s: 組み込みシステム
            - 量子化、プルーニング対応
            """)
        
        # YOLOv8の特徴
        with st.expander("🏆 YOLOv8の特徴"):
            st.markdown("""
            **🚀 主要な特徴:**
            
            **1. Anchor-Free設計**
            - アンカーボックスが不要
            - シンプルで効率的
            - 学習が容易
            
            **2. マルチスケール検出**
            - 異なるサイズの物体を検出
            - FPN + PANによる特徴融合
            - 小物体検出性能向上
            
            **3. リアルタイム性能**
            - 高速推論
            - 低レイテンシー
            - エッジデバイス対応
            
            **4. 高精度**
            - 最新の損失関数
            - 効率的なアーキテクチャ
            - データ拡張技術
            
            **5. 使いやすさ**
            - シンプルなAPI
            - 豊富なドキュメント
            - 活発なコミュニティ
            """) 