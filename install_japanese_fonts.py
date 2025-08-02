#!/usr/bin/env python3
"""
日本語フォントインストールスクリプト
YOLOアプリケーションで日本語表示を改善するためのフォントをインストール
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def install_japanese_fonts():
    """日本語フォントをインストール"""
    system = platform.system()
    
    print("🌏 日本語フォントのインストールを開始します...")
    
    if system == "Linux":
        install_japanese_fonts_linux()
    elif system == "Darwin":  # macOS
        install_japanese_fonts_macos()
    elif system == "Windows":
        install_japanese_fonts_windows()
    else:
        print(f"❌ サポートされていないOS: {system}")
        return False
    
    return True


def install_japanese_fonts_linux():
    """Linuxでの日本語フォントインストール"""
    print("🐧 Linux環境で日本語フォントをインストール中...")
    
    # パッケージマネージャーを検出
    if os.path.exists("/etc/debian_version"):
        # Debian/Ubuntu系
        packages = [
            "fonts-noto-cjk",
            "fonts-noto-cjk-extra",
            "fonts-ipafont",
            "fonts-ipafont-gothic",
            "fonts-ipafont-mincho"
        ]
        
        try:
            subprocess.run(["sudo", "apt", "update"], check=True)
            subprocess.run(["sudo", "apt", "install", "-y"] + packages, check=True)
            print("✅ Debian/Ubuntu系で日本語フォントのインストールが完了しました")
        except subprocess.CalledProcessError as e:
            print(f"❌ パッケージインストールに失敗しました: {e}")
            return False
            
    elif os.path.exists("/etc/redhat-release"):
        # RedHat/CentOS系
        packages = [
            "google-noto-cjk-fonts",
            "ipa-gothic-fonts",
            "ipa-mincho-fonts"
        ]
        
        try:
            subprocess.run(["sudo", "yum", "install", "-y"] + packages, check=True)
            print("✅ RedHat/CentOS系で日本語フォントのインストールが完了しました")
        except subprocess.CalledProcessError as e:
            print(f"❌ パッケージインストールに失敗しました: {e}")
            return False
    
    else:
        print("⚠️ サポートされていないLinuxディストリビューションです")
        print("手動で日本語フォントをインストールしてください")
        return False
    
    # フォントキャッシュの更新
    try:
        subprocess.run(["fc-cache", "-fv"], check=True)
        print("✅ フォントキャッシュを更新しました")
    except subprocess.CalledProcessError:
        print("⚠️ フォントキャッシュの更新に失敗しました")
    
    return True


def install_japanese_fonts_macos():
    """macOSでの日本語フォントインストール"""
    print("🍎 macOS環境で日本語フォントをインストール中...")
    
    # macOSには日本語フォントが標準で含まれている
    print("✅ macOSには日本語フォントが標準で含まれています")
    print("フォント設定は自動的に行われます")
    
    return True


def install_japanese_fonts_windows():
    """Windowsでの日本語フォントインストール"""
    print("🪟 Windows環境で日本語フォントをインストール中...")
    
    # Windowsには日本語フォントが標準で含まれている
    print("✅ Windowsには日本語フォントが標準で含まれています")
    print("フォント設定は自動的に行われます")
    
    return True


def check_japanese_fonts():
    """日本語フォントの確認"""
    print("🔍 日本語フォントの確認中...")
    
    try:
        import matplotlib.font_manager as fm
        
        # 利用可能なフォントを取得
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 日本語フォントのリスト
        japanese_fonts = [
            'Noto Sans CJK JP',
            'Noto Sans JP', 
            'Hiragino Sans',
            'Yu Gothic',
            'Meiryo',
            'Takao',
            'IPAexGothic',
            'IPAPGothic',
            'VL PGothic'
        ]
        
        found_fonts = []
        for font in japanese_fonts:
            if font in available_fonts:
                found_fonts.append(font)
        
        if found_fonts:
            print(f"✅ 日本語フォントが見つかりました: {', '.join(found_fonts)}")
            return True
        else:
            print("❌ 日本語フォントが見つかりませんでした")
            return False
            
    except ImportError:
        print("❌ matplotlibがインストールされていません")
        return False


def main():
    """メイン関数"""
    print("🚀 YOLOアプリケーション用日本語フォントインストーラー")
    print("=" * 50)
    
    # 現在のフォント状況を確認
    if check_japanese_fonts():
        print("✅ 日本語フォントは既に利用可能です")
        return
    
    # フォントのインストール
    if install_japanese_fonts():
        print("\n🔄 フォントキャッシュを更新中...")
        
        # 再度フォントを確認
        if check_japanese_fonts():
            print("✅ 日本語フォントのインストールが完了しました")
            print("YOLOアプリケーションを再起動してください")
        else:
            print("❌ 日本語フォントのインストールに失敗しました")
    else:
        print("❌ フォントインストールに失敗しました")


if __name__ == "__main__":
    main() 