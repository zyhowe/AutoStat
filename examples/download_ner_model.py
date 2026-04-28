"""
NER 模型下载脚本 - 使用 huggingface-cli 或直接下载
"""

import os
import subprocess
import sys
from pathlib import Path

# 设置国内镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

MODEL_ID = "shibing624/bert4ner-base-chinese"
LOCAL_DIR = Path("./models/bert4ner")
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"


def download_with_cli():
    """使用 huggingface-cli 下载"""
    print("\n📥 使用 huggingface-cli 下载模型...")
    print(f"   模型: {MODEL_ID}")
    print(f"   目标目录: {LOCAL_DIR.absolute()}")

    # 确保目录存在
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    # 构建命令
    cmd = [
        sys.executable, "-m", "huggingface_hub.commands.huggingface_cli",
        "download", MODEL_ID,
        "--local-dir", str(LOCAL_DIR),
        "--local-dir-use-symlinks", "False"
    ]

    print(f"   执行命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=False, check=True)
        print(f"✅ 下载完成: {LOCAL_DIR.absolute()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 下载失败: {e}")
        return False


def download_with_python():
    """使用 Python API 下载（带重试）"""
    print("\n📥 使用 Python API 下载模型...")

    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(LOCAL_DIR),
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4,
            endpoint="https://hf-mirror.com"
        )
        print(f"✅ 下载完成: {LOCAL_DIR.absolute()}")
        return True
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False


def verify_model():
    """验证模型是否可用"""
    print("\n🔍 验证模型...")

    try:
        # 设置镜像
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

        from transformers import AutoTokenizer, AutoModelForTokenClassification

        # 从本地加载
        tokenizer = AutoTokenizer.from_pretrained(
            str(LOCAL_DIR),
            trust_remote_code=True,
            local_files_only=True
        )
        model = AutoModelForTokenClassification.from_pretrained(
            str(LOCAL_DIR),
            trust_remote_code=True,
            local_files_only=True
        )

        print(f"✅ 模型验证成功！")
        print(f"   支持实体类型: {len(model.config.id2label)} 种")
        labels = list(model.config.id2label.values())
        print(f"   标签示例: {labels[:10]}...")

        return True

    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False


def main():
    print("=" * 60)
    print("  NER 模型下载工具")
    print("=" * 60)
    print(f"\n模型: {MODEL_ID}")
    print(f"本地路径: {LOCAL_DIR.absolute()}")

    # 检查是否已存在
    if LOCAL_DIR.exists() and (LOCAL_DIR / "config.json").exists():
        print("\n✅ 模型已存在，跳过下载")
        verify_model()
        return

    # 尝试下载
    print("\n选择下载方式:")
    print("  1. huggingface-cli (推荐) - 需要先安装 huggingface_hub")
    print("  2. Python API")

    choice = input("\n请输入选择 [1/2] (默认1): ").strip() or "1"

    success = False
    if choice == "1":
        # 检查 huggingface_cli 是否可用
        try:
            subprocess.run([sys.executable, "-m", "huggingface_hub.commands.huggingface_cli", "--help"],
                           capture_output=True, check=True)
            success = download_with_cli()
        except:
            print("⚠️ huggingface-cli 不可用，请先安装: pip install -U huggingface_hub")
            success = download_with_python()
    else:
        success = download_with_python()

    if success:
        verify_model()
    else:
        print("\n⚠️ 下载失败，请尝试手动下载")
        print(f"   访问: https://hf-mirror.com/{MODEL_ID}")
        print(f"   将文件放入: {LOCAL_DIR.absolute()}")


if __name__ == "__main__":
    main()