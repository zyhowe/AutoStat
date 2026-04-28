"""
BERT模型下载脚本 - 使用国内镜像源
"""

import os
import sys
from pathlib import Path

# 设置环境变量使用国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 模型配置
MODELS = {
    "bert-base-chinese": {
        "url": "https://hf-mirror.com/google-bert/bert-base-chinese",
        "description": "BERT中文基础模型 (约400MB)"
    },
    "chinese-roberta-wwm-ext": {
        "url": "https://hf-mirror.com/hfl/chinese-roberta-wwm-ext",
        "description": "哈工大中文RoBERTa (约420MB)"
    }
}

DEFAULT_MODEL = "bert-base-chinese"
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"


def setup_mirror():
    """设置镜像源"""
    print("=" * 60)
    print("🔧 设置 HuggingFace 镜像源")
    print("=" * 60)

    # 方案1：使用 hf-mirror.com
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    print("✅ 已设置镜像: https://hf-mirror.com")

    # 方案2：也可以使用其他镜像（注释备用）
    # os.environ['HF_ENDPOINT'] = 'https://huggingface.su'
    # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def download_with_transformers(model_name: str, cache_dir: str = None):
    """使用 transformers 下载模型"""
    print("\n" + "=" * 60)
    print(f"📥 下载模型: {model_name}")
    print("=" * 60)

    try:
        from transformers import AutoTokenizer, AutoModel

        print(f"⏳ 正在下载 {model_name}...")
        print(f"   缓存目录: {cache_dir or CACHE_DIR}")
        print(f"   镜像源: {os.environ.get('HF_ENDPOINT', '官方源')}")
        print("")

        # 下载 tokenizer
        print("  📝 下载 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            resume_download=True
        )
        print("  ✅ Tokenizer 下载完成")

        # 下载模型
        print("  🧠 下载模型权重...")
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            resume_download=True
        )
        print("  ✅ 模型权重下载完成")

        print(f"\n✅ 模型 {model_name} 下载成功！")
        print(f"   缓存位置: {cache_dir or CACHE_DIR}")
        return True

    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        return False


def download_with_requests(model_name: str, save_dir: str = None):
    """使用 requests 直接下载（备选方案）"""
    print("\n" + "=" * 60)
    print(f"📥 备选方案: 直接下载 {model_name}")
    print("=" * 60)

    if save_dir is None:
        save_dir = str(CACHE_DIR / f"models--{model_name.replace('/', '--')}")

    os.makedirs(save_dir, exist_ok=True)

    # 需要下载的文件列表
    files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "pytorch_model.bin"
    ]

    base_url = f"{os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')}/{model_name}/resolve/main"

    success_count = 0
    for file_name in files:
        file_url = f"{base_url}/{file_name}"
        file_path = os.path.join(save_dir, file_name)

        if os.path.exists(file_path):
            print(f"  ⏭️ 跳过已存在: {file_name}")
            success_count += 1
            continue

        print(f"  ⏳ 下载: {file_name}")
        try:
            import requests
            response = requests.get(file_url, timeout=60, stream=True)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"  ✅ 完成: {file_name}")
                success_count += 1
            else:
                print(f"  ❌ 失败: {file_name} (HTTP {response.status_code})")
        except Exception as e:
            print(f"  ❌ 失败: {file_name} - {e}")

    print(f"\n下载完成: {success_count}/{len(files)} 个文件")
    return success_count == len(files)


def test_model(model_name: str):
    """测试模型是否可用"""
    print("\n" + "=" * 60)
    print("🧪 测试模型加载")
    print("=" * 60)

    try:
        from transformers import AutoTokenizer, AutoModel
        import torch

        print(f"⏳ 加载模型 {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # 测试推理
        test_text = "你好，世界！"
        inputs = tokenizer(test_text, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        embedding = outputs.last_hidden_state[:, 0, :].numpy()
        print(f"✅ 模型测试成功！")
        print(f"   输入: {test_text}")
        print(f"   输出向量维度: {embedding.shape}")
        return True

    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return False


def list_cached_models():
    """列出已缓存的模型"""
    print("\n" + "=" * 60)
    print("📦 已缓存的模型")
    print("=" * 60)

    cache_path = CACHE_DIR
    if not cache_path.exists():
        print("  暂无缓存模型")
        return []

    models = []
    for item in cache_path.iterdir():
        if item.is_dir() and "models--" in item.name:
            model_name = item.name.replace("models--", "").replace("--", "/")
            size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
            size_mb = size / (1024 * 1024)
            models.append({
                "name": model_name,
                "path": item,
                "size_mb": size_mb
            })
            print(f"  📁 {model_name}: {size_mb:.1f} MB")

    if not models:
        print("  暂无缓存模型")
    return models


def clear_cache(model_name: str = None):
    """清理缓存"""
    print("\n" + "=" * 60)
    print("🗑️ 清理缓存")
    print("=" * 60)

    import shutil

    if model_name:
        cache_name = f"models--{model_name.replace('/', '--')}"
        cache_path = CACHE_DIR / cache_name
        if cache_path.exists():
            shutil.rmtree(cache_path)
            print(f"✅ 已删除: {model_name}")
        else:
            print(f"❌ 未找到: {model_name}")
    else:
        # 清理所有
        for item in CACHE_DIR.iterdir():
            if item.is_dir() and "models--" in item.name:
                shutil.rmtree(item)
                print(f"✅ 已删除: {item.name}")
        print("✅ 缓存已清理")


def main():
    """主函数"""
    print("=" * 60)
    print("  BERT 模型下载工具")
    print("=" * 60)

    # 设置镜像源
    setup_mirror()

    # 显示已缓存的模型
    list_cached_models()

    print("\n" + "=" * 60)
    print("  选择要下载的模型")
    print("=" * 60)

    for key, info in MODELS.items():
        print(f"  {key}: {info['description']}")

    print(f"\n默认: {DEFAULT_MODEL}")
    print("提示: 按回车使用默认，或输入模型名称")

    # 获取用户输入
    choice = input("\n请输入模型名称: ").strip()
    if not choice:
        model_name = DEFAULT_MODEL
    elif choice in MODELS:
        model_name = choice
    else:
        print(f"❌ 未知模型: {choice}")
        print(f"   使用默认: {DEFAULT_MODEL}")
        model_name = DEFAULT_MODEL

    print(f"\n开始下载: {model_name}")

    # 下载模型
    success = download_with_transformers(model_name)

    if success:
        # 测试模型
        test_model(model_name)
        print("\n" + "=" * 60)
        print("🎉 下载完成！")
        print("=" * 60)
        print("\n现在可以运行文本分析程序了")
    else:
        print("\n⚠️ 自动下载失败，请尝试手动下载")
        print(f"   访问: {os.environ.get('HF_ENDPOINT')}/{model_name}")
        print("   将文件放入: " + str(CACHE_DIR))


if __name__ == "__main__":
    main()