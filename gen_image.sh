#!/bin/bash
# GPT 图片生成：一键调用脚本
# 用法: ./gen_image.sh "图片描述" [尺寸] [参考图路径]
# 示例: ./gen_image.sh "一只橘猫"                    # 默认 1024x1024
# 示例: ./gen_image.sh "旅行海报" "1024x1365"       # 3:4 竖版
# 示例: ./gen_image.sh "保持面部" "1024x1536" "/path/to/ref.png"  # 带参考图
#
# API Key: 自动从 ~/.openclaw/workspace/.env 读取

PROMPT="$1"
SIZE="${2:-1024x1024}"
REF="${3:-}"
SKILL_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- 检查 ----
if [ -z "$PROMPT" ]; then
    echo "用法: $0 \"图片描述\" [尺寸] [参考图路径]"
    echo "示例: $0 \"一只橘猫\" \"1024x1365\""
    echo "示例: $0 \"保持面部特征\" \"1024x1536\" \"/path/to/ref.png\""
    exit 1
fi

# ---- 执行 ----
cd "$SKILL_DIR" && \
uv run --with Pillow,requests python gpt_image_api.py "$PROMPT" "$SIZE" "$REF"
