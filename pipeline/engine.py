#!/usr/bin/env python3
"""
艺游未境-Wanderix · Pipeline 引擎
支持模板化工作流，PLOC 手账涂鸦等自动化 Pipeline

模板加载流程：
  1. registry.json 读取所有模板元数据
  2. 动态加载模板 JSON
  3. VLM 识别 → 模板注入 → GPT Image 生成
"""
import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Optional

# ── 路径 ────────────────────────────────────────────────────────────────────
SKILL_DIR   = Path(__file__).parent.parent
TEMPLATE_DIR = SKILL_DIR / "templates"
REGISTRY_FILE = TEMPLATE_DIR / "registry.json"

# ── 模板注册表 ──────────────────────────────────────────────────────────────
def load_registry() -> dict:
    if not REGISTRY_FILE.exists():
        return {"templates": {}}
    with open(REGISTRY_FILE, encoding="utf-8") as f:
        return json.load(f)


def save_registry(registry: dict):
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


# ── 模板加载 ────────────────────────────────────────────────────────────────
def load_template(name: str) -> dict:
    """按名称加载模板，返回模板字典"""
    registry = load_registry()
    if name not in registry.get("templates", {}):
        raise ValueError(f"模板 '{name}' 不存在，可用: {list(registry.get('templates', {}).keys())}")

    tmpl_meta = registry["templates"][name]
    tmpl_path = TEMPLATE_DIR / tmpl_meta["path"]
    if not tmpl_path.exists():
        raise FileNotFoundError(f"模板文件不存在: {tmpl_path}")

    with open(tmpl_path, encoding="utf-8") as f:
        return json.load(f)


# ── VLM 识别 ────────────────────────────────────────────────────────────────
def run_vlm_identify(image_path: str, prompt: str = "列出图片中可标注的元素") -> str:
    """
    调用 Ollama VLM 识别图像元素。
    返回识别结果文本。
    """
    vlm_script = SKILL_DIR.parent / "ollama_vlm.py"
    if not vlm_script.exists():
        # 降级：返回占位符
        return "（VLM 脚本不可用，请手动指定标注元素）"

    cmd = [
        sys.executable, str(vlm_script),
        prompt, image_path
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60
        )
        return result.stdout.strip() if result.stdout else "（VLM 无输出）"
    except subprocess.TimeoutExpired:
        return "（VLM 识别超时）"
    except Exception as e:
        return f"（VLM 错误: {e}）"


# ── 标注文案生成 ─────────────────────────────────────────────────────────────
def generate_ploc_annotations(vlm_result: str, template_name: str) -> list[str]:
    """
    根据 VLM 识别结果生成 3-5 条繁体大字标注。
    这里做基础格式化，具体内容由 GPT 二次生成。
    """
    lines = vlm_result.strip().split("\n")
    # 取前 5 个元素，格式化为标注文案
    annotations = []
    for line in lines[:5]:
        cleaned = line.strip().strip("-,.。").strip()
        if cleaned:
            annotations.append(cleaned)
    return annotations


# ── Pipeline 主函数 ──────────────────────────────────────────────────────────
def run_pipeline(
    template_name: str,
    input_image: str,
    output_path: Optional[str] = None,
    vlm_result: Optional[str] = None,
) -> str:
    """
    执行 Pipeline 工作流。

    参数:
      template_name: 模板名，如 "PLOC_治愈手账涂鸦"
      input_image:   输入图片路径
      output_path:   输出图片路径（可选）
      vlm_result:    VLM 识别结果（可选，跳过识别步骤）
    """
    # 1. 加载模板
    tmpl = load_template(template_name)

    # 2. VLM 识别（如未提供）
    if vlm_result is None:
        vlm_result = run_vlm_identify(input_image)

    # 3. 生成标注
    annotations = generate_ploc_annotations(vlm_result, template_name)
    annotation_text = " ✦ ".join(annotations)

    # 4. 构建完整提示词
    prompt_template = tmpl.get("prompt_template", "")
    full_prompt = prompt_template + f"\n附上溫柔治愈繁體手寫大字註解：{annotation_text}"

    # 5. 调用 gpt_image_api 生成
    sys.path.insert(0, str(SKILL_DIR))
    from gpt_image_api import generate_image, _GATES_CONFIRMED

    result = generate_image(
        prompt=full_prompt,
        output_path=output_path,
        size_raw=tmpl.get("default_size", "3:4"),
        reference_images=[input_image],
        gates_confirmed=_GATES_CONFIRMED,
    )

    if result:
        return result
    raise RuntimeError(f"Pipeline 生成失败（模板: {template_name}）")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="艺游未境 Pipeline 引擎")
    parser.add_argument("command", choices=["run", "list", "register"],
                        help="run=执行Pipeline / list=列出模板 / register=注册模板")
    parser.add_argument("template", nargs="?", help="模板名（run/register 时用）")
    parser.add_argument("--input", "-i", help="输入图片路径")
    parser.add_argument("--output", "-o", help="输出图片路径")
    parser.add_argument("--vlm", help="VLM 识别结果（跳过 VLM 步骤）")

    args = parser.parse_args()

    if args.command == "list":
        registry = load_registry()
        print(f"\n📋 已注册模板 ({len(registry.get('templates', {}))}):\n")
        for name, meta in registry.get("templates", {}).items():
            print(f"  ▶ {name}")
            print(f"    用途: {meta.get('description', 'N/A')}")
            print(f"    文件: {meta.get('path', 'N/A')}")
            print()
        return

    elif args.command == "run":
        if not args.template:
            print("❌ 需要指定模板名: pipeline.py run <模板名> --input <图片>")
            sys.exit(1)
        if not args.input:
            print("❌ 需要指定输入图片: pipeline.py run <模板名> --input <图片>")
            sys.exit(1)

        output = run_pipeline(
            template_name=args.template,
            input_image=args.input,
            output_path=args.output,
            vlm_result=args.vlm,
        )
        print(f"\n✅ Pipeline 完成: {output}")
        return

    elif args.command == "register":
        print("❌ register 功能开发中，请手动编辑 registry.json")
        sys.exit(1)


if __name__ == "__main__":
    main()
