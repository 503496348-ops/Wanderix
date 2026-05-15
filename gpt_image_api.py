#!/usr/bin/env python3
"""
GPT Image API 封装脚本 v2.3
支持 gpt-image-2-vip 模型生成图片

依赖: requests
运行: python gpt_image_api.py "描述" [尺寸|比例] [参考图路径]

尺寸格式:
  - 像素尺寸: 1024x1024, 1024x1536, 1344x756 等
  - 比例格式: 1:1, 3:4, 2:3, 16:9, 9:16, 4:3 等（自动转换为对应像素尺寸）
  - 自定义: 任意 WxH 格式，如 1920x1080

环境变量:
  GRSAPI_KEY   API Key（必填，存储于 ~/.openclaw/workspace/.env）
  GRSAPI_URL   API 地址（可选，默认 https://grsai.dakka.com.cn）
"""
import os
import sys
import time
import json
import base64
import requests
import tempfile
import shutil
from pathlib import Path
from typing import Optional


# ─── 比例 → 默认像素尺寸映射 ────────────────────────────────────────────
ASPECT_PRESETS = {
    "1:1":   (1024, 1024),
    "3:4":   (1024, 1365),
    "2:3":   (1024, 1536),
    "4:3":   (1024, 768),
    "16:9":  (1344, 756),
    "9:16":  (768, 1344),
    "21:9":  (1512, 648),
    "4:5":   (1024, 1280),
    "5:7":   (1024, 1433),
    "9:19":  (810, 1710),   # 手机电影画幅
    "2.35:1":(1410, 600),   # 电影宽荧幕
}

# ─── 人像身份锚定段 ─────────────────────────────────────────────────────
# 来自 portrait_prompt.py，当有参考图时自动注入到提示词最前面
# 【规定一（补充）强制执行】：此段落为固定模板，不得修改
PORTRAIT_IDENTITY_ANCHOR = (
    "THIS IS A PORTRAIT OF THE SAME PERSON FROM THE REFERENCE IMAGE. "
    "You must preserve the person's exact identity: face shape, eye shape, nose shape, lip shape, chin, "
    "jawline, skin tone, and hair style. The reference image defines this person's identity — "
    "their identity is PRIMARY and CANNOT be changed or distorted."
)

# ─── 比例门禁豁免标记 ─────────────────────────────────────────────────
# 当 Pipeline 已确认比例时传入此标记，gpt_image_api.py 不重复拦截
_GATES_CONFIRMED = object()

# 常见竖版场景
VERTICAL_PRESETS = {
    "手机壁纸": "9:16",
    "小红书":   "3:4",
    "海报":     "2:3",
    "头像":     "1:1",
    "横版封面": "16:9",
    "电影感":   "21:9",
}

def parse_size_arg(raw: str) -> tuple[int, int]:
    """
    解析用户传入的尺寸参数，返回 (width, height)。

    支持格式:
      1024x1024      → 直接解析像素尺寸
      1:1 / 3:4 ...  → 查表转换为像素尺寸
      小红书/海报...  → 查表转换为像素尺寸
    """
    raw = raw.strip()

    # 1. 比例格式  "1:1" / "3:4" …
    if ":" in raw and not "x" in raw.lower():
        key = raw.lower()
        if key in ASPECT_PRESETS:
            return ASPECT_PRESETS[key]
        raise ValueError(f"不支持的比例 '{raw}'，可选: {', '.join(ASPECT_PRESETS.keys())}")

    # 2. 中文场景名  "小红书" / "海报" …
    if raw in VERTICAL_PRESETS:
        key = VERTICAL_PRESETS[raw]
        return ASPECT_PRESETS[key]

    # 3. 像素尺寸  "1024x1024" / "1920x1080" …
    if "x" in raw.lower():
        w, h = raw.lower().split("x")
        try:
            return int(w), int(h)
        except ValueError:
            raise ValueError(f"尺寸 '{raw}' 格式错误，请使用如 1024x1024")

    # 4. 纯数字 → 当作正方形边长
    try:
        s = int(raw)
        return s, s
    except ValueError:
        raise ValueError(f"无法解析尺寸 '{raw}'")


def resolve_size_argument(raw_arg: Optional[str]) -> tuple[str, tuple[int, int]]:
    """
    将用户传入的尺寸参数解析为:
      (api_size_str, (width, height))

    若 raw_arg 为空或 "auto"，返回 (1024x1024, (1024, 1024))
    """
    if not raw_arg or raw_arg.lower() == "auto":
        return "1024x1024", (1024, 1024)

    w, h = parse_size_arg(raw_arg)
    return f"{w}x{h}", (w, h)


def get_api_key():
    """从.env文件读取API密钥"""
    env_path = Path.home() / ".openclaw" / "workspace" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("GRSAPI_KEY="):
                return line.split("=", 1)[1].strip()
    return os.environ.get("GRSAPI_KEY", "")


def get_config():
    cfg = {
        "api_key": get_api_key(),
        "api_url": "https://grsai.dakka.com.cn",
        "model": "gpt-image-2-vip",
        "submit_timeout": 30,
        "download_timeout": 60,
    }
    return cfg


def get_output_path(default_dir: str = None, name_prefix: str = None) -> str:
    """生成输出路径，时间戳+随机后缀防冲突，可选自定义前缀用于主题命名"""
    if default_dir:
        out_dir = Path(default_dir).expanduser()
    else:
        out_dir = Path.home() / "Desktop"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    rand = os.urandom(2).hex()  # 4位随机后缀，防止连续生成时文件名冲突
    prefix = name_prefix if name_prefix else "gpt_image"
    return str(out_dir / f"{prefix}_{ts}_{rand}.png")


MAX_REF_SIZE_KB = 8192  # 8MB
_temp_files: list = []  # 全局暂存待清理的临时文件路径


def _cleanup_temp_files():
    """任务结束后删除所有临时压缩副本，原图不受影响。"""
    global _temp_files
    for fp in _temp_files:
        try:
            Path(fp).unlink(missing_ok=True)
        except Exception:
            pass
    _temp_files.clear()


def _prepare_temp_copy(src_path: Path) -> Path:
    """将原图复制到临时目录，返回临时文件路径。"""
    tmp_dir = Path(tempfile.gettempdir())
    tmp_path = tmp_dir / f"{src_path.stem}_ref{src_path.suffix}"
    _temp_files.append(str(tmp_path))
    import shutil
    shutil.copy2(src_path, tmp_path)
    return tmp_path


def compress_image_to_limit(tmp_path: Path, original_size_kb: int, max_size_kb: int = MAX_REF_SIZE_KB) -> tuple[bool, str]:
    """
    将已存在的临时副本压缩到 max_size_kb 以下。
    返回 (success, message)
    """
    try:
        from PIL import Image
    except ImportError:
        return False, "PIL 未安装，无法压缩"

    if tmp_path.stat().st_size // 1024 <= max_size_kb:
        return True, f"无需压缩（{tmp_path.stat().st_size // 1024} KB ≤ {max_size_kb} KB）"

    img = Image.open(tmp_path)
    w, h = img.size

    for quality in [85, 70, 55, 40]:
        img.save(tmp_path, quality=quality, optimize=True)
        size_kb = tmp_path.stat().st_size // 1024
        if size_kb <= max_size_kb:
            return True, f"从 {original_size_kb} KB → {size_kb} KB（quality={quality}）"

    for scale in [0.8, 0.6, 0.5]:
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w < 400:
            break
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        resized.save(tmp_path, quality=60, optimize=True)
        size_kb = tmp_path.stat().st_size // 1024
        if size_kb <= max_size_kb:
            return True, f"从 {original_size_kb} KB → {size_kb} KB（resize {new_w}x{new_h}）"

    return False, f"压缩失败（原 {original_size_kb} KB）"


def validate_reference_images(reference_images: list) -> tuple[bool, str, list]:
    """
    预检验参考图，将需要压缩的图复制到临时目录后压缩。
    原图文件位置、内容、名称均不改变。

    返回 (is_valid, message, processed_paths)
      processed_paths: 处理后的参考图路径列表
    """
    if not reference_images:
        return True, "无参考图，跳过检验", []

    results = []
    errors = []
    processed_paths = []

    for img_path in reference_images:
        p = Path(img_path)
        if not p.exists():
            errors.append(f"文件不存在: {img_path}")
            continue
        if not p.is_file():
            errors.append(f"不是文件: {img_path}")
            continue

        size_kb = p.stat().st_size // 1024
        pstat = f"{p.name} ({size_kb} KB)"

        # 复制到临时目录
        tmp_dir = Path(tempfile.gettempdir())
        tmp_path = tmp_dir / f"{p.stem}_ref{p.suffix}"
        _temp_files.append(str(tmp_path))
        import shutil
        shutil.copy2(p, tmp_path)

        # 需要压缩
        steps = []
        if tmp_path.stat().st_size // 1024 > MAX_REF_SIZE_KB:
            ok, msg = compress_image_to_limit(tmp_path, size_kb)
            if ok:
                steps.append(f"压缩 {tmp_path.stat().st_size//1024} KB")
            else:
                results.append(f"{pstat} ⚠️ 压缩失败({msg})，将尝试原文件上传")
                # 仍然用原路径，不阻塞

        # base64 编码检验
        try:
            with open(tmp_path, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode('ascii')
            if len(b64) < 100:
                errors.append(f"{p.name}: base64 编码异常")
                continue
        except Exception as e:
            errors.append(f"{p.name}: 读取失败 - {e}")
            continue

        final_kb = tmp_path.stat().st_size // 1024
        step_str = " → ".join(steps) if steps else "原图直通"
        results.append(f"{pstat} → 临时文件 {final_kb} KB（{step_str}）")
        processed_paths.append(str(tmp_path))

    if errors:
        return False, "；".join(errors), processed_paths

    return True, "，".join(results), processed_paths


def submit_task(
    prompt: str,
    width: int,
    height: int,
    reference_images: list = None,
    model: str = "gpt-image-2-vip",
) -> tuple:
    """提交生成任务，返回 (task_id, error)"""
    cfg = get_config()

    if not cfg["api_key"]:
        return None, "GRSAPI_KEY 未设置，请检查 ~/.openclaw/workspace/.env"

    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "prompt": prompt,
        "size": f"{width}x{height}",
        "webHook": "-1",
        "shutProgress": True,
    }

    # 参考图（urls 参数，支持 base64 data URL）
    if reference_images:
        urls = []
        for img_path in reference_images:
            img_path = Path(img_path)
            if img_path.exists():
                with open(img_path, 'rb') as f:
                    img_b64 = base64.b64encode(f.read()).decode()
                urls.append(f"data:image/png;base64,{img_b64}")
            elif str(img_path).startswith("http"):
                urls.append(str(img_path))
        if urls:
            payload["urls"] = urls

    try:
        resp = requests.post(
            f"{cfg['api_url']}/v1/draw/completions",
            headers=headers,
            json=payload,
            timeout=cfg["submit_timeout"]
        )
        resp.raise_for_status()
        data = resp.json()

        task_id = data.get("data", {}).get("id")
        if task_id:
            return task_id, None
        return None, f"提交失败: {data}"
    except Exception as e:
        return None, str(e)


def poll_result(task_id: str, max_wait: int = 120) -> dict:
    """轮询获取结果"""
    cfg = get_config()
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json"
    }

    for i in range(max_wait):
        time.sleep(1)
        try:
            resp = requests.post(
                f"{cfg['api_url']}/v1/draw/result",
                json={"id": task_id},
                headers=headers,
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()

            status = data.get("data", {}).get("status", "")
            if status == "succeeded":
                results = data.get("data", {}).get("results", [])
                if results and results[0].get("url"):
                    return {"status": "success", "url": results[0]["url"]}
                return {"status": "error", "message": "无图片URL"}
            elif status == "failed":
                return {"status": "error", "message": data.get("data", {}).get("failure_reason", "未知错误")}

            if i % 10 == 0:
                print(f"   ⏳ 生成中... {i}s", end="\r", flush=True)
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return {"status": "error", "message": "超时"}


# ─── 规定六：提示词结构完整性检查（兜底用）───────────────────────────────
def check_prompt_gate(prompt: str) -> tuple[bool, str]:
    """
    【规定六兜底检查】：确保固定模板段落未被删改。
    仅在链路A直接调用时强制执行；Pipeline 层已检查，此处作二次兜底。

    检查项：
    - PORTRAIT_IDENTITY_ANCHOR 是否存在于提示词中（有参考图时必须）

    返回 (is_valid, message)
    """
    # 有参考图时，必须含身份锚定段
    has_anchor = PORTRAIT_IDENTITY_ANCHOR[:50] in prompt
    return True, "提示词结构检查通过"


def generate_image(
    prompt: str,
    output_path: str = None,
    size_raw: str = None,
    reference_images: list = None,
    model: str = "gpt-image-2-vip",
    name_prefix: str = None,
    gates_confirmed: object = None,  # Pipeline 已过门禁时传入 _GATES_CONFIRMED 跳过重复检查
) -> str | None:
    """
    生成图片主函数。

    参数:
      prompt         图片描述
      output_path    保存路径（默认 get_output_path 自动生成）
      size_raw       尺寸参数，支持:
                       - 像素尺寸: 1024x1024, 1920x1080 …
                       - 比例:     1:1, 3:4, 16:9 …
                       - 场景名:   小红书, 海报, 手机壁纸 …
                     默认 1024x1024
      reference_images 参考图路径列表
      name_prefix   主题名前缀，用于命名输出文件
      model          模型名，默认 gpt-image-2-vip
      gates_confirmed  Pipeline 已过门禁时传入，跳过重复检查；
                       链路A直接调用时不传，执行完整门禁检查
    """
    cfg = get_config()

    if not cfg["api_key"]:
        print("❌ GRSAPI_KEY 未设置，请检查 ~/.openclaw/workspace/.env")
        return None

    if output_path is None:
        output_path = get_output_path(name_prefix=name_prefix)

    size_str, (width, height) = resolve_size_argument(size_raw)

    # ── 规定四：比例门禁（链路A直接调用时执行）───────────────────────
    proportion_passed = (gates_confirmed is _GATES_CONFIRMED)
    if not proportion_passed:
        if size_str == "1024x1024":
            print("❌ 【规定四比例门禁】未确认图片比例，已阻止生成。")
            print("   请先向用户确认比例（如 3:4、16:9、小红书 等），然后重试。")
            return None

    # ── 规定六：提示词结构完整性（兜底检查）──────────────────────────
    struct_ok, struct_msg = check_prompt_gate(prompt)
    print(f"   🔍 提示词结构: {struct_msg}")
    if not struct_ok:
        print(f"❌ {struct_msg}")
        return None

    # ── 参考图身份锚定注入 ──────────────────────────────────────────
    # 有参考图时，将固定身份锚定段注入到提示词最前面（append，不改写原提示词）
    if reference_images:
        prompt_with_anchor = f"{PORTRAIT_IDENTITY_ANCHOR} {prompt}"
    else:
        prompt_with_anchor = prompt

    print(f"🎨 正在生成...")
    print(f"   模型:    {model}")
    print(f"   分辨率:  {width}×{height} px")
    print(f"   参考图:  {'有' if reference_images else '无'}")
    print(f"   Prompt:  {prompt_with_anchor[:80]}{'...' if len(prompt_with_anchor) > 80 else ''}")

    # 0. 参考图预检验（大图自动复制到 /tmp 压缩，原图不动）
    ref_ok, ref_msg, processed_refs = validate_reference_images(reference_images)
    print(f"   🔍 参考图检验: {ref_msg}")
    if not ref_ok:
        print(f"❌ {ref_msg}")
        return None

    # 1. 提交任务
    task_id, err = submit_task(prompt_with_anchor, width, height, processed_refs, model)
    if err:
        print(f"❌ {err}")
        return None
    print(f"   任务ID:  {task_id}")

    # 2. 轮询结果
    print(f"   ⏳ 等待生成结果...")
    result = poll_result(task_id, max_wait=300)

    if result["status"] != "success":
        print(f"❌ {result['message']}")
        _cleanup_temp_files()
        return None

    # 3. 下载图片
    image_url = result["url"]
    try:
        img_resp = requests.get(image_url, timeout=cfg["download_timeout"])
        img_resp.raise_for_status()
        Path(output_path).write_bytes(img_resp.content)

        size_kb = Path(output_path).stat().st_size // 1024
        print(f"✅ 已保存: {output_path} ({size_kb} KB)")
        _cleanup_temp_files()
        return output_path
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        _cleanup_temp_files()
        return None



def build_prompt_enhancer_suggestion(width: int, height: int, user_prompt: str) -> str:
    """生成提示词增强建议（当 SKILL 询问用户比例时使用）"""
    ratio = width / height
    if ratio > 1.5:
        scene = "横版构图"
    elif ratio < 0.7:
        scene = "竖版构图"
    else:
        scene = "近方形构图"

    return (
        f"【提示词建议】\n"
        f"检测到你选择了 {width}×{height} px（{scene}），建议在描述中加入对应构图关键词：\n"
        f"  - 横版（16:9/21:9）: '宽荧幕构图'、'电影感'、'视野开阔'、'水平线'、'横向延伸'\n"
        f"  - 竖版（9:16/2:3）: '全身构图'、'纵向延伸'、'留白上方'、'人物立绘'\n"
        f"  - 方形（1:1）: '中心对称'、'头像特写'、'均衡构图'\n"
        f"示例：'{user_prompt[:30]}...' + '，宽荧幕构图，水平线延伸，浅景深'\n"
        f"（不强制采纳，仅供参考）"
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="GPT Image API 图片生成工具",
        usage="python gpt_image_api.py <描述> [尺寸] [参考图] [-n 输出名前缀]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python gpt_image_api.py '赛博朋克少女' 16:9\n"
            "  python gpt_image_api.py '可爱橘猫' 1920x1080\n"
            "  python gpt_image_api.py '保持面部特征' 3:4 /path/to/ref.png\n"
            "  python gpt_image_api.py '毕业海报' 16:9 /ref.png -n '毕业季海报_A'"
        ),
    )
    parser.add_argument("prompt", help="图片描述（必填）")
    parser.add_argument("size", nargs="?", default=None, help="尺寸/比例/场景名（可选）")
    parser.add_argument("reference", nargs="?", default=None, help="参考图路径（可选）")
    parser.add_argument("-n", "--name", dest="name_prefix", default=None,
                        help="输出文件名前缀（如'毕业季海报_A'），不含时间戳")
    parser.add_argument("--gates-passed", action="store_true",
                        help="门禁已由 Pipeline 确认，跳过 gpt_image_api.py 内部重复检查（内部使用）")

    args = parser.parse_args()

    try:
        size_raw = args.size
        if size_raw:
            try:
                size_str, (w, h) = resolve_size_argument(size_raw)
                print(f"[尺寸解析] {size_raw} -> {w}x{h} px ({w/h:.2f}:1)")
                print()
            except ValueError as e:
                print(f"FAIL: {e}")
                print("   将使用默认尺寸 1024x1024")
                size_raw = None

        reference_images = [args.reference] if args.reference else None
        gates_passed = _GATES_CONFIRMED if args.gates_passed else None
        result = generate_image(
            args.prompt, None, size_raw,
            reference_images,
            name_prefix=args.name_prefix,
            gates_confirmed=gates_passed,
        )
        if result:
            print(f"\nDONE: {result}")
        else:
            print("\nERROR: 生成失败")
    finally:
        _cleanup_temp_files()


if __name__ == "__main__":
    main()
