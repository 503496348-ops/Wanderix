#!/usr/bin/env python3
"""
人像提示词生成器
核心原则：以参考图人物一致性为主，风格描述为辅

用法:
  python3 portrait_prompt.py "痞帅男性肖像，电影感侧光"
  python3 portrait_prompt.py "毕业照海报" --aspect 2:3
  python3 portrait_prompt.py "常子涵开心肖像" --portrait "zihan"
"""
import sys
import argparse


# ─── 人物一致性锚定段（永远放在 Prompt 最前面）────────────────────────
PORTRAIT_IDENTITY_BLOCK = """THIS IS A PORTRAIT OF THE SAME PERSON FROM THE REFERENCE IMAGE.
You must preserve the person's exact identity features: their unique face shape, eye shape, nose, lips, chin, jawline, skin tone, and hair style. The reference image defines who this person is — their identity is PRIMARY and CANNOT be changed.
Keep the person looking like the SAME individual shown in the reference photo."""


# ─── 人像常用风格词库 ────────────────────────────────────────────────
PORTRAIT_STYLE_TERMS = {
    "cinematic": "cinematic film grain, dramatic chiaroscuro lighting, shallow depth of field, moody atmosphere, anamorphic lens flare",
    "痞帅": "handsome rebellious cool guy, slightly smirk expression, confident cool demeanor, light stubble, sharp jawline, deep mysterious eyes, film noir aesthetic",
    "scholarly": "scholarly gentle appearance, warm kind eyes, refined intellectual aura, soft lighting, bookish atmosphere",
    "graduation": "graduation ceremony atmosphere, academic regalia elements subtly integrated, proud confident expression, warm golden hour lighting",
    "fantasy": "fantasy character portrait, epic cinematic lighting, ethereal glow, legendary hero aesthetic, dramatic backlighting",
    "watercolor": "dreamlike watercolor illustration style, soft wet-on-wet technique, flowing pigments, paper texture, poetic atmosphere",
    "neon": "cyberpunk neon lighting, blue and magenta rim light, rain-slicked reflections, dark moody background",
    "natural": "natural soft window light, clean background, authentic skin texture with pores visible, minimal makeup, candid realistic portrait",
    "dramatic": "dramatic Rembrandt lighting, hard shadows on one side of face, cinematic high contrast, studio portrait",
}


def build_portrait_prompt(user_description: str, style: str = None, aspect: str = None) -> str:
    """
    生成人像专用提示词。

    策略：先放参考图人物身份锚定，再放风格指令，最后接用户描述。
    身份锚定永远不会被风格描述覆盖。

    参数:
      user_description: 用户的场景/风格描述
      style: 预设风格（可选，见 PORTRAIT_STYLE_TERMS）
      aspect: 构图比例（可选，提示构图）
    """
    parts = []

    # 1. 身份锚定（第一句，不可省略）
    parts.append(PORTRAIT_IDENTITY_BLOCK)

    # 2. 构图控制（若有）
    if aspect:
        aspect_notes = {
            "2:3": "portrait orientation, close-up on face and upper body, upper torso visible",
            "3:4": "portrait orientation, head-and-shoulders or half-body shot",
            "1:1": "square composition, centered face portrait",
            "4:5": "portrait orientation, elegant full face with neck and shoulders",
            "16:9": "wide cinematic portrait, face dominant with atmospheric background",
        }
        if aspect in aspect_notes:
            parts.append(f"Composition: {aspect_notes[aspect]}.")

    # 3. 风格词（若有预设风格）
    if style and style in PORTRAIT_STYLE_TERMS:
        parts.append(f"Style: {PORTRAIT_STYLE_TERMS[style]}.")

    # 4. 用户描述（保持原样，不翻译不变写）
    if user_description.strip():
        parts.append(f"Subject description: {user_description.strip()}.")

    # 5. 质量保证（最后防线）
    parts.append("High quality, photorealistic, sharp focus on facial features, natural skin texture.")

    return "\n".join(parts)


def print_prompt(prompt: str):
    """带分隔线打印提示词"""
    print(f"\n{'='*60}")
    print(f"📝 生成的人像提示词")
    print(f"{'='*60}")
    print(f"\n{prompt}\n")
    print(f"{'='*60}")
    # 标注关键结构
    print(f"\n🔑 结构解析:")
    lines = prompt.split('\n')
    for i, line in enumerate(lines, 1):
        tag = "【身份锚定】" if i == 1 else ("【构图】" if 'Composition' in line else
               "【风格】" if 'Style' in line else
               "【用户描述】" if 'Subject' in line else "【质量】")
        print(f"  {tag} {line[:70]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="人像提示词生成器")
    parser.add_argument("description", help="人像描述（保持原样传入）")
    parser.add_argument("--style", "-s", default=None, help=f"预设风格，可选: {', '.join(PORTRAIT_STYLE_TERMS.keys())}")
    parser.add_argument("--aspect", "-a", default=None, help="构图比例: 2:3, 3:4, 1:1, 4:5, 16:9")
    parser.add_argument("--quiet", "-q", action="store_true", help="仅输出提示词，不打印解析")
    args = parser.parse_args()

    prompt = build_portrait_prompt(args.description, style=args.style, aspect=args.aspect)

    if args.quiet:
        print(prompt)
    else:
        print_prompt(prompt)
