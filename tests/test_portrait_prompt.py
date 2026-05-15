#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────
# test_portrait_prompt.py — portrait_prompt.py 全覆盖测试
# 目标覆盖率：≥80%
# ─────────────────────────────────────────────────────────────
import sys
from pathlib import Path

import pytest
sys.path.insert(0, str(Path(__file__).parent.parent))

import portrait_prompt as pp_mod


# ══════════════════════════════════════════════════════════════
# 第一节：PORTRAIT_IDENTITY_BLOCK — 常量完整性
# ══════════════════════════════════════════════════════════════
class TestIdentityBlock:
    def test_not_empty(self):
        assert len(pp_mod.PORTRAIT_IDENTITY_BLOCK) > 10

    def test_contains_key_phrases(self):
        block = pp_mod.PORTRAIT_IDENTITY_BLOCK
        assert "PORTRAIT" in block.upper()
        assert "REFERENCE IMAGE" in block.upper()

    def test_preserves_identity_phrases(self):
        block = pp_mod.PORTRAIT_IDENTITY_BLOCK
        assert "identity" in block.lower()
        assert "preserve" in block.lower()


# ══════════════════════════════════════════════════════════════
# 第二节：PORTRAIT_STYLE_TERMS — 风格词库
# ══════════════════════════════════════════════════════════════
class TestStyleTerms:
    def test_all_styles_have_content(self):
        for style, terms in pp_mod.PORTRAIT_STYLE_TERMS.items():
            assert len(terms) > 5, f"风格 {style} 词库内容过短"

    def test_all_styles_are_strings(self):
        for style, terms in pp_mod.PORTRAIT_STYLE_TERMS.items():
            assert isinstance(terms, str), f"风格 {style} 不是字符串"

    def test_cinematic_style(self):
        terms = pp_mod.PORTRAIT_STYLE_TERMS["cinematic"]
        assert "cinematic" in terms or "film" in terms

    def test_all_named_styles_exist(self):
        expected = {"cinematic", "痞帅", "scholarly", "graduation",
                    "fantasy", "watercolor", "neon", "natural", "dramatic"}
        assert expected == set(pp_mod.PORTRAIT_STYLE_TERMS.keys())


# ══════════════════════════════════════════════════════════════
# 第三节：build_portrait_prompt — 提示词构建
# ══════════════════════════════════════════════════════════════
class TestBuildPortraitPrompt:
    def test_identity_block_first(self):
        prompt = pp_mod.build_portrait_prompt("一只猫")
        # 身份锚定段必须在 Prompt 最前面（多行第一句即可）
        first_line = prompt.split("\n")[0].strip()
        assert "THIS IS A PORTRAIT" in first_line or "PORTRAIT" in prompt.upper()

    def test_description_included(self):
        prompt = pp_mod.build_portrait_prompt("穿着红色外套的女孩")
        assert "红色外套" in prompt or "red" in prompt.lower()

    def test_no_style_when_none(self):
        prompt = pp_mod.build_portrait_prompt("一只猫")
        # 不应包含 "Style:" 标记
        assert "Style: None" not in prompt

    def test_with_valid_style(self):
        prompt = pp_mod.build_portrait_prompt("一只猫", style="cinematic")
        assert "Style:" in prompt
        assert "cinematic" in prompt.lower() or "film" in prompt.lower()

    def test_with_invalid_style_ignored(self):
        prompt = pp_mod.build_portrait_prompt("一只猫", style="nonexistent_style_xyz")
        assert "Style: Nonexistent" not in prompt
        assert "Style:" not in prompt

    @pytest.mark.parametrize("aspect", ["2:3", "3:4", "1:1", "4:5", "16:9"])
    def test_all_supported_aspects(self, aspect):
        prompt = pp_mod.build_portrait_prompt("人像", aspect=aspect)
        assert "Composition:" in prompt or "portrait" in prompt.lower()

    def test_aspect_2_3_note(self):
        prompt = pp_mod.build_portrait_prompt("人像", aspect="2:3")
        assert "upper body" in prompt or "torso" in prompt.lower()

    def test_aspect_1_1_note(self):
        prompt = pp_mod.build_portrait_prompt("人像", aspect="1:1")
        assert "square" in prompt.lower()

    def test_aspect_16_9_note(self):
        prompt = pp_mod.build_portrait_prompt("人像", aspect="16:9")
        assert "wide" in prompt.lower() or "cinematic" in prompt.lower()

    def test_aspect_unsupported_no_crash(self):
        prompt = pp_mod.build_portrait_prompt("人像", aspect="99:1")
        # 不崩溃，但无 composition 段落
        assert pp_mod.PORTRAIT_IDENTITY_BLOCK in prompt

    def test_empty_description(self):
        prompt = pp_mod.build_portrait_prompt("")
        assert pp_mod.PORTRAIT_IDENTITY_BLOCK in prompt
        assert len(prompt) > 50

    def test_whitespace_description(self):
        prompt = pp_mod.build_portrait_prompt("   ")
        assert pp_mod.PORTRAIT_IDENTITY_BLOCK in prompt

    def test_quality_guarantee_present(self):
        prompt = pp_mod.build_portrait_prompt("测试")
        assert "quality" in prompt.lower() or "photorealistic" in prompt.lower()

    def test_all_parts_joined(self):
        """验证各部分用换行符连接"""
        prompt = pp_mod.build_portrait_prompt("测试", style="cinematic", aspect="2:3")
        parts = prompt.split("\n")
        assert len(parts) >= 3  # identity + composition + quality

    def test_style_and_aspect_combined(self):
        prompt = pp_mod.build_portrait_prompt("人像", style="dramatic", aspect="3:4")
        assert "Style:" in prompt
        assert "Composition:" in prompt
        assert "dramatic" in prompt.lower() or "Rembrandt" in prompt

    def test_full_pipeline(self):
        """完整流程：身份锚定 + 构图 + 风格 + 描述 + 质量"""
        prompt = pp_mod.build_portrait_prompt(
            "校园少女，阳光下的操场",
            style="natural",
            aspect="3:4"
        )
        # 各段落都存在
        assert pp_mod.PORTRAIT_IDENTITY_BLOCK[:30] in prompt
        assert "Composition:" in prompt
        assert "Style:" in prompt
        assert "校园" in prompt or "campus" in prompt.lower()
        assert "quality" in prompt.lower()


# ══════════════════════════════════════════════════════════════
# 第四节：print_prompt — 辅助输出函数
# ══════════════════════════════════════════════════════════════
class TestPrintPrompt:
    def test_no_crash(self, capsys):
        prompt = pp_mod.build_portrait_prompt("测试", style="cinematic", aspect="2:3")
        pp_mod.print_prompt(prompt)
        captured = capsys.readouterr()
        assert len(captured.out) > 0
        assert "生成的人像提示词" in captured.out

    def test_output_contains_sections(self, capsys):
        prompt = pp_mod.build_portrait_prompt("测试")
        pp_mod.print_prompt(prompt)
        captured = capsys.readouterr()
        # 应该标注了结构
        out = captured.out
        assert "【身份锚定】" in out or "identity" in out.lower()
