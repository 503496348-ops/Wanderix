#!/usr/bin/env python3
"""
艺游未境-Wanderix · GPT Image 2 图片生成 Skill

一念画成，图像即现。
集成 GPT Image 2 (gpt-image-2-vip) 模型，支持：
  - 自定义分辨率 / 比例快捷词 / 中文场景名
  - 参考图身份锚定（大图自动压缩）
  - PLOC 手账涂鸦 Pipeline 自动化
  - 常子涵肖像定制模板

目录结构：
  艺游未境-Wanderix/
  ├── SKILL.md               # 技能说明书（含完整 PRD）
  ├── gpt_image_api.py       # API 封装核心
  ├── portrait_prompt.py     # 人像提示词模板
  ├── check_reference.py      # 参考图预检
  ├── gen_image.sh           # Shell 一键入口
  ├── pipeline/              # Pipeline 自动化工作流
  │   ├── engine.py
  │   └── templates/
  │       ├── registry.json
  │       ├── default/
  │       └── user/
  ├── tests/                 # pytest 测试套件（≥80% 覆盖率）
  ├── docs/                  # 技能手册
  ├── .env.example          # 环境变量模板
  └── README.md              # 部署指南
"""
__version__ = "1.0.0"
__author__ = "小乖（AI私人助理）"
