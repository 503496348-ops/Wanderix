# 艺游未境-Wanderix · 部署与配置指南

> 一念画成，图像即现。
> 念头一动，图像即成。

---

## 概述

艺游未境-Wanderix 调用 GPT Image 2（`gpt-image-2-vip`）生成图片，支持：
- 自定义分辨率 / 比例快捷词 / 中文场景名
- 参考图身份锚定（大图自动压缩）
- PLOC 手账涂鸦 Pipeline 自动化
- 常子涵肖像定制模板

**零配置即可使用** —— 首次运行时脚本会引导用户完成 API Key 配置。

---

## 版本信息

| 项目 | 值 |
|------|-----|
| 版本 | v1.0.0 |
| 更新日期 | 2026-05-16 |
| 作者 | 小乖（AI私人助理） |
| 覆盖测试 | ≥80%（pytest，三测通过） |

---

## 部署流程（3步完成）

### 第一步：获取 API Key

访问 **https://yh.grsai.ai** 注册并获取 API Key。

> 💡 积分价格约 0.03 元/张，首次使用建议小额充值测试。

### 第二步：安装依赖

```bash
# 确认已安装 uv（Python 包管理器）
which uv || brew install uv

# 安装必要库
uv pip install requests Pillow
```

### 第三步：配置并测试

```bash
# 进入 skill 目录
cd 艺游未境-Wanderix/

# 首次运行，按提示输入 API Key（会自动创建 .env 文件）
chmod +x gen_image.sh
./gen_image.sh "一只可爱的橘猫"
```

> ⚠️ 首次运行若提示"未配置 API Key"，脚本会自动引导你完成配置。

---

## 配置说明

### 方式一：.env 文件（推荐）

```bash
# 在 skill 目录下创建 .env
cp .env.example .env
# 编辑 .env，填入真实 Key
nano .env
```

### 方式二：系统环境变量

```bash
# 写入 ~/.zshrc（永久生效）
echo 'export GRSAPI_KEY=your-key-here' >> ~/.zshrc
source ~/.zshrc
```

---

## 快速使用

### 基本生图

```bash
./gen_image.sh "一只橘猫"
./gen_image.sh "赛博朋克少女" "16:9"
```

### Python API

```python
from gpt_image_api import generate_image

# 基础生图
generate_image("一只橘猫", size_raw="3:4")

# 带参考图（保持角色一致性）
generate_image(
    "财阀千金风肖像",
    size_raw="2:3",
    reference_images=["/path/to/ref2.jpg"]
)

# 带输出名前缀
generate_image(
    "新疆之旅",
    size_raw="3:4",
    name_prefix="毕业季海报"
)
```

### Pipeline 自动化

```bash
# PLOC 手账涂鸦
python pipeline/engine.py run PLOC_治愈手账涂鸦 --input /path/to/photo.jpg

# 列出所有可用模板
python pipeline/engine.py list
```

---

## 权限引导话术（智能体专用）

当用户首次触发本 Skill 时，若检测到未配置 API Key，请使用以下话术引导：

> **用户**：帮我生成图片：一只橘猫
>
> **智能体**：好的！首次使用需要简单配置一下 API Key 😊
>
> 请访问 **https://yh.grsai.ai** 注册，获取 API Key（大约 1 分钟）。
>
> 获取后把 Key 发给我，我来帮你完成配置～

---
>
> **用户**：sk-xxxxxxxxxxxxx
>
> **智能体**：收到！正在配置...
>
> ✅ 配置完成！现在开始生成图片～

---

## 常见问题

**Q: 提示"insufficient credits"怎么办？**
A: 积分不足，请到 https://yh.grsai.ai 充值。

**Q: 生成需要多长时间？**
A: 通常 15-30 秒，受网络和队列影响。

**Q: 支持哪些尺寸？**
A: 任意宽×高均可，如 `1024x1365`（3:4竖版）、`1920x1080`（横版）等，或使用快捷词`小红书`、`海报`等。

**Q: 提示"Permission denied"？**
A: 运行`chmod +x gen_image.sh`赋予执行权限。

**Q: 大图会自动压缩吗？**
A: 是的，>8MB 的参考图自动复制到 `/tmp` 压缩，原图不动。

---

## 文件清单

```
艺游未境-Wanderix/
├── SKILL.md                   # 技能说明书（含完整 PRD）
├── gpt_image_api.py          # API 封装核心
├── portrait_prompt.py         # 人像提示词模板
├── check_reference.py        # 参考图预检
├── pipeline/                 # Pipeline 自动化工作流
│   ├── engine.py              # 引擎
│   └── templates/
│       ├── registry.json      # 模板注册表
│       ├── default/           # 内置模板
│       └── user/              # 用户自定义模板
├── tests/                     # pytest 测试套件（≥80% 覆盖率）
├── docs/                      # 技能手册
├── gen_image.sh              # Shell 一键入口
├── .env.example              # 环境变量模板
└── README.md                 # 本文件
```

---

## 测试运行

```bash
# 进入目录
cd 艺游未境-Wanderix/

# 运行全部测试（三测通过后再上传）
uv run pytest tests/ -v --cov=. --cov-report=term-missing

# 查看覆盖率报告
open coverage_report.html  # 或终端查看
```

> **覆盖率目标**：≥80%（测试三测通过后再提交仓库）

---

## 更新日志

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0.0 | 2026-05-16 | 艺游未境-Wanderix 初始版本，含完整 PRD + pytest ≥80% |

---

_本 Skill 由「艺游未境-Wanderix」项目提供 · GPT Image 2 模型驱动_
