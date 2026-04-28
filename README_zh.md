<div align="center">

# 中国传统命理评测基准 (Chinese Fortune Telling Bench)

**面向大语言模型的中国传统命理评测基准 —— 涵盖八字与紫微斗数。**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Questions](https://img.shields.io/badge/Questions-160-green.svg)](./data/data.json)
[![Years](https://img.shields.io/badge/Years-2022--2025-orange.svg)](./data)

[English](./README.md) | 中文

</div>

---

## 简介

本基准全部采用选择题形式,以**与标准答案完全一致**作为评分标准。题目来源于 **全球算命师大赛**([hkjfma.org](https://hkjfma.org))2022–2025 年度赛题,原始数据位于 [data/raw/](./data/raw/)。

| 文件 | 说明 |
|---|---|
| [data/data.json](./data/data.json) | 对 160 道选择题进行整理,划分入事业、健康、婚姻、子女、财运等十二大类事件类型。 |
| [data/fortune_api_results.json](./data/fortune_api_results.json) | 借助 [iztro](https://github.com/SylarLong/iztro) 预先排定的八字与紫微斗数命盘,通过 `case_id` （命主索引） 与 `data.json` （题目）关联;当传入 `--astro` 时在，会在运行时注入命盘信息,用于将**排盘**与**推理**两个环节解耦。 |

> 使用 `--year` 按年份筛选题目;使用 `--stats` 查看数据集统计信息。

---

## 安装

```bash
git clone https://github.com/DestinyLinker/MingLi-Bench.git
cd MingLi-Bench
pip install -r requirements.txt
```

---

## 配置

命令行工具会从 `.env` 文件读取 API 密钥和默认参数。复制模板文件后,**只需填写你实际调用的服务商**即可 —— 空值或占位值会被自动忽略。

```bash
cp .env.example .env
```

<details>
<summary><b>支持的服务商</b></summary>

```dotenv
# OpenRouter —— 一个密钥即可调用大多数模型
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# 原生接口(仅当你直接调用对应服务时才需填写)
OPENAI_API_KEY=sk-...
# OPENAI_BASE_URL=https://api.openai.com/v1   # 如需对接 OpenAI 兼容网关,可在此覆盖
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com

# 豆包 / 火山引擎 —— 密钥与 endpoint id 均为必填
DOUBAO_API_KEY=...
DOUBAO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
DOUBAO_ENDPOINT_ID=ep-...

# 默认参数(可被命令行参数覆盖)
TIMEOUT=60
MAX_WORKERS=5
MAX_TOKENS=8192
TEMPERATURE=0.0
```

</details>

`.env` 已加入 `.gitignore`。如需切换另一套密钥,可通过 `--env-file /path/to/other.env` 指定。

key 配置完成后,可先做一次**连通性自检**:

```bash
python -m mingli_bench.cli --list-models   # 查看受支持的模型列表
python -m mingli_bench.cli --stats         # 查看数据集统计信息
```

---

## 快速上手

> **推荐默认配置:** 建议始终开启 `--cot` 与 `--astro`。思维链(CoT)让模型有足够的空间对命盘进行逐步推演;`--astro` 会注入预先排好的八字 / 紫微斗数命盘,使得评测分数反映的是**推理能力**而非**由生辰排盘**的准确度。仅当你希望对这两项做消融实验时,才建议关闭它们。

常用的调用方式有两种。

### 1. 通过 OpenRouter(一个密钥,覆盖大多数模型)

无需指定 `--platform`,将模型名写成 OpenRouter 的 `provider/model` 形式即可。命令行会识别到 `/` 并自动路由至 OpenRouter。

```bash
python -m mingli_bench.cli --model openai/gpt-4o               --year 2025 --cot --astro --max-workers 8
python -m mingli_bench.cli --model anthropic/claude-sonnet-4-6 --year 2025 --cot --astro
python -m mingli_bench.cli --model google/gemini-2.5-pro       --year 2025 --cot --astro
python -m mingli_bench.cli --model deepseek/deepseek-r1        --year 2025 --cot --astro
```

### 2. 直接调用原生接口

当模型名无法被自动识别(例如带版本号的豆包 endpoint id),或你希望强制走某个 OpenAI 兼容网关时,可显式传入 `--platform`。

```bash
# 原生豆包 / 火山引擎接口
python -m mingli_bench.cli \
    --platform doubao --model doubao-seed-2-0-pro-260215 \
    --year 2025 --cot --astro --max-workers 8

# 任意模型名,通过 OpenAI 兼容网关调用(网关地址由 OPENAI_BASE_URL 指定)
python -m mingli_bench.cli \
    --platform openai --model doubao-seed-2-0-pro-260215 \
    --year 2025 --cot --astro --max-workers 8
```

`--platform` 可选值:`openai`、`openrouter`、`anthropic`、`google`、`deepseek`、`doubao`。

---

## 提示词示例

以 [data/data.json](./data/data.json) 中的 `ftb_0001` 为例,不加任何参数时:

```text
以下是一道关于中国传统命理的题目。

命主信息：
男命：1974年4月28日下午4:40分 出生地点：usa
结合中国传统命理学（包括但不限于四柱八字、紫微斗数等）进行推算，请直接给出答案，用'答案：X'的格式（X为A、B、C或D）。

问题：此命1996年发生何事？

选项：
A. 患上严重抑郁痴
B. 回港认识现任妻子
C. 交通意外，撞车，人平安
D. 得到一笔意外之财
```

加 `--cot`,上面的指令行替换为:

```text
结合中国传统命理学（包括但不限于四柱八字、紫微斗数等），请先分析推理过程，然后给出答案。最后用'答案：X'的格式给出你的选择（X为A、B、C或D）。
```

加 `--astro`,在指令与问题之间插入预先排好的命盘:

```text
八字命盘信息：
八字：甲寅 戊辰 己亥 壬申
时辰：申时
五行局：金四局
生肖：虎

紫微命盘信息：
十二宫位星曜分布：
命宫：天同 火星
兄弟：七杀 天马
夫妻：天梁 左辅 右弼 天钺 地劫
子女：廉贞 天相
财帛：巨门
疾厄：贪狼
迁移：太阴 地空 擎羊
仆役：紫微 天府 文昌 禄存
官禄：天机 天魁 陀罗
田宅：破军 文曲
福德：太阳 铃星
父母：武曲
```

---

## 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model, -m` | **必填** | 模型名。含 `/` 时视为 OpenRouter id;否则根据前缀推断服务商(`gpt-*`、`claude-*`、`gemini-*`、`deepseek-*`、`doubao-*`)。 |
| `--platform` | 自动推断 | 强制指定调用平台,覆盖基于前缀的推断。 |
| `--year, -y` | 全部 | 仅评测指定年份的题目,可选:`2022`、`2023`、`2024`、`2025`。 |
| `--max-workers` | `5` | API 并发请求数。速率限制允许时可提高到 8–16;触发限流则应调低。 |
| `--cot` | 关闭 | 在提示词前加入思维链(CoT)指令。 |
| `--astro` | 关闭 | 从 `data/fortune_api_results.json` 中注入预先排好的八字 / 紫微斗数命盘(模型无需自行根据生辰排盘)。 |
| `--sample, -s N` | 全部 | 仅评测前 N 道题目,适合做快速自测。 |
| `--categories, -c` | 全部 | 按类别筛选,例如 `--categories 事业 婚姻`。可选:事业、健康、外貌、婚姻、子女、学业、官非、家庭、性格、灾劫、财运、运势。 |
| `--shuffle-options` | 关闭 | 每题随机打乱选项顺序,避免位置偏差。 |
| `--output-dir, -o` | `logs` | 结果文件的输出目录。 |
| `--no-save` | 关闭 | 仅输出到终端,不写入文件。 |
| `--env-file` | `.env` | 指定其他 env 文件。 |
| `--list-models` | — | 列出受支持的模型并退出。 |
| `--stats` | — | 打印数据集统计信息并退出(可与 `--year` 联合使用)。 |

完整帮助文档:`python -m mingli_bench.cli --help`。

---

## 输出结果

每次运行都会在 `--output-dir`(默认 `logs/`)下生成三类文件:

| 文件 | 说明 |
|---|---|
| `<model>_results.json` | 每道题目的预测结果、打分与整体统计。 |
| `<model>_summary.txt`  | 核心指标摘要。 |
| `<model>_responses/`   | 模型原始响应,每题保存为一个独立文件。 |

---

## 许可证

本项目基于 [MIT License](./LICENSE) 开源。

## 联系方式

- Issues: [github.com/DestinyLinker/MingLi-Bench/issues](https://github.com/DestinyLinker/MingLi-Bench/issues)
- 邮箱: [help@destinylinker.com](mailto:help@destinylinker.com)
