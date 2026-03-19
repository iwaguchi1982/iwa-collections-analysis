# iwa-collections-analysis
再利用可能で実用的なワークフローを実現する、探索的バイオインフォマティクス解析プラットフォーム。
Exploratory bioinformatics analysis platform for reusable and practical workflows.
現在未実装：ローカルLLM（外部流出無し）で解釈や、確認ポイントを補助する予定。
Currently not implemented: Local LLM (no external leakage) is planned to assist with interpretation and verification points.
Used for target gene discovery.
標的遺伝子の探索に利用。
cBIoportal dataset: Your own data, etc.
cBIoportalデータセット:お手持ちのデータ等。

## Overview

`iwa-collections-analysis` is a source-available exploratory analysis platform designed to support practical bioinformatics workflows.  
This project aims to provide reusable scripts, small utilities, and workflow components for data exploration, preprocessing, and downstream analysis.

## Current Scope

This repository currently focuses on:

- exploratory analysis workflows
- reusable helper scripts and sub-apps
- practical execution in real research environments
- stepwise development toward a broader analysis platform

## Intended Users

This project is intended for:

- bioinformatics engineers
- researchers who want reproducible exploratory workflows
- users who prefer practical script-based analysis environments

## Environment

Tested environment:

- OS: Ubuntu 24.04 / WSL2 Ubuntu
- Python: 3.10+
- Package / environment manager: Pixi

## Installation

Clone this repository:

```bash
git clone https://github.com/iwaguchi1982/iwa-collections-analysis.git
cd iwa-collections-analysis

# Install Pixi if it is not already installed:
```bash
curl -fsSL https://pixi.sh/install.sh | sh
# Install project dependencies:
pixi install
```

## Usage

Run the application inside the Pixi environment:
```bash
pixi shell
python -m streamlit run iwa_collections_analysis.py
```
Alternatively, you can launch it directly without entering the shell:
```bash
pixi run python -m streamlit run iwa_collections_analysis.py
```
## Documentation

- [Overview](docs/overview.md)
- [Features](docs/features.md)
- [FAQ](docs/faq.md)
- [Deployment Guide (JA)](DEPLOYMENT_GUIDE_JA.md)

## Notes
> 本プロジェクトは探索研究向けの開発中ソフトウェアです。
> 臨床判断、規制対応用途、診断用途を意図したものではありません。

## LICENSE
This project is distributed under the Iwa Collections Non-Resale License 1.0.
