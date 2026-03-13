# iwa-collections-analysis

Exploratory bioinformatics analysis platform for reusable and practical workflows.

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
git clone https://github.com/yourname/iwa-collections-analysis.git
cd iwa-collections-analysis

# Install Pixi if it is not already installed: or Install Pixi by following the official instructions for your environment, then run:
# Exp:
curl -fsSL https://pixi.sh/install.sh | sh
# Install project dependencies:
pixi install
```

## Usage

Run the application inside the Pixi environment:
```
pixi shell
python -m streamlit run iwa_collections_analysis.py
```
Alternatively, you can launch it directly without entering the shell:
```
pixi run python -m streamlit run iwa_collections_analysis.py
```