# UI Sketch-to-Code Converter

**Course**: COM71123-3  
**Project**: UI Sketch-to-Code Converter  
**Phase**: Week 1 - Environment Setup and Core Dependencies

## Project Overview
The **UI Sketch-to-Code Converter** is an AI-driven tool designed to transform hand-drawn or digital UI wireframe sketches into clean, valid HTML/CSS code. It combines computer vision (OpenCV) for sketch segmentation, local vision-language models (LLaVA-v1.6-7B via Ollama) for code generation, and HTML sanitization/validation (DOMPurify & html-validate).

## Core Stack & Dependencies

- **Python**: 3.12+
  - `opencv-python`: Image pre-processing and contour extraction
  - `numpy`: Numerical operations on image matrices
  - `pillow`: Image conversion and handling
  - `requests`: Client for local Ollama REST API
- **Node.js**: v24+
  - `html-validate`: Validation of generated HTML structures
  - `dompurify` & `jsdom`: XSS sanitization of generated markup
- **AI Model**:
  - `LLaVA-v1.6-7B` running locally on **Ollama**

## Environment Setup

### 1. Python Environment Setup
```bash
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Install dependencies:
pip install -r requirements.txt
```

### 2. Node.js Environment Setup
```bash
npm install
```

### 3. Local Model Deployment (Ollama)
```bash
ollama pull llava:7b
ollama run llava:7b
```

### 4. Verification
Run the verification suite:
```bash
python tests/verify_env.py
```
