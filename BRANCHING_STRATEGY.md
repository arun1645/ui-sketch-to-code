# Branching Strategy & Workflow Guidelines

## Overview
This document outlines the Git branching strategy and repository workflow for the **UI Sketch-to-Code Converter** project (COM71123-3).

## Branch Architecture

```
main (Production / Stable Releases)
  ▲
  │ (Pull Request via code review)
  │
develop (Integration Branch)
  ▲
  ├───────────────────────┬───────────────────────┐
  │                       │                       │
feature/env-setup     feature/cv-pipeline    feature/llm-codegen
```

### 1. `main` Branch
- **Purpose**: Production-ready, stable code releases.
- **Rules**:
  - Direct commits to `main` are restricted.
  - Changes must be merged from `develop` via approved Pull Requests (PRs).
  - Every commit on `main` must pass all automated verification tests.

### 2. `develop` Branch
- **Purpose**: Primary integration branch for active development.
- **Rules**:
  - Contains the latest merged features for the upcoming milestone.
  - Must remain buildable and pass test suites at all times.

### 3. Feature Branches (`feature/<feature-name>`)
- **Purpose**: Isolated development of specific features, bug fixes, or experimental code.
- **Naming Conventions**:
  - `feature/sketch-parser` - OpenCV component extraction
  - `feature/llava-integration` - Local Ollama model prompting & generation
  - `feature/html-validator` - Code sanitization & validation with DOMPurify / html-validate
  - `fix/<bug-description>` - Bug fixes

## Workflow & PR Process

1. **Branch Creation**:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/<feature-name>
   ```

2. **Commit Standards**:
   - Use imperative, descriptive commit messages:
     - `feat: add OpenCV contour detection for UI buttons`
     - `fix: resolve DOMPurify script tag strip issue`
     - `docs: update setup and environment requirements`

3. **Pull Request Checklist**:
   - [ ] Feature branch is up to date with `develop`.
   - [ ] All automated unit tests (`tests/verify_env.py`, test suites) pass locally.
   - [ ] Code is formatted and docstrings/comments are present.
   - [ ] At least one group member code review approval.
