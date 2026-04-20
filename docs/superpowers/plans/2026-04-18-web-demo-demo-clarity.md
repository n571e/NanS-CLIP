# Web Demo Benchmark Clarity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reframe the NanS-CLIP web demo into a benchmark-backed presentation where every showcased retrieval result comes from the validation split and is judged against standard answers instead of live open-ended guesses.

**Architecture:** Keep the existing Flask + React structure, but switch retrieval data loading from the live cleaned corpus to the validation benchmark assets (`valid_texts.jsonl` + LMDB images). The backend will recover `image_id -> filename` mappings where possible so benchmark hits can still open provenance details, while the frontend will present curated benchmark cases, standard-answer verdicts, and a secondary image-to-text benchmark flow.

**Tech Stack:** Python, Flask, React, TypeScript, Vite, Vitest, unittest, LMDB

---

### Task 1: Benchmark Data Plumbing

**Files:**
- Create: `web_demo/backend/benchmark_data.py`
- Modify: `web_demo/backend/tests/test_demo_backend.py`
- Test: `web_demo/backend/tests/test_demo_backend.py`

- [ ] **Step 1: Write failing unit tests for validation-text loading and image-id mapping helpers**
- [ ] **Step 2: Run `D:\anaconda3\envs\pytorch\python.exe -m unittest web_demo.backend.tests.test_demo_backend -v` and verify the new tests fail for missing benchmark helpers**
- [ ] **Step 3: Implement benchmark helpers for ordered validation text loading, ground-truth maps, and cached `image_id -> filename` recovery from `valid_imgs.tsv`**
- [ ] **Step 4: Re-run `D:\anaconda3\envs\pytorch\python.exe -m unittest web_demo.backend.tests.test_demo_backend -v` and verify the new helper tests pass**

### Task 2: Backend Search And API Shift To Validation Benchmark

**Files:**
- Modify: `web_demo/backend/demo_backend.py`
- Modify: `web_demo/backend/app.py`
- Modify: `web_demo/backend/tests/test_api.py`
- Test: `web_demo/backend/tests/test_api.py`

- [ ] **Step 1: Write failing API tests for benchmark corpus summary fields, standard-answer verdicts, and benchmark image serving**
- [ ] **Step 2: Run `D:\anaconda3\envs\pytorch\python.exe -m unittest web_demo.backend.tests.test_api -v` and verify the new benchmark tests fail**
- [ ] **Step 3: Replace live-corpus search data with validation benchmark data, annotate search results against benchmark ground truth, and add a benchmark-image route**
- [ ] **Step 4: Re-run `D:\anaconda3\envs\pytorch\python.exe -m unittest web_demo.backend.tests.test_api -v` and verify the API tests pass**

### Task 3: Frontend Narrative Rewrite For Standard-Answer Retrieval

**Files:**
- Modify: `web_demo/frontend/src/App.tsx`
- Modify: `web_demo/frontend/src/styles.css`
- Modify: `web_demo/frontend/src/__tests__/App.test.tsx`
- Test: `web_demo/frontend/src/__tests__/App.test.tsx`

- [ ] **Step 1: Write failing frontend tests for benchmark wording, curated case chips, and `标准答案 / 非标准答案` result badges**
- [ ] **Step 2: Run `npm test -- --run` in `web_demo/frontend` and verify the new assertions fail**
- [ ] **Step 3: Update the UI to remove live free-text search, foreground benchmark case selection, show the actual validation query text, and use benchmark image routes**
- [ ] **Step 4: Keep image-to-text as a secondary validation-set flow driven by the selected benchmark sample instead of uploads**
- [ ] **Step 5: Re-run `npm test -- --run` in `web_demo/frontend` and verify the frontend tests pass**

### Task 4: Docs And Verification

**Files:**
- Modify: `web_demo/README.md`
- Test: `web_demo/backend/tests`
- Test: `web_demo/frontend`

- [ ] **Step 1: Update the README to describe the benchmark-backed demo and the provenance-when-available behavior**
- [ ] **Step 2: Run `D:\anaconda3\envs\pytorch\python.exe -m unittest discover -s web_demo/backend/tests -v`**
- [ ] **Step 3: Run `npm test -- --run` in `web_demo/frontend`**
- [ ] **Step 4: Run `npm run build` in `web_demo/frontend`**
- [ ] **Step 5: Run a real-data smoke check for one curated benchmark case and verify the compare response reports benchmark-grounded verdicts**

---

**Spec coverage check:** The plan covers the approved direction: validation-set-backed retrieval, standard-answer verdicts instead of manual judgement, preservation of provenance where a benchmark image maps back to an original filename, and a benchmark-only text-to-image plus optional image-to-text presentation.
