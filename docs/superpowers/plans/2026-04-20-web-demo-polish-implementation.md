# Web Demo Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the NanS-CLIP web demo frontend into the approved evidence-first presentation while keeping the existing benchmark-backed retrieval behavior.

**Architecture:** Keep the current React single-page frontend and existing backend API contracts, but reorganize the page around a compact hero, a first-class featured case plus provenance block, and a cleaner benchmark comparison section. Implement the narrative shift entirely in the frontend unless a tiny payload gap appears, and preserve image-to-text as a secondary flow instead of a competing first-class panel.

**Tech Stack:** React, TypeScript, Vite, Vitest, CSS

---

### Task 1: Lock The New Narrative In Tests

**Files:**
- Modify: `web_demo/frontend/src/__tests__/App.test.tsx`
- Test: `web_demo/frontend/src/__tests__/App.test.tsx`

- [ ] **Step 1: Write failing frontend assertions for the evidence-first layout**

```tsx
it("renders the compact hero and evidence-first featured case layout", async () => {
  render(<App />);

  await screen.findByRole("heading", { name: "NanS-CLIP" });
  expect(screen.getByRole("heading", { name: "当前精选案例" })).toBeInTheDocument();
  expect(screen.getByRole("heading", { name: "标准答案检索对比" })).toBeInTheDocument();
  expect(screen.getByText("当前展示")).toBeInTheDocument();
  expect(screen.getByText("原始链接")).toBeInTheDocument();
});
```

- [ ] **Step 2: Run the focused test file and verify it fails**

Run: `npm test -- --run src/__tests__/App.test.tsx`
Expected: FAIL because the new headings and layout wording do not exist yet.

- [ ] **Step 3: Add failing assertions for visual demotion of image-to-text and stronger verdict-first copy**

```tsx
it("keeps image-to-text behind a secondary entry and foregrounds verdict summaries", async () => {
  render(<App />);

  await screen.findByText("Top-5 未命中标准答案");
  expect(screen.getByRole("button", { name: "查看图搜文能力" })).toBeInTheDocument();
  expect(screen.queryByRole("heading", { name: "图搜文（可选）" })).not.toBeInTheDocument();
});
```

- [ ] **Step 4: Re-run the focused test file and verify the new assertions fail for the intended reasons**

Run: `npm test -- --run src/__tests__/App.test.tsx`
Expected: FAIL with missing text / heading assertions, not test syntax errors.

### Task 2: Rebuild App Structure Around The Featured Case

**Files:**
- Modify: `web_demo/frontend/src/App.tsx`
- Test: `web_demo/frontend/src/__tests__/App.test.tsx`

- [ ] **Step 1: Refactor the top-level JSX so the hero becomes compact and the featured case becomes the central section**

```tsx
<section className="hero-panel hero-panel--compact">
  <div className="hero-copy">
    <span className="hero-kicker">NanS-CLIP Benchmark Demo</span>
    <h1>{summary.hero.title}</h1>
    <p className="hero-subtitle">{summary.hero.subtitle}</p>
    <p className="hero-description">{summary.hero.description}</p>
  </div>
  <div className="hero-stats hero-stats--compact">
    ...
  </div>
</section>

<section className="content-card featured-case-card">
  <div className="section-heading section-heading--split">
    <div>
      <p className="sample-kicker">当前展示</p>
      <h2>当前精选案例</h2>
      <p>{displayedCase?.explain ?? summary.benchmark_corpus.note}</p>
    </div>
  </div>
  ...
</section>
```

- [ ] **Step 2: Move provenance details into the default visible layout and keep the modal as a secondary detail view**

```tsx
<div className="featured-case-grid">
  <div className="featured-case-copy">...</div>
  <dl className="featured-case-proof">
    <div>
      <dt>来源站点</dt>
      <dd>{featuredItem.source || "未知"}</dd>
    </div>
    <div>
      <dt>原始链接</dt>
      <dd><a href={featuredItem.original_url}>{featuredItem.original_url}</a></dd>
    </div>
    ...
  </dl>
</div>
```

- [ ] **Step 3: Tighten the comparison section language so standard-answer summaries lead and image-to-text becomes a reveal flow**

```tsx
<section className="content-card search-card search-card--full">
  <div className="section-heading">
    <h2>标准答案检索对比</h2>
    <p>同一条验证集样本，直接比较 Zero-Shot 与 DoRA 在标准答案上的排序差异。</p>
  </div>

  <div className="secondary-entry">
    <button type="button" className="secondary-toggle" onClick={...}>
      查看图搜文能力
    </button>
  </div>
</section>
```

- [ ] **Step 4: Run the focused frontend test file and verify the updated structure satisfies the new assertions**

Run: `npm test -- --run src/__tests__/App.test.tsx`
Expected: PASS for the new layout-focused assertions or reveal the next missing copy mismatch.

### Task 3: Replace The Old Ornamental Styling With The Approved Editorial System

**Files:**
- Modify: `web_demo/frontend/src/styles.css`
- Test: `web_demo/frontend/src/__tests__/App.test.tsx`

- [ ] **Step 1: Rewrite root page variables and shells to a calmer paper-toned editorial palette**

```css
:root {
  font-family: "Georgia", "Noto Serif SC", "Songti SC", serif;
  color: #141413;
  background: #f5f4ed;
}

body {
  background:
    radial-gradient(circle at top left, rgba(201, 100, 66, 0.08), transparent 28%),
    linear-gradient(180deg, #f5f4ed 0%, #f2efe6 100%);
}
```

- [ ] **Step 2: Add dedicated layout styles for the compact hero, featured case grid, curated chips, and benchmark comparison columns**

```css
.hero-panel--compact { ... }
.featured-case-card { ... }
.featured-case-grid { ... }
.featured-case-proof { ... }
.demo-case-row--compact { ... }
.search-card--editorial { ... }
```

- [ ] **Step 3: Restyle result cards so verdict badges and hit summaries visually outrank raw similarity**

```css
.column-summary {
  color: #c96442;
  font-weight: 600;
}

.judgement-badge.hit {
  background: rgba(201, 100, 66, 0.14);
  color: #c96442;
}
```

- [ ] **Step 4: Run the focused test file again and verify the structural behavior still passes after the CSS rewrite**

Run: `npm test -- --run src/__tests__/App.test.tsx`
Expected: PASS

### Task 4: Full Frontend Verification And Demo Readiness

**Files:**
- Modify: `web_demo/README.md`
- Test: `web_demo/frontend/src/__tests__/App.test.tsx`
- Test: `web_demo/frontend`

- [ ] **Step 1: Update the README wording so it describes the polished evidence-first presentation**

```md
The current web demo is a benchmark-backed presentation page. It opens on a curated validation case, keeps provenance visible by default, and presents `Zero-Shot` versus `DoRA` as a standard-answer comparison instead of an open live search interface.
```

- [ ] **Step 2: Run the full frontend test suite**

Run: `npm test -- --run`
Expected: PASS

- [ ] **Step 3: Run the frontend production build**

Run: `npm run build`
Expected: PASS with Vite production bundle output

- [ ] **Step 4: Perform a quick manual smoke check against the running backend**

Run: `npm run dev`
Expected: the page loads with the compact hero, visible provenance, curated case chips, benchmark comparison columns, and a secondary image-to-text entry.

---

**Spec coverage check:**
- compact hero: covered in Task 2 and Task 3
- featured case plus visible provenance: covered in Task 2
- verdict-first comparison layout: covered in Task 2 and Task 3
- secondary image-to-text flow: covered in Task 2
- polished editorial visual direction: covered in Task 3
- verification and demo-readiness wording: covered in Task 4
