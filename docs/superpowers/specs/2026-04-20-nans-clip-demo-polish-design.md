# NanS-CLIP Web Demo Polish Design

## Goal

Reframe the current `web_demo` frontend into a teacher-facing presentation page that feels credible, calm, and well-designed, while preserving the benchmark-backed retrieval logic that is already in place.

This redesign is not a product-platform expansion. It is a presentation-layer refinement for a 3 to 5 minute research demo. The page should help a teacher quickly understand:

1. What NanS-CLIP does
2. Why the demo data is real and traceable
3. How `DoRA` improves benchmark retrieval over `Zero-Shot`
4. That image-to-text retrieval still exists as a secondary capability

## Current Context

The existing frontend already has the right functional building blocks:

- benchmark-backed text-to-image comparison
- benchmark-backed image-to-text comparison
- curated demo cases
- provenance lookup for mapped samples
- standard-answer verdicts instead of manual judgement

What is currently weak is the presentation rhythm. The page still feels like a mixed demo console rather than a deliberate research showcase. Important evidence exists, but it is not yet arranged in the order that best supports a teacher conversation.

## Approved Direction

The approved direction is `B. 证据先行型`.

This means the redesigned page should prioritize trust-building before feature breadth:

- keep a concise hero
- surface the current selected benchmark case immediately after the hero
- make provenance a first-class section instead of a side detail
- keep text-to-image retrieval as the main narrative
- keep image-to-text as a secondary entry
- remove any separate metrics-focused showcase block from the main page

## Design References

The visual direction is informed by [DESIGN-claude.md](/D:/Desktop/KnowledgeBase/多模态/CLIP/NanS-CLIP/DESIGN-claude.md), but should be interpreted rather than copied.

The intended visual character is:

- warm and editorial, not futuristic
- refined and trustworthy, not decorative or theatrical
- soft paper-toned surfaces with restrained contrast
- serif headlines plus cleaner sans-serif utility text
- terracotta used as selective emphasis, not everywhere
- card depth driven by subtle borders and ring shadows instead of heavy gradients

The design should feel closer to a research exhibit page than to a dashboard admin panel.

## Information Architecture

The main page should be reduced to three primary blocks.

### 1. Project Summary Hero

Purpose:

- establish what the project is
- state that the demo is benchmark-backed
- show only the minimum statistics needed to orient the viewer

Content:

- project title
- short subtitle describing NanS-CLIP as a benchmark-backed cross-modal retrieval demo for Southern Song cultural imagery
- one supporting sentence explaining that the current page compares `Zero-Shot` and `DoRA` against standard answers
- three compact statistics:
  - validation images
  - validation texts
  - demo mode or benchmark-backed label

Constraints:

- do not let the hero dominate the page
- do not use oversized decorative backgrounds that compete with the content
- do not place provenance details here

### 2. Featured Case And Provenance

Purpose:

- make the selected benchmark case the narrative anchor of the page
- prove that the demo corresponds to real data rather than a shell UI

Layout:

- two-column desktop layout
- stacked mobile layout

Left column:

- current case label
- case title or query keyword
- standard query text
- short explanation of why this case is worth showing
- primary action to rerun the benchmark comparison for this case
- secondary action to open the full provenance modal if extra details are available

Right column:

- provenance archive card
- source site
- `original_url`
- representative modern Chinese text
- representative ancient-style text
- keywords

Below the featured area:

- a compact row of curated case chips
- only preselected benchmark cases that are stable and explainable in conversation

Constraints:

- provenance must be visible on the page by default
- provenance should not depend on opening a modal
- free-text search should not appear in this block

### 3. Standard-Answer Retrieval Comparison

Purpose:

- show the actual model difference in a way that a teacher can understand quickly
- make correctness more important than raw similarity score

Primary mode:

- text-to-image comparison is the default and primary mode

Secondary mode:

- image-to-text comparison remains available through a lighter-weight entry inside this section

Text-to-image requirements:

- two equal columns: `Zero-Shot` and `DoRA`
- both columns must render the same number of results
- each result card must show:
  - rank
  - standard-answer verdict
  - ranking change versus the other model
  - brief judgement reason
  - similarity score as a secondary detail
- the column header must show a compact hit summary such as Top-K hits and first-hit rank

Visual priority:

- `标准答案` badges and hit summaries are primary signals
- similarity values are secondary
- `DoRA` may receive mild visual emphasis only when the selected case demonstrates a better benchmark outcome

Image-to-text behavior:

- kept as a secondary capability
- entered through an explicit toggle or reveal action
- should not visually compete with the default text-to-image narrative
- still uses benchmark-backed standard-answer verdicts

## Interaction Rules

The page should behave as a case-driven presentation, not as an open exploration console.

Rules:

- on initial load, automatically select one curated benchmark case
- when the selected case changes, the page must refresh all related content together:
  - featured case copy
  - provenance details
  - text-to-image comparison results
  - image-to-text context reset if necessary
- no free-form search box on the main page
- no upload-first interaction model
- curated case chips should be few, high-signal, and chosen specifically because they tell a better `DoRA` story

This keeps the page deterministic and easier to present during a conversation.

## Visual System

### Typography

- use serif typography for the hero title and section titles
- use sans-serif for chips, badges, buttons, metadata, and dense explanatory text
- keep large headings calm and editorial rather than oversized and dramatic

### Color

- primary page background: warm paper tone
- elevated surfaces: ivory or soft off-white
- main text: warm near-black
- secondary text: warm olive-gray
- accent color: restrained terracotta
- avoid cool gray or blue-heavy UI surfaces

### Surfaces And Depth

- replace heavy decorative gradients with simpler paper-toned layering
- use thin warm borders and gentle ring shadows
- keep cards rounded and soft
- use whitespace to create hierarchy before using stronger color

### Tone

The page should read as:

- serious enough for research discussion
- warm enough to feel human and authored
- restrained enough to keep trust high

It should not read as:

- a gamified retrieval toy
- a museum-themed landing page
- a generic dashboard template

## Content Strategy

The copy should support spoken explanation.

This means:

- fewer slogans
- more precise descriptive language
- each section should be understandable when read aloud during a demo
- case descriptions should help the presenter explain why a keyword matters

The page copy should not overclaim. It should show that `DoRA` improves selected benchmark cases, not suggest universal dominance in every scenario.

## Out Of Scope

This redesign does not include:

- changing training logic
- changing benchmark evaluation logic
- adding live open-ended search back into the main workflow
- adding image upload retrieval
- adding category browsing
- rebuilding backend APIs unless a small payload change is needed to support the new layout

## Implementation Notes

The work should mainly target the current frontend:

- [web_demo/frontend/src/App.tsx](/D:/Desktop/KnowledgeBase/多模态/CLIP/NanS-CLIP/web_demo/frontend/src/App.tsx)
- [web_demo/frontend/src/styles.css](/D:/Desktop/KnowledgeBase/多模态/CLIP/NanS-CLIP/web_demo/frontend/src/styles.css)
- [web_demo/frontend/src/__tests__/App.test.tsx](/D:/Desktop/KnowledgeBase/多模态/CLIP/NanS-CLIP/web_demo/frontend/src/__tests__/App.test.tsx)

Expected structural changes:

- simplify the hero markup
- move featured provenance into the central narrative position
- tighten the curated case selector
- rebalance the comparison section so verdicts and rank movement dominate
- visually demote image-to-text into a secondary flow

Backend changes are not part of the primary redesign scope. Only add or reshape payload fields if the new presentation cannot be completed with the current benchmark-backed data and the added fields remain narrowly scoped to case copy or highlight metadata.

## Acceptance Criteria

The redesign is successful if the following are true:

1. A teacher can open the page and understand the demo purpose within the first screen.
2. The selected sample's provenance is visible without opening a modal.
3. The selected case clearly shows benchmark-backed `Zero-Shot` versus `DoRA` differences.
4. Correctness is easier to see than similarity score.
5. The page looks intentional and polished without becoming visually noisy.
6. Image-to-text remains available, but no longer distracts from the main narrative.
7. The resulting UI still works on desktop and mobile layouts.

## Planning Readiness Check

This spec is intentionally scoped to one implementation track:

- frontend narrative restructure
- frontend visual redesign
- minor supporting data adjustments only if needed

It does not require a parallel redesign of the backend benchmark logic, so it is narrow enough for a single implementation plan.
