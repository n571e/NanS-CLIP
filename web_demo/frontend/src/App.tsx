import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import "./styles.css";

type DemoCase = {
  query: string;
  query_text: string;
  focus: string;
  explain: string;
  image_id?: number | null;
  filename?: string | null;
  ground_truth_image_ids?: number[];
};

type SummaryPayload = {
  hero: {
    title: string;
    subtitle: string;
    description: string;
  };
  benchmark_corpus: {
    label: string;
    image_count: number;
    text_count: number;
    mapped_image_count: number;
    note: string;
  };
  sources: Array<{ name: string; count: number }>;
  preset_queries: string[];
  demo_cases: DemoCase[];
  models: Array<{ key: string; label: string; ready: boolean }>;
};

type ItemPayload = {
  filename: string;
  title: string;
  source: string;
  original_url: string;
  description: string;
  representative_texts: {
    modern_chinese: string;
    ancient_style: string;
    keywords: string;
  };
};

type SearchResult = {
  image_id: number;
  filename?: string | null;
  rank: number;
  score: number;
  title: string;
  image_url: string;
  rank_change_vs_other: number;
  preview_text?: string;
  judgement: string;
  judgement_reason: string;
};

type TextMatchResult = {
  candidate_id: string;
  filename?: string | null;
  image_id?: number | null;
  rank: number;
  score: number;
  title: string;
  text: string;
  text_type_label: string;
  rank_change_vs_other: number;
  judgement: string;
  judgement_reason: string;
};

type ComparePayload = {
  query: string;
  query_text: string;
  top_k: number;
  pool_size: number;
  demo_case?: DemoCase | null;
  ground_truth: {
    image_ids: number[];
    count: number;
    zero_shot_hit_count: number;
    zero_shot_first_hit_rank: number | null;
    dora_hit_count: number;
    dora_first_hit_rank: number | null;
  };
  timings_ms: {
    zero_shot: number;
    dora: number;
  };
  results: {
    zero_shot: SearchResult[];
    dora: SearchResult[];
  };
};

type ImageToTextPayload = {
  top_k: number;
  pool_size: number;
  query_source: {
    type: string;
    image_id?: number | null;
    filename?: string | null;
    image_url?: string | null;
    label: string;
  };
  ground_truth: {
    candidate_ids: string[];
    count: number;
    zero_shot_hit_count: number;
    zero_shot_first_hit_rank: number | null;
    dora_hit_count: number;
    dora_first_hit_rank: number | null;
  };
  timings_ms: {
    zero_shot: number;
    dora: number;
  };
  results: {
    zero_shot: TextMatchResult[];
    dora: TextMatchResult[];
  };
};

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

function formatRankChange(value: number) {
  if (value > 0) {
    return `↑${value}`;
  }
  if (value < 0) {
    return `↓${Math.abs(value)}`;
  }
  return "—";
}

function verdictTone(judgement: string) {
  if (judgement === "标准答案") {
    return "hit";
  }
  if (judgement === "非标准答案") {
    return "off";
  }
  return "pending";
}

function formatHitSummary(hitCount: number, firstHitRank: number | null, topK: number) {
  if (hitCount > 0 && firstHitRank) {
    return `Top-${topK} 命中 ${hitCount} · 首个标准答案 #${firstHitRank}`;
  }
  return `Top-${topK} 未命中标准答案`;
}

function formatTiming(value: number) {
  return `${value.toFixed(1)} ms`;
}

function ResultCard({
  item,
  onOpenItem,
}: {
  item: SearchResult;
  onOpenItem: (filename?: string | null) => Promise<void>;
}) {
  return (
    <article className="result-card">
      <div className="result-thumb">
        <img src={item.image_url} alt={item.title} loading="lazy" />
      </div>
      <div className="result-meta">
        <div className="result-topline">
          <span className="rank-tag">#{item.rank}</span>
          <span className={`judgement-badge ${verdictTone(item.judgement)}`}>{item.judgement}</span>
        </div>
        <h4>{item.title}</h4>
        {item.preview_text ? <p className="result-preview">{item.preview_text}</p> : null}
        <p className="judgement-reason">{item.judgement_reason}</p>
        <div className="result-footer">
          <span className="score-text">相似度 {item.score.toFixed(3)}</span>
          <span className={`rank-delta ${item.rank_change_vs_other > 0 ? "up" : item.rank_change_vs_other < 0 ? "down" : ""}`}>
            {formatRankChange(item.rank_change_vs_other)}
          </span>
          {item.filename ? (
            <button type="button" className="text-link" onClick={() => void onOpenItem(item.filename)}>
              查看溯源
            </button>
          ) : null}
        </div>
      </div>
    </article>
  );
}

function TextResultCard({
  item,
  onOpenItem,
}: {
  item: TextMatchResult;
  onOpenItem: (filename?: string | null) => Promise<void>;
}) {
  return (
    <article className="text-result-card">
      <div className="result-topline">
        <span className="rank-tag">#{item.rank}</span>
        <span className={`judgement-badge ${verdictTone(item.judgement)}`}>{item.judgement}</span>
      </div>
      <span className="text-type-tag">{item.text_type_label}</span>
      <h4>{item.text}</h4>
      <p>{item.judgement_reason}</p>
      <div className="text-result-footer">
        <span className="score-text">相似度 {item.score.toFixed(3)}</span>
        <span className={`rank-delta ${item.rank_change_vs_other > 0 ? "up" : item.rank_change_vs_other < 0 ? "down" : ""}`}>
          {formatRankChange(item.rank_change_vs_other)}
        </span>
        {item.filename ? (
          <button type="button" className="text-link" onClick={() => void onOpenItem(item.filename)}>
            查看出处
          </button>
        ) : null}
      </div>
    </article>
  );
}

export default function App() {
  const [summary, setSummary] = useState<SummaryPayload | null>(null);
  const [activeDemoCase, setActiveDemoCase] = useState<DemoCase | null>(null);
  const [featuredItem, setFeaturedItem] = useState<ItemPayload | null>(null);
  const [featuredItemLoading, setFeaturedItemLoading] = useState(false);
  const [compare, setCompare] = useState<ComparePayload | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [showImageToText, setShowImageToText] = useState(false);
  const [imageCompare, setImageCompare] = useState<ImageToTextPayload | null>(null);
  const [imageCompareLoading, setImageCompareLoading] = useState(false);
  const [pageError, setPageError] = useState<string | null>(null);
  const [modalItem, setModalItem] = useState<ItemPayload | null>(null);
  const [modalLoading, setModalLoading] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const modalRequestRef = useRef(0);

  useEffect(() => {
    let cancelled = false;

    async function loadPage() {
      try {
        const summaryData = await fetchJson<SummaryPayload>("/api/summary");
        const initialCase = summaryData.demo_cases[0] ?? null;

        const [initialItem, initialCompare] = await Promise.all([
          initialCase?.filename ? fetchJson<ItemPayload>(`/api/item/${initialCase.filename}`) : Promise.resolve(null),
          initialCase
            ? fetchJson<ComparePayload>("/api/search/compare", {
                method: "POST",
                headers: {
                  "Content-Type": "application/json",
                },
                body: JSON.stringify({ query: initialCase.query, top_k: 5 }),
              })
            : Promise.resolve(null),
        ]);

        if (!cancelled) {
          setSummary(summaryData);
          setActiveDemoCase(initialCase);
          setFeaturedItem(initialItem);
          setCompare(initialCompare);
        }
      } catch (error) {
        if (!cancelled) {
          setPageError(error instanceof Error ? error.message : "unknown_error");
        }
      }
    }

    void loadPage();

    return () => {
      cancelled = true;
    };
  }, []);

  async function activateDemoCase(demoCase: DemoCase) {
    setActiveDemoCase(demoCase);
    setCompareLoading(true);
    setShowImageToText(false);
    setImageCompare(null);
    try {
      const [itemPayload, comparePayload] = await Promise.all([
        demoCase.filename ? fetchJson<ItemPayload>(`/api/item/${demoCase.filename}`) : Promise.resolve(null),
        fetchJson<ComparePayload>("/api/search/compare", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ query: demoCase.query, top_k: 5 }),
        }),
      ]);
      setFeaturedItem(itemPayload);
      setCompare(comparePayload);
    } finally {
      setCompareLoading(false);
    }
  }

  async function runImageToTextSearch(imageId?: number | null) {
    if (!imageId) {
      return;
    }
    setImageCompareLoading(true);
    try {
      const payload = await fetchJson<ImageToTextPayload>("/api/search/image-to-text", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image_id: imageId, top_k: 5 }),
      });
      setImageCompare(payload);
    } finally {
      setImageCompareLoading(false);
    }
  }

  async function openItem(filename?: string | null) {
    if (!filename) {
      return;
    }
    const requestId = modalRequestRef.current + 1;
    modalRequestRef.current = requestId;
    setIsModalOpen(true);
    setModalItem(null);
    setModalLoading(true);
    try {
      const payload = await fetchJson<ItemPayload>(`/api/item/${filename}`);
      if (modalRequestRef.current === requestId) {
        setModalItem(payload);
      }
    } finally {
      if (modalRequestRef.current === requestId) {
        setModalLoading(false);
      }
    }
  }

  function closeModal() {
    modalRequestRef.current += 1;
    setIsModalOpen(false);
    setModalItem(null);
    setModalLoading(false);
  }

  if (pageError) {
    return (
      <main className="page-shell error-shell">
        <section className="error-panel">
          <h1>页面初始化失败</h1>
          <p>请先确认后端服务已经启动。</p>
          <p className="error-detail">{pageError}</p>
        </section>
      </main>
    );
  }

  if (!summary) {
    return (
      <main className="page-shell loading-shell">
        <p className="loading-copy">正在整理南宋图文证据链…</p>
      </main>
    );
  }

  const displayedCase = compare?.demo_case ?? activeDemoCase;
  const featuredImageId = displayedCase?.image_id ?? imageCompare?.query_source.image_id ?? null;
  const modalContent =
    isModalOpen || modalLoading
      ? createPortal(
          <div className="modal-backdrop" role="dialog" aria-modal="true" onClick={closeModal}>
            <div className="modal-card" onClick={(event) => event.stopPropagation()}>
              <button type="button" className="modal-close" onClick={closeModal}>
                关闭
              </button>
              {modalLoading || !modalItem ? (
                <p>正在加载样本溯源…</p>
              ) : (
                <>
                  <p className="sample-kicker">样本溯源</p>
                  <h3>{modalItem.title}</h3>
                  <dl className="detail-list">
                    <div>
                      <dt>来源站点</dt>
                      <dd>{modalItem.source}</dd>
                    </div>
                    <div>
                      <dt>原始链接</dt>
                      <dd>
                        <a href={modalItem.original_url} target="_blank" rel="noreferrer">
                          {modalItem.original_url}
                        </a>
                      </dd>
                    </div>
                    <div>
                      <dt>描述</dt>
                      <dd>{modalItem.description}</dd>
                    </div>
                    <div>
                      <dt>现代中文</dt>
                      <dd>{modalItem.representative_texts.modern_chinese}</dd>
                    </div>
                    <div>
                      <dt>古风文本</dt>
                      <dd>{modalItem.representative_texts.ancient_style}</dd>
                    </div>
                    <div>
                      <dt>关键词</dt>
                      <dd>{modalItem.representative_texts.keywords}</dd>
                    </div>
                  </dl>
                </>
              )}
            </div>
          </div>,
          document.body,
        )
      : null;

  return (
    <main className="page-shell">
      <div className="page-texture" />

      <section className="hero-panel hero-panel--compact">
        <div className="hero-copy">
          <p className="hero-kicker">NanS-CLIP Benchmark Demo</p>
          <h1>{summary.hero.title}</h1>
          <p className="hero-subtitle">{summary.hero.subtitle}</p>
          <p className="hero-description">{summary.hero.description}</p>
          <div className="hero-note">
            <span className="hero-note-label">当前展示</span>
            <strong>标准答案对比</strong>
            <p>{summary.benchmark_corpus.note}</p>
          </div>
        </div>

        <div className="hero-stats hero-stats--compact">
          <article className="hero-stat">
            <span>验证图像</span>
            <strong>{summary.benchmark_corpus.image_count}</strong>
          </article>
          <article className="hero-stat">
            <span>验证文本</span>
            <strong>{summary.benchmark_corpus.text_count}</strong>
          </article>
          <article className="hero-stat hero-stat--accent">
            <span>当前展示</span>
            <strong>标准答案对比</strong>
            <small>{summary.benchmark_corpus.label}</small>
          </article>
        </div>
      </section>

      <section className="content-card featured-case-card">
        <div className="section-heading section-heading--split">
          <div>
            <p className="sample-kicker">当前展示</p>
            <h2>当前展示案例</h2>
            <p>{displayedCase?.explain ?? summary.benchmark_corpus.note}</p>
          </div>
          <p className="section-aside">
            先看真实样本与来源，再看同一条验证样本在 Zero-Shot 与 DoRA 下的标准答案排序差异。
          </p>
        </div>

        {featuredItemLoading && !featuredItem ? (
          <div className="search-placeholder">
            <p>正在加载当前展示案例…</p>
          </div>
        ) : featuredItem ? (
          <div className="featured-case-grid">
            <article className="featured-case-copy">
              <span className="case-pill">Benchmark Sample</span>
              <h3>{displayedCase?.query ?? featuredItem.title}</h3>
              {displayedCase ? <p className="case-focus">{displayedCase.focus}</p> : null}
              {displayedCase ? (
                <div className="case-query-block">
                  <span>标准查询文本</span>
                  <p>{displayedCase.query_text}</p>
                </div>
              ) : null}
              <p className="featured-description">{featuredItem.description}</p>
              <div className="sample-actions">
                {displayedCase ? (
                  <button type="button" onClick={() => void activateDemoCase(displayedCase)}>
                    重新运行标准答案对比
                  </button>
                ) : null}
                <button type="button" className="secondary" onClick={() => void openItem(featuredItem.filename)}>
                  展开完整溯源档案
                </button>
              </div>
            </article>

            <article className="featured-case-proof">
              <p className="proof-label">样本溯源档案</p>
              <dl className="proof-list">
                <div>
                  <dt>来源站点</dt>
                  <dd>{featuredItem.source || "未知"}</dd>
                </div>
                <div>
                  <dt>原始链接</dt>
                  <dd>
                    <a href={featuredItem.original_url} target="_blank" rel="noreferrer">
                      {featuredItem.original_url}
                    </a>
                  </dd>
                </div>
                <div>
                  <dt>现代中文</dt>
                  <dd>{featuredItem.representative_texts.modern_chinese}</dd>
                </div>
                <div>
                  <dt>古风文本</dt>
                  <dd>{featuredItem.representative_texts.ancient_style}</dd>
                </div>
                <div>
                  <dt>关键词</dt>
                  <dd>{featuredItem.representative_texts.keywords}</dd>
                </div>
              </dl>
            </article>
          </div>
        ) : (
          <div className="search-placeholder">
            <p>该验证样本暂未恢复到原始公开来源，下面仍会按验证集标准答案展示检索结果。</p>
            {displayedCase ? <p>标准查询文本：{displayedCase.query_text}</p> : null}
          </div>
        )}

        <div className="case-switcher">
          <div className="case-switcher-head">
            <div>
              <p className="sample-kicker">案例切换</p>
              <h3>切换验证集案例</h3>
            </div>
            <p>切换不同验证样本，查看对应来源档案与标准答案排序结果。</p>
          </div>

          <div className="demo-case-row demo-case-row--compact">
            {summary.demo_cases.map((demoCase) => (
              <button
                type="button"
                key={demoCase.query}
                className={`demo-case-chip ${displayedCase?.query === demoCase.query ? "active" : ""}`}
                onClick={() => void activateDemoCase(demoCase)}
              >
                <strong>{demoCase.query}</strong>
                <span>{demoCase.focus}</span>
              </button>
            ))}
          </div>

          <div className="source-strip">
            {summary.sources.map((source) => (
              <article className="source-tile" key={source.name}>
                <span>{source.name === "Wikimedia Commons" ? "Wikimedia Commons 来源" : source.name}</span>
                <strong>{source.count}</strong>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="content-card search-card search-card--editorial">
        <div className="section-heading section-heading--split">
          <div>
            <p className="sample-kicker">标准答案对比</p>
            <h2>标准答案检索对比</h2>
            <p>同一条验证样本，直接比较 Zero-Shot 与 DoRA 在标准答案上的排序差异。</p>
          </div>
          <div className="secondary-entry">
            <button type="button" className="secondary-toggle" onClick={() => setShowImageToText(true)}>
              查看图搜文能力
            </button>
          </div>
        </div>

        {displayedCase ? (
          <section className="compare-context compare-context--featured">
            <div>
              <p className="sample-kicker">当前标准查询文本</p>
              <h3>
                {displayedCase.query} · {displayedCase.focus}
              </h3>
            </div>
            <div className="compare-context-copy">
              <p className="compare-query-text">{displayedCase.query_text}</p>
              <p>{displayedCase.explain}</p>
            </div>
          </section>
        ) : null}

        {compareLoading && !compare ? (
          <div className="search-placeholder">
            <p>正在比较 Zero-Shot 与 DoRA 的标准答案排位…</p>
          </div>
        ) : compare ? (
          <div className="compare-grid">
            <section className="result-column">
              <div className="column-head">
                <div>
                  <h3>Zero-Shot</h3>
                  <p className="column-summary">
                    {formatHitSummary(compare.ground_truth.zero_shot_hit_count, compare.ground_truth.zero_shot_first_hit_rank, compare.top_k)}
                  </p>
                </div>
                <span>{formatTiming(compare.timings_ms.zero_shot)}</span>
              </div>
              {compare.results.zero_shot.map((item) => (
                <ResultCard key={`zero-${item.image_id}`} item={item} onOpenItem={openItem} />
              ))}
            </section>

            <section className="result-column emphasized">
              <div className="column-head">
                <div>
                  <h3>DoRA</h3>
                  <p className="column-summary">
                    {formatHitSummary(compare.ground_truth.dora_hit_count, compare.ground_truth.dora_first_hit_rank, compare.top_k)}
                  </p>
                </div>
                <span>{formatTiming(compare.timings_ms.dora)}</span>
              </div>
              {compare.results.dora.map((item) => (
                <ResultCard key={`dora-${item.image_id}`} item={item} onOpenItem={openItem} />
              ))}
            </section>
          </div>
        ) : (
          <div className="search-placeholder">
            <p>先点一个验证集案例，再看标准答案是否被排进 Top-K。</p>
          </div>
        )}

        {showImageToText ? (
          <section className="secondary-flow">
            <div className="secondary-flow-head">
              <div>
                <p className="sample-kicker">次级能力</p>
                <h3>图搜文结果</h3>
              </div>
              <p>这里仍然只使用验证集标准样本，用来补充展示 NanS-CLIP 的反向跨模态检索能力。</p>
            </div>

            <div className="image-tool-row">
              {featuredImageId ? (
                <button
                  type="button"
                  onClick={() => void runImageToTextSearch(featuredImageId)}
                  disabled={imageCompareLoading}
                >
                  {imageCompareLoading ? "匹配中…" : "使用当前标准样本图"}
                </button>
              ) : null}
            </div>

            {imageCompare ? (
              <>
                <section className="image-query-preview">
                  <p className="sample-kicker">当前查询图像</p>
                  <div className="image-query-card">
                    <div className="image-query-thumb">
                      <img src={imageCompare.query_source.image_url ?? ""} alt={imageCompare.query_source.label} />
                    </div>
                    <div>
                      <h4>{imageCompare.query_source.label}</h4>
                      <p>下方结果同样按验证集标准文本自动判定，重点看标准答案是否被排进前列。</p>
                    </div>
                  </div>
                </section>

                <div className="compare-grid compare-grid--secondary">
                  <section className="result-column">
                    <div className="column-head">
                      <div>
                        <h3>Zero-Shot</h3>
                        <p className="column-summary">
                          {formatHitSummary(
                            imageCompare.ground_truth.zero_shot_hit_count,
                            imageCompare.ground_truth.zero_shot_first_hit_rank,
                            imageCompare.top_k,
                          )}
                        </p>
                      </div>
                      <span>{formatTiming(imageCompare.timings_ms.zero_shot)}</span>
                    </div>
                    {imageCompare.results.zero_shot.map((item) => (
                      <TextResultCard key={`zero-${item.candidate_id}`} item={item} onOpenItem={openItem} />
                    ))}
                  </section>
                  <section className="result-column emphasized">
                    <div className="column-head">
                      <div>
                        <h3>DoRA</h3>
                        <p className="column-summary">
                          {formatHitSummary(
                            imageCompare.ground_truth.dora_hit_count,
                            imageCompare.ground_truth.dora_first_hit_rank,
                            imageCompare.top_k,
                          )}
                        </p>
                      </div>
                      <span>{formatTiming(imageCompare.timings_ms.dora)}</span>
                    </div>
                    {imageCompare.results.dora.map((item) => (
                      <TextResultCard key={`dora-${item.candidate_id}`} item={item} onOpenItem={openItem} />
                    ))}
                  </section>
                </div>
              </>
            ) : (
              <div className="search-placeholder search-placeholder--secondary">
                <p>点击“使用当前标准样本图”后，就能看到验证集图搜文的标准答案排序。</p>
              </div>
            )}
          </section>
        ) : null}
      </section>

      {modalContent}
    </main>
  );
}
