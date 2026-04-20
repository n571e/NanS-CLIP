import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "../App";

const summaryPayload = {
  hero: {
    title: "NanS-CLIP",
    subtitle: "南宋文博标准检索 Demo",
    description: "所有对比都基于 valid 验证集标准答案，而不是开放式 live 搜索。",
  },
  benchmark_corpus: {
    label: "验证集检索池",
    image_count: 1117,
    text_count: 4868,
    mapped_image_count: 1089,
    note: "本页所有命中与否均按 valid 验证集图文配对自动判定。",
  },
  sources: [
    { name: "Baidu Images", count: 1035 },
    { name: "Wikimedia Commons", count: 54 },
  ],
  preset_queries: ["德寿宫", "马远", "雷峰塔", "临安", "保俶塔"],
  demo_cases: [
    {
      query: "德寿宫",
      query_text: "南宋德寿宫遗址博物馆",
      focus: "南宋宫殿遗址",
      explain: "用验证集标准文本查询，展示 DoRA 是否把标准答案排得更靠前。",
      image_id: 42,
      filename: "deshou.png",
      ground_truth_image_ids: [42],
    },
    {
      query: "马远",
      query_text: "马远 踏歌图",
      focus: "南宋院体画家",
      explain: "用南宋画家相关验证文本，观察 DoRA 对领域画家语义的排序变化。",
      image_id: 16,
      filename: "mayuan.png",
      ground_truth_image_ids: [16],
    },
  ],
  models: [
    { key: "zero_shot", label: "Zero-Shot", ready: true },
    { key: "dora", label: "DoRA", ready: true },
  ],
};

const itemPayload = {
  filename: "deshou.png",
  title: "南宋德寿宫遗址",
  source: "Wikimedia Commons",
  original_url:
    "https://example.com/archive/deshou/source/very-long-link/with-query/that-should-wrap-correctly-in-proof-card/deshou.png?source=benchmark&view=full",
  description: "杭州德寿宫遗址博物馆公开图像",
  representative_texts: {
    modern_chinese: "南宋德寿宫遗址博物馆",
    ancient_style: "宫阙旧影，南宋遗踪",
    keywords: "德寿宫, 南宋, 宫殿遗址",
  },
};

const comparePayload = {
  query: "德寿宫",
  query_text: "南宋德寿宫遗址博物馆",
  top_k: 5,
  pool_size: 1117,
  demo_case: summaryPayload.demo_cases[0],
  ground_truth: {
    image_ids: [42],
    count: 1,
    zero_shot_hit_count: 0,
    zero_shot_first_hit_rank: null,
    dora_hit_count: 1,
    dora_first_hit_rank: 1,
  },
  timings_ms: { zero_shot: 12.5, dora: 13.2 },
  results: {
    zero_shot: [
      {
        image_id: 99,
        filename: null,
        rank: 1,
        score: 0.88,
        title: "北京太庙",
        image_url: "/api/eval-images/99",
        rank_change_vs_other: -3,
        preview_text: "明清皇家建筑",
        judgement: "非标准答案",
        judgement_reason: "该图片不在当前验证文本的标准答案集合中。",
      },
    ],
    dora: [
      {
        image_id: 42,
        filename: "deshou.png",
        rank: 1,
        score: 0.76,
        title: "南宋德寿宫遗址",
        image_url: "/api/eval-images/42",
        rank_change_vs_other: 3,
        preview_text: "南宋德寿宫遗址博物馆",
        judgement: "标准答案",
        judgement_reason: "该图片是当前验证文本的标准配对图像。",
      },
    ],
  },
};

const imageToTextPayload = {
  top_k: 5,
  pool_size: 4868,
  query_source: {
    type: "benchmark",
    image_id: 42,
    filename: "deshou.png",
    image_url: "/api/eval-images/42",
    label: "南宋德寿宫遗址",
  },
  ground_truth: {
    candidate_ids: ["text:0"],
    count: 1,
    zero_shot_hit_count: 0,
    zero_shot_first_hit_rank: null,
    dora_hit_count: 1,
    dora_first_hit_rank: 1,
  },
  timings_ms: { zero_shot: 9.8, dora: 10.4 },
  results: {
    zero_shot: [
      {
        candidate_id: "text:4",
        filename: null,
        image_id: null,
        rank: 1,
        score: 0.78,
        title: "其他文本",
        text: "北京皇家建筑",
        text_type_label: "验证文本",
        rank_change_vs_other: -1,
        judgement: "非标准答案",
        judgement_reason: "该文本不在当前验证图像的标准答案集合中。",
      },
    ],
    dora: [
      {
        candidate_id: "text:0",
        filename: "deshou.png",
        image_id: 42,
        rank: 1,
        score: 0.88,
        title: "南宋德寿宫遗址",
        text: "南宋德寿宫遗址博物馆",
        text_type_label: "验证文本",
        rank_change_vs_other: 1,
        judgement: "标准答案",
        judgement_reason: "该文本是当前验证图像的标准配对文本。",
      },
    ],
  },
};

describe("App", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("/api/summary")) {
          return new Response(JSON.stringify(summaryPayload));
        }
        if (url.endsWith("/api/examples")) {
          return new Response(JSON.stringify([]));
        }
        if (url.endsWith("/api/item/deshou.png")) {
          return new Response(JSON.stringify(itemPayload));
        }
        if (url.endsWith("/api/search/compare")) {
          return new Response(JSON.stringify(comparePayload));
        }
        if (url.endsWith("/api/search/image-to-text")) {
          return new Response(JSON.stringify(imageToTextPayload));
        }
        return new Response("Not found", { status: 404 });
      }),
    );
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("renders the compact hero and evidence-first featured case layout", async () => {
    render(<App />);

    expect(screen.getByText("正在整理南宋图文证据链…")).toBeInTheDocument();

    await screen.findByRole("heading", { name: "NanS-CLIP" });
    await screen.findByRole("heading", { name: "当前展示案例" });
    expect(screen.getByText("验证图像")).toBeInTheDocument();
    expect(screen.getAllByText("验证文本").length).toBeGreaterThan(0);
    expect(screen.getAllByText("当前展示").length).toBeGreaterThan(0);
    expect(screen.getAllByText("标准答案对比").length).toBeGreaterThan(0);
    expect(screen.getByRole("heading", { name: "切换验证集案例" })).toBeInTheDocument();
    expect(screen.queryByText("保留少量精选案例，优先展示 DoRA 对标准答案排序更清晰的样本。")).not.toBeInTheDocument();
    expect(screen.getByText("来源站点")).toBeInTheDocument();
    expect(screen.getByText("原始链接")).toBeInTheDocument();
    expect(screen.getAllByText("南宋德寿宫遗址博物馆").length).toBeGreaterThan(0);
    expect(screen.getByRole("heading", { name: "标准答案检索对比" })).toBeInTheDocument();
    expect(screen.queryByText("实时图库")).not.toBeInTheDocument();
  });

  it("shows an error state when summary loading fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.endsWith("/api/summary")) {
          return new Response("boom", { status: 500 });
        }
        return new Response(JSON.stringify([]));
      }),
    );

    render(<App />);

    await screen.findByText("页面初始化失败");
    expect(screen.getByText("请先确认后端服务已经启动。")).toBeInTheDocument();
  });

  it("shows verdict-first summaries and benchmark reasoning in compare results", async () => {
    const user = userEvent.setup();
    render(<App />);

    await screen.findByRole("button", { name: /德寿宫/ });
    await user.click(screen.getByRole("button", { name: /德寿宫/ }));

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: "Zero-Shot" })).toBeInTheDocument();
    });
    expect(screen.getByRole("heading", { name: "DoRA" })).toBeInTheDocument();
    expect(screen.getByText("Top-5 未命中标准答案")).toBeInTheDocument();
    expect(screen.getByText("Top-5 命中 1 · 首个标准答案 #1")).toBeInTheDocument();
    expect(screen.getByText("标准答案")).toBeInTheDocument();
    expect(screen.getByText("非标准答案")).toBeInTheDocument();
    expect(screen.getByText(/标准配对图像/)).toBeInTheDocument();
    expect(screen.getByText(/不在当前验证文本的标准答案集合/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "查看图搜文能力" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "图搜文（可选）" })).not.toBeInTheDocument();
  });

  it("opens the full provenance modal from the featured proof card", async () => {
    const user = userEvent.setup();
    const { container } = render(<App />);

    await screen.findByRole("button", { name: "展开完整溯源档案" });
    await user.click(screen.getByRole("button", { name: "展开完整溯源档案" }));

    const dialog = await screen.findByRole("dialog");
    expect(container.contains(dialog)).toBe(false);
    expect(dialog.parentElement).toBe(document.body);
    expect(screen.getAllByText("Wikimedia Commons").length).toBeGreaterThan(0);
    expect(
      screen.getAllByText(
        "https://example.com/archive/deshou/source/very-long-link/with-query/that-should-wrap-correctly-in-proof-card/deshou.png?source=benchmark&view=full",
      ).length,
    ).toBeGreaterThan(0);
    expect(screen.getAllByText("宫阙旧影，南宋遗踪").length).toBeGreaterThan(0);
  });

  it("keeps image-to-text as a secondary retrieval entry behind a reveal button", async () => {
    const user = userEvent.setup();
    render(<App />);

    await screen.findByRole("button", { name: "查看图搜文能力" });
    await user.click(screen.getByRole("button", { name: "查看图搜文能力" }));
    await user.click(screen.getByRole("button", { name: "使用当前标准样本图" }));

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: "图搜文结果" })).toBeInTheDocument();
    });
    expect(screen.getAllByText("南宋德寿宫遗址博物馆").length).toBeGreaterThan(0);
    expect(screen.getAllByText("验证文本").length).toBeGreaterThan(0);
    expect(screen.queryByText("上传查询图片")).not.toBeInTheDocument();
  });
});
