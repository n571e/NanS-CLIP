# NanS-CLIP Web Demo

## 目录结构

- `backend/`: Flask 推理后端，负责加载 Zero-Shot / DoRA 模型、读取 `valid` 验证集、缓存 benchmark embedding，并在能恢复原始 `filename` 时返回样本溯源。
- `frontend/`: React + Vite 前端，负责展示项目摘要、首屏样本溯源、验证集标准检索对比和可选图搜文入口。

## 当前展示逻辑

`web_demo` 当前采用“benchmark-backed demo”方式：

- 首页以更紧凑的项目摘要开场，但主叙事会立即落到“当前精选案例 + 样本溯源”上，优先建立页面真实性。
- 首屏直接展示代表样本溯源，把来源站点、原始链接和代表文本显式放在最上方。
- `文搜图` 主链路不再接受自由输入，而是只展示来自 `valid` 验证集的精选标准案例。
- `Zero-Shot` 与 `DoRA` 对比时，结果直接显示 `标准答案 / 非标准答案`，并以 `Top-K` 命中摘要、首个命中排名和排序变化作为主要讲解信号；相似度只作为辅助信息。
- `图搜文` 保留为次入口，通过“查看图搜文能力”按钮展开，但同样只使用验证集标准样本，不再接受外部上传图片。
- 对于能从 `valid_imgs.tsv` 恢复回原始 `filename` 的 benchmark 图像，页面会继续展示公开来源与原始链接；未恢复成功的样本仍可参与标准答案检索，但不强行展示溯源卡。

## 启动方式

如果你在独立 worktree 中运行，需要把原始仓库根目录告诉后端，这样它才能读取本地 `data/` 和 `clip_data/`。

```powershell
$env:NANS_CLIP_SOURCE_ROOT='D:\Desktop\KnowledgeBase\多模态\CLIP\NanS-CLIP'
& 'D:\anaconda3\envs\pytorch\python.exe' -m web_demo.backend.run
```

前端开发服务器：

```powershell
cd web_demo/frontend
npm install
npm run dev
```

默认地址：

- 前端：`http://127.0.0.1:5173`
- 后端：`http://127.0.0.1:8765`

`vite.config.ts` 已经把 `/api` 代理到本地后端。

## 已验证

- `GET /api/summary`
- `GET /api/examples`
- `GET /api/item/:filename`
- `POST /api/search/compare`
- `GET /api/eval-images/:image_id`
- `POST /api/search/image-to-text`

真实数据烟测将基于 `valid` 验证集执行，不再使用 `5583` 张 live 图库。

`web_demo` 当前保留 `文搜图` 作为主演示链路，同时在检索对比区提供“查看图搜文能力”次入口，可直接使用当前标准样本图试跑反向检索。
