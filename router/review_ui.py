from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal


ReviewResult = Literal["correct", "incorrect"]
ReviewSplit = Literal["dev", "test"]


class ReviewDatasetError(RuntimeError):
    pass


class ReviewConflictError(ReviewDatasetError):
    pass


class ReviewDatasetStore:
    """Atomic, single-process review access to the JSONL gold-set candidate."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()

    def state(
        self,
        *,
        split: ReviewSplit,
        after: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            records = self._read()
            available_splits = [
                candidate
                for candidate in ("dev", "test")
                if any(record.get("split") == candidate for record in records)
            ]
            selected_split = (
                split
                if split in available_splits
                else (available_splits[0] if available_splits else split)
            )
            scoped = [
                record
                for record in records
                if record.get("split") == selected_split
            ]
            reviewed = [record for record in scoped if self._is_reviewed(record)]
            correct = [
                record
                for record in reviewed
                if self._review_result(record) == "correct"
            ]
            incorrect = [
                record
                for record in reviewed
                if self._review_result(record) == "incorrect"
            ]
            pending = [record for record in scoped if not self._is_reviewed(record)]
            next_record = self._next_pending(scoped, pending, after)
            return {
                "dataset_id": self._dataset_id(records),
                "available_splits": available_splits,
                "split": selected_split,
                "progress": {
                    "total": len(scoped),
                    "reviewed": len(reviewed),
                    "correct": len(correct),
                    "incorrect": len(incorrect),
                    "remaining": len(pending),
                    "percent": (
                        round(len(reviewed) / len(scoped) * 100, 1)
                        if scoped
                        else 0.0
                    ),
                    "ready_for_evaluation": (
                        bool(scoped) and not pending and not incorrect
                    ),
                },
                "case": self._public_case(next_record) if next_record else None,
            }

    def record(
        self,
        *,
        case_id: str,
        result: ReviewResult,
        reviewer: str,
        revision: str,
    ) -> ReviewSplit:
        reviewer = reviewer.strip()
        if not reviewer:
            raise ValueError("reviewer must not be blank")
        with self._lock:
            records = self._read()
            matches = [record for record in records if record.get("id") == case_id]
            if len(matches) != 1:
                if not matches:
                    raise KeyError(case_id)
                raise ReviewDatasetError(f"Duplicate case id: {case_id}")
            record = matches[0]
            if self._is_reviewed(record):
                if (
                    self._review_result(record) == result
                    and record.get("reviewed_by") == reviewer
                ):
                    return self._record_split(record)
                raise ReviewConflictError(
                    "This case was already reviewed; refresh before continuing."
                )
            if self._revision(record) != revision:
                raise ReviewConflictError(
                    "This case changed after the page loaded; refresh before continuing."
                )
            split = self._record_split(record)
            record["review_result"] = result
            record["verified"] = result == "correct"
            record["review_status"] = (
                "verified" if result == "correct" else "rejected"
            )
            record["reviewed_by"] = reviewer
            record["reviewed_at"] = datetime.now(UTC).isoformat()
            self._write(records)
            return split

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            raise ReviewDatasetError(f"Review dataset does not exist: {self.path}")
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                case_id = record.get("id")
                if not isinstance(case_id, str) or not case_id:
                    raise ReviewDatasetError(
                        f"Missing case id on line {line_number}"
                    )
                if case_id in seen:
                    raise ReviewDatasetError(f"Duplicate case id: {case_id}")
                seen.add(case_id)
                records.append(record)
        except (OSError, json.JSONDecodeError) as exc:
            raise ReviewDatasetError("Unable to read the review dataset") from exc
        return records

    def _write(self, records: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            f".{self.path.name}.{uuid.uuid4().hex}.tmp"
        )
        fd: int | None = None
        try:
            fd = os.open(
                temporary,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                fd = None
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
        finally:
            if fd is not None:
                os.close(fd)
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _review_result(record: dict[str, Any]) -> ReviewResult | None:
        result = record.get("review_result")
        if result in {"correct", "incorrect"}:
            return result
        if record.get("verified") is True:
            return "correct"
        return None

    @classmethod
    def _is_reviewed(cls, record: dict[str, Any]) -> bool:
        return cls._review_result(record) is not None

    @staticmethod
    def _next_pending(
        scoped: list[dict[str, Any]],
        pending: list[dict[str, Any]],
        after: str | None,
    ) -> dict[str, Any] | None:
        if not pending:
            return None
        if not after:
            return pending[0]
        start = next(
            (
                index + 1
                for index, record in enumerate(scoped)
                if record.get("id") == after
            ),
            0,
        )
        pending_ids = {record["id"] for record in pending}
        for offset in range(len(scoped)):
            candidate = scoped[(start + offset) % len(scoped)]
            if candidate["id"] in pending_ids:
                return candidate
        return None

    @staticmethod
    def _public_case(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": record["id"],
            "text": str(record.get("text", "")),
            "context": record.get("context") or [],
            "language": record.get("language"),
            "split": record.get("split"),
            "proposed_route": record.get("expected_route"),
            "revision": ReviewDatasetStore._revision(record),
        }

    @staticmethod
    def _record_split(record: dict[str, Any]) -> ReviewSplit:
        split = record.get("split")
        if split not in {"dev", "test"}:
            raise ReviewDatasetError("Review case has an invalid split")
        return split

    @staticmethod
    def _dataset_id(records: list[dict[str, Any]]) -> str:
        values = {
            str(record.get("dataset_id"))
            for record in records
            if record.get("dataset_id")
        }
        if len(values) > 1:
            raise ReviewDatasetError("Review split contains multiple dataset IDs")
        return next(iter(values), "unversioned-dataset")

    @staticmethod
    def _revision(record: dict[str, Any]) -> str:
        payload = json.dumps(
            {
                "id": record.get("id"),
                "text": record.get("text"),
                "context": record.get("context") or [],
                "expected_route": record.get("expected_route"),
                "review_result": record.get("review_result"),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


REVIEW_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Router 人工审核</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #17211b;
      --muted: #66736b;
      --line: #dce5df;
      --paper: #ffffff;
      --canvas: #f3f7f4;
      --green: #16794a;
      --green-soft: #e8f5ed;
      --red: #b23b35;
      --red-soft: #fbeceb;
      --blue: #265da8;
      --blue-soft: #eaf1fb;
      --shadow: 0 18px 55px rgba(28, 53, 39, .09);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 8% 0%, #e3f1e8 0, transparent 32rem),
        var(--canvas);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
        "Segoe UI", "PingFang SC", "Hiragino Sans GB", sans-serif;
    }
    main {
      width: min(900px, calc(100% - 32px));
      margin: 0 auto;
      padding: 42px 0 64px;
    }
    header {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 24px;
      margin-bottom: 22px;
    }
    h1 { margin: 0 0 6px; font-size: clamp(25px, 4vw, 36px); }
    .subtitle { margin: 0; color: var(--muted); }
    .controls { display: flex; gap: 10px; flex-wrap: wrap; }
    input, select, button {
      border: 1px solid var(--line);
      border-radius: 10px;
      font: inherit;
    }
    input, select {
      min-height: 42px;
      padding: 0 12px;
      background: var(--paper);
      color: var(--ink);
    }
    input { width: 160px; }
    .panel {
      background: var(--paper);
      border: 1px solid rgba(214, 226, 218, .9);
      border-radius: 18px;
      box-shadow: var(--shadow);
    }
    .progress-panel { padding: 18px 20px; margin-bottom: 16px; }
    .progress-line {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      color: var(--muted);
      font-size: 14px;
    }
    .progress-track {
      height: 9px;
      margin: 12px 0 10px;
      overflow: hidden;
      border-radius: 999px;
      background: #e7ede9;
    }
    .progress-bar {
      height: 100%;
      width: 0;
      border-radius: inherit;
      background: linear-gradient(90deg, #2b8a58, #58ad78);
      transition: width .2s ease;
    }
    .stats { display: flex; gap: 18px; color: var(--muted); font-size: 13px; }
    .stats strong { color: var(--ink); }
    .case { padding: clamp(20px, 5vw, 34px); }
    .eyebrow {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 24px;
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: .08em;
    }
    .context {
      display: grid;
      gap: 8px;
      margin-bottom: 18px;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #f8faf8;
    }
    .context:empty { display: none; }
    .message { color: #4f5d55; font-size: 14px; line-height: 1.55; }
    .message strong { color: var(--ink); }
    .prompt {
      min-height: 112px;
      margin: 0 0 26px;
      font-size: clamp(20px, 3.5vw, 28px);
      font-weight: 650;
      line-height: 1.5;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .proposal {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 26px;
      padding: 16px 18px;
      border: 1px solid #cfddef;
      border-radius: 12px;
      background: var(--blue-soft);
    }
    .proposal-label { color: #49627e; font-size: 14px; }
    .route {
      padding: 7px 12px;
      border-radius: 999px;
      background: var(--blue);
      color: white;
      font-size: 15px;
      font-weight: 750;
      letter-spacing: .04em;
    }
    .actions { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    button {
      min-height: 54px;
      padding: 0 18px;
      cursor: pointer;
      font-weight: 700;
      transition: transform .1s ease, opacity .1s ease;
    }
    button:hover { transform: translateY(-1px); }
    button:disabled { cursor: not-allowed; opacity: .5; transform: none; }
    .incorrect { border-color: #efc8c5; background: var(--red-soft); color: var(--red); }
    .correct { border-color: #bddfc9; background: var(--green-soft); color: var(--green); }
    .skip {
      display: block;
      min-height: 36px;
      margin: 13px auto 0;
      border: 0;
      background: transparent;
      color: var(--muted);
      font-size: 13px;
      font-weight: 500;
    }
    .hint {
      margin: 16px 4px 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
      text-align: center;
    }
    .empty { padding: 64px 24px; text-align: center; }
    .empty h2 { margin: 0 0 8px; }
    .empty p { margin: 0; color: var(--muted); }
    .error {
      display: none;
      margin-bottom: 14px;
      padding: 12px 15px;
      border-radius: 10px;
      background: var(--red-soft);
      color: var(--red);
      font-size: 14px;
    }
    [hidden] { display: none !important; }
    @media (max-width: 680px) {
      main { width: min(100% - 20px, 900px); padding-top: 22px; }
      header { align-items: stretch; flex-direction: column; }
      .controls { display: grid; grid-template-columns: 1fr 1fr; }
      input { width: 100%; }
      .stats { justify-content: space-between; gap: 6px; }
      .actions { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>Router 人工审核</h1>
        <p class="subtitle">只判断当前 JSON 分类是否正确。</p>
      </div>
      <div class="controls">
        <input id="reviewer" aria-label="审核员名称" maxlength="100"
          placeholder="审核员名称">
        <select id="split" aria-label="数据集">
          <option value="dev">开发集</option>
          <option value="test">锁定测试集</option>
        </select>
      </div>
    </header>

    <div id="error" class="error" role="alert"></div>

    <section class="panel progress-panel" aria-label="审核进度">
      <div class="progress-line">
        <span id="progressText">正在读取数据…</span>
        <strong id="progressPercent">0%</strong>
      </div>
      <div class="progress-track">
        <div id="progressBar" class="progress-bar"></div>
      </div>
      <div class="stats">
        <span>正确 <strong id="correctCount">0</strong></span>
        <span>错误 <strong id="incorrectCount">0</strong></span>
        <span>剩余 <strong id="remainingCount">0</strong></span>
      </div>
    </section>

    <section id="casePanel" class="panel case" hidden>
      <div class="eyebrow">
        <span id="language"></span>
        <span id="datasetId">人类复核</span>
      </div>
      <div id="context" class="context"></div>
      <p id="prompt" class="prompt"></p>
      <div class="proposal">
        <span class="proposal-label">JSON 当前分类</span>
        <span id="route" class="route"></span>
      </div>
      <div class="actions">
        <button id="incorrect" class="incorrect" type="button">
          错误 <small>（N）</small>
        </button>
        <button id="correct" class="correct" type="button">
          正确 <small>（Y）</small>
        </button>
      </div>
      <button id="skip" class="skip" type="button">暂时跳过（S）</button>
    </section>

    <section id="emptyPanel" class="panel empty" hidden>
      <h2>本组审核完成</h2>
      <p id="emptyText">所有样本均可进入评测。</p>
    </section>

    <p class="hint">
      快捷键：Y 正确，N 错误，S 跳过。错误记录只做标记，不会自动修改分类。
    </p>
  </main>

  <script>
    const ui = {
      reviewer: document.querySelector("#reviewer"),
      split: document.querySelector("#split"),
      error: document.querySelector("#error"),
      progressText: document.querySelector("#progressText"),
      progressPercent: document.querySelector("#progressPercent"),
      progressBar: document.querySelector("#progressBar"),
      correctCount: document.querySelector("#correctCount"),
      incorrectCount: document.querySelector("#incorrectCount"),
      remainingCount: document.querySelector("#remainingCount"),
      casePanel: document.querySelector("#casePanel"),
      emptyPanel: document.querySelector("#emptyPanel"),
      emptyText: document.querySelector("#emptyText"),
      language: document.querySelector("#language"),
      datasetId: document.querySelector("#datasetId"),
      context: document.querySelector("#context"),
      prompt: document.querySelector("#prompt"),
      route: document.querySelector("#route"),
      correct: document.querySelector("#correct"),
      incorrect: document.querySelector("#incorrect"),
      skip: document.querySelector("#skip"),
    };
    const app = { current: null, busy: false };

    ui.reviewer.value = localStorage.getItem("routerReviewer") || "";
    ui.reviewer.addEventListener("change", () => {
      localStorage.setItem("routerReviewer", ui.reviewer.value.trim());
    });
    ui.split.addEventListener("change", () => loadCase());
    ui.correct.addEventListener("click", () => submitReview("correct"));
    ui.incorrect.addEventListener("click", () => submitReview("incorrect"));
    ui.skip.addEventListener("click", () => loadCase(app.current?.id));
    document.addEventListener("keydown", (event) => {
      if (event.target.matches("input, select, textarea")) return;
      const key = event.key.toLowerCase();
      if (key === "y") submitReview("correct");
      if (key === "n") submitReview("incorrect");
      if (key === "s") loadCase(app.current?.id);
    });

    async function loadCase(after = null) {
      setBusy(true);
      clearError();
      const params = new URLSearchParams({ split: ui.split.value });
      if (after) params.set("after", after);
      try {
        const response = await fetch(`/review/api/state?${params}`, {
          headers: { "Accept": "application/json" },
          cache: "no-store",
        });
        if (!response.ok) throw new Error(await errorMessage(response));
        render(await response.json());
      } catch (error) {
        showError(error.message || "无法读取审核数据");
      } finally {
        setBusy(false);
      }
    }

    async function submitReview(result) {
      if (app.busy || !app.current) return;
      const reviewer = ui.reviewer.value.trim();
      if (!reviewer) {
        showError("请先填写审核员名称。");
        ui.reviewer.focus();
        return;
      }
      localStorage.setItem("routerReviewer", reviewer);
      setBusy(true);
      clearError();
      try {
        const response = await fetch(
          `/review/api/cases/${encodeURIComponent(app.current.id)}`,
          {
            method: "POST",
            headers: {
              "Accept": "application/json",
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              result,
              reviewer,
              revision: app.current.revision,
            }),
          },
        );
        if (!response.ok) throw new Error(await errorMessage(response));
        render(await response.json());
      } catch (error) {
        showError(error.message || "保存失败，请重试");
      } finally {
        setBusy(false);
      }
    }

    function render(data) {
      const progress = data.progress;
      ui.split.value = data.split;
      ui.progressText.textContent = `已审核 ${progress.reviewed} / ${progress.total}`;
      ui.progressPercent.textContent = `${progress.percent}%`;
      ui.progressBar.style.width = `${progress.percent}%`;
      ui.correctCount.textContent = progress.correct;
      ui.incorrectCount.textContent = progress.incorrect;
      ui.remainingCount.textContent = progress.remaining;
      ui.datasetId.textContent = data.dataset_id;
      app.current = data.case;
      ui.casePanel.hidden = !data.case;
      ui.emptyPanel.hidden = Boolean(data.case);
      ui.emptyText.textContent = progress.ready_for_evaluation
        ? "所有样本均可进入评测。"
        : "存在 rejected 样本，修正前不能进入正式评测。";
      if (!data.case) return;
      ui.language.textContent = `${data.case.language || "unknown"} · ${data.case.split}`;
      ui.prompt.textContent = data.case.text;
      ui.route.textContent = data.case.proposed_route || "未设置";
      ui.context.replaceChildren();
      for (const message of data.case.context || []) {
        const row = document.createElement("div");
        row.className = "message";
        const role = document.createElement("strong");
        role.textContent = `${roleName(message.role)}：`;
        const content = document.createTextNode(String(message.content || ""));
        row.append(role, content);
        ui.context.append(row);
      }
    }

    function roleName(role) {
      return { user: "用户", assistant: "助手", system: "系统" }[role] || role;
    }

    function setBusy(value) {
      app.busy = value;
      ui.correct.disabled = value;
      ui.incorrect.disabled = value;
      ui.skip.disabled = value;
    }

    function showError(message) {
      ui.error.textContent = message;
      ui.error.style.display = "block";
    }

    function clearError() {
      ui.error.textContent = "";
      ui.error.style.display = "none";
    }

    async function errorMessage(response) {
      try {
        const payload = await response.json();
        return payload.detail || payload.error?.message || "请求失败";
      } catch {
        return "请求失败";
      }
    }

    loadCase();
  </script>
</body>
</html>
"""
