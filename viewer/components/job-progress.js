export function formatDuration(seconds) {
  if (seconds == null || !Number.isFinite(Number(seconds))) return 'estimating…';
  const value = Math.max(0, Math.round(Number(seconds)));
  if (value < 60) return `${value}s`;
  const minutes = Math.round(value / 60);
  return minutes < 60 ? `${minutes} min` : `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

// The phases a job API reports about itself rather than about one of its stages.
const TERMINAL_PHASES = new Set(['done', 'error', 'cancelled']);

/** Shared stage-list and overall-progress renderer for local workshop jobs. */
export class JobProgressPanel {
  constructor({ stages, bar, label, eta }) {
    this.elements = { stages, bar, label, eta };
    this.phases = [];
    this.rows = new Map();
  }

  configure(meta) {
    const normalized = meta || { stages: ['running'], stage_labels: { running: 'Running' } };
    this.phases = normalized.stages.map((phase) => [
      phase,
      normalized.stage_labels[phase] || phase,
    ]);
    this.rows.clear();
    this.elements.stages.innerHTML = '';
    for (const [phase, label] of this.phases) {
      const row = document.createElement('div');
      row.className = 'stage-row';
      row.dataset.phase = phase;
      row.innerHTML = `<span class="stage-dot">○</span><span>${label}</span>` +
        '<span class="stage-detail">queued</span>';
      this.elements.stages.appendChild(row);
      this.rows.set(phase, row);
    }
  }

  reset() {
    for (const row of this.rows.values()) {
      row.className = 'stage-row';
      row.querySelector('.stage-dot').textContent = '○';
      row.querySelector('.stage-detail').textContent = 'queued';
    }
  }

  /** Tick every stage. The job reported completion, so nothing is still running. */
  complete() {
    for (const row of this.rows.values()) this.#mark(row, 'done', '✓', 'done');
  }

  /** Stop the running stage pretending to progress, and leave the rest as they were. */
  halt(detail) {
    for (const row of this.rows.values()) {
      if (row.classList.contains('active')) this.#mark(row, 'failed', '×', detail);
    }
  }

  apply(event) {
    const phase = event.phase;

    // A terminal event is about the job, not a stage, so it never matches a row. Without
    // this the last stage stays "active · estimating…" forever, because a stage is only
    // ticked when a *later* stage starts and there is no later stage (seen 2026-09-21: a
    // finished asset whose Compress row read "estimating…" while its GLB sat on disk).
    if (TERMINAL_PHASES.has(phase) && !this.rows.has(phase)) {
      if (phase === 'done') {
        this.complete();
        this.elements.bar.style.width = '100%';
      } else {
        this.halt(phase === 'cancelled' ? 'cancelled' : 'failed');
      }
      this.elements.label.textContent = event.message || phase;
      this.elements.eta.textContent = event.elapsed_seconds == null
        ? '' : `Took ${formatDuration(event.elapsed_seconds)}`;
      return;
    }
    if (!this.rows.has(phase)) return;

    const current = this.phases.findIndex(([name]) => name === phase);
    for (let index = 0; index < current; index++) {
      this.#mark(this.rows.get(this.phases[index][0]), 'done', '✓', 'done');
    }

    const row = this.rows.get(phase);
    const done = event.stage_pct === 100;
    if (done) {
      this.#mark(row, 'done', '✓', 'done');
    } else {
      // Only say what is actually known. A stage with no sub-progress of its own reports
      // no percentage, and printing a bare "· ~estimating…" for it reads as a stall.
      const measure = event.step != null ? `${event.step}/${event.total}`
        : (event.stage_pct == null ? '' : `${event.stage_pct}%`);
      const eta = event.stage_eta_seconds == null ? '' : `~${formatDuration(event.stage_eta_seconds)}`;
      this.#mark(row, 'active', '●', [measure, eta].filter(Boolean).join(' · ') || 'running');
    }

    this.elements.bar.style.width = `${Math.max(0, Math.min(100, event.overall_pct || 0))}%`;
    this.elements.label.textContent = event.message || phase;
    this.elements.eta.textContent = event.total_eta_seconds == null
      ? (event.elapsed_seconds == null ? 'Total still estimating'
        : `${formatDuration(event.elapsed_seconds)} elapsed`)
      : `${formatDuration(event.elapsed_seconds)} elapsed · ~${formatDuration(event.total_eta_seconds)} left`;
  }

  #mark(row, state, dot, detail) {
    row.className = `stage-row ${state}`;
    row.querySelector('.stage-dot').textContent = dot;
    row.querySelector('.stage-detail').textContent = detail;
  }
}
