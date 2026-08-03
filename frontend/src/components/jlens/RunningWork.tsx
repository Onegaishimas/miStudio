/**
 * J-space work in flight, for the model on screen.
 *
 * WHY THIS EXISTS. A 45-minute fit burned the GPU with nothing in the panel
 * saying so. The fit card tracks its own submission in component state, so it
 * only ever knew about a fit THIS browser tab had started — a fit queued from
 * the API, from MCP, from a second tab, or before a refresh was invisible, and
 * the only evidence was the GPU meter in the header.
 *
 * SOURCED FROM `task_queue`, NOT FROM LOCAL STATE. That is the same table the
 * System Monitor's Active Operations reads, so the two cannot disagree about
 * what is running — and a job survives a refresh in the list because the row
 * outlives the page.
 */

import { useEffect, useState } from 'react';
import { Loader2 } from 'lucide-react';
import { getActiveTasks } from '../../api/taskQueue';
import { TaskType, type TaskQueueEntry } from '../../types/taskQueue';

const POLL_MS = 5000;

const LABELS: Partial<Record<TaskType, string>> = {
  [TaskType.JLENS_FIT]: 'Fitting a J-lens',
  [TaskType.JLENS_BAND_REPORT]: 'Measuring the band report',
  [TaskType.JLENS_INTERVENTION]: 'Running an intervention',
  [TaskType.JLENS_READOUT]: 'Reading out',
  [TaskType.JLENS_PROBE]: 'Probing',
};

export function isJSpaceWork(entry: TaskQueueEntry): boolean {
  return String(entry.task_type).startsWith('jlens_');
}

interface RunningWorkProps {
  /** Only work for the model on screen; another model's fit is not this page's. */
  modelId: string;
}

export function RunningWork({ modelId }: RunningWorkProps) {
  const [rows, setRows] = useState<TaskQueueEntry[]>([]);

  useEffect(() => {
    let live = true;
    const tick = async () => {
      try {
        const res = await getActiveTasks();
        if (!live) return;
        setRows((res.data ?? []).filter(isJSpaceWork));
      } catch {
        // A polling failure must not replace a real list with an empty one:
        // "nothing is running" and "I could not ask" look identical, and the
        // first is the reading that stops someone investigating.
      }
    };
    void tick();
    const id = window.setInterval(tick, POLL_MS);
    return () => {
      live = false;
      window.clearInterval(id);
    };
  }, []);

  const mine = rows.filter((r) => !modelId || r.entity_id === modelId);
  if (!mine.length) return null;

  return (
    <section className="mb-4 shrink-0 rounded-lg border border-emerald-300 bg-emerald-50 p-3 dark:border-emerald-700 dark:bg-emerald-950/40">
      <div className="flex flex-wrap items-center gap-2">
        <Loader2 className="h-3.5 w-3.5 animate-spin text-emerald-600 dark:text-emerald-400" />
        <span className="text-xs font-medium text-emerald-800 dark:text-emerald-300">
          {mine.length === 1 ? 'Running now' : `${mine.length} jobs running`}
        </span>
      </div>
      <ul className="mt-2 space-y-1.5">
        {mine.map((r) => {
          const pct = r.progress == null ? null : Math.round(r.progress);
          return (
            <li key={r.id} className="flex items-center gap-2">
              <span className="w-44 shrink-0 text-[11px] text-emerald-800 dark:text-emerald-300">
                {LABELS[r.task_type] ?? String(r.task_type)}
              </span>
              <span className="h-1.5 flex-1 overflow-hidden rounded bg-emerald-200 dark:bg-emerald-900">
                <span
                  className="block h-full bg-emerald-500 transition-all"
                  // A null progress renders as an EMPTY bar, never a full one:
                  // a task that has not reported yet is at the start of its
                  // work, and showing 100% would say the opposite.
                  style={{ width: `${pct ?? 0}%` }}
                />
              </span>
              <span className="w-24 shrink-0 text-right font-mono text-[10px] text-emerald-700 dark:text-emerald-400">
                {pct == null ? r.status : `${pct}% · ${r.status}`}
              </span>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
