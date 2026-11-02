// Label overrides live in this browser only (localStorage). The shipped labels are in data/sample.json; anything you
// change on the Labels page is layered on top, and the Receipts page re-scores with your version.
const KEY = "otg.labels.v1";
export function readOverrides() { try { return JSON.parse(localStorage.getItem(KEY) || "{}"); } catch { return {}; } }
export function writeOverrides(o) { try { localStorage.setItem(KEY, JSON.stringify(o)); return true; } catch { return false; } }
export function loadLabels(sample) {
  const o = readOverrides();
  const labels = new Map(sample.map((s) => [s.id, { label: o[s.id]?.label ?? s.gold, borderline: o[s.id]?.borderline ?? s.borderline, by: o[s.id] ? "you" : s.labeler }]));
  return { labels, edited: Object.keys(o).length > 0 };
}
