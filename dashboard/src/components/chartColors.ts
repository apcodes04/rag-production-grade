import { useMemo } from "react";

/** Recharts needs real colour strings, not CSS variables, so the chart colours
 *  are read once from the computed root style. There is a single theme now, so
 *  they never change after load. */
export function useChartColors() {
  return useMemo(() => {
    const style = getComputedStyle(document.documentElement);
    const get = (name: string) => style.getPropertyValue(name).trim();
    return {
      c1: get("--c-1"), c2: get("--c-2"), c3: get("--c-3"),
      grid: get("--c-grid"), axis: get("--c-axis"),
      card: get("--card"), line: get("--line"), ink: get("--ink"),
      lime: get("--lime"), ember: get("--ember"), spot: get("--spot"),
    };
  }, []);
}
