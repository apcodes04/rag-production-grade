import { useEffect, useState } from "react";

type Theme = "light" | "dark";
const KEY = "rag-theme";

function apply(theme: Theme) {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem(KEY, theme);
  } catch {
    /* private mode: the choice just will not be remembered */
  }
  // Charts read their colours from CSS variables, which React cannot observe.
  // This tells them to re-read after the theme attribute changes.
  window.dispatchEvent(new CustomEvent("themechange", { detail: theme }));
}

/** A glass switch. The knob carries the sun in light mode, the moon in dark. */
export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("dark");

  // The real value is set by the inline script in index.html before paint;
  // this only syncs React's copy of it.
  useEffect(() => {
    setTheme(document.documentElement.dataset.theme === "light" ? "light" : "dark");
  }, []);

  const dark = theme === "dark";

  return (
    <button type="button" role="switch" aria-checked={dark} aria-label="Dark mode"
            className="glass-toggle"
            onClick={() => {
              const next: Theme = dark ? "light" : "dark";
              setTheme(next);
              apply(next);
            }}>
      <span aria-hidden className="gt-end sun">☀️</span>
      <span aria-hidden className="gt-end moon">🌙</span>
      <span aria-hidden className="gt-knob">
        <span className="gt-glyph">{dark ? "🌙" : "☀️"}</span>
      </span>
    </button>
  );
}

/** Current values of the chart colour variables, kept in step with the theme.
 *  Recharts needs real colour strings, not CSS variables, so they are read
 *  from the computed style and re-read whenever the theme changes. */
export function useChartColors() {
  const read = () => {
    const style = getComputedStyle(document.documentElement);
    const get = (name: string) => style.getPropertyValue(name).trim();
    return {
      c1: get("--c-1"), c2: get("--c-2"), c3: get("--c-3"),
      grid: get("--c-grid"), axis: get("--c-axis"),
      card: get("--card"), line: get("--line"), ink: get("--ink"),
      lime: get("--lime"), ember: get("--ember"), spot: get("--spot"),
    };
  };
  const [colors, setColors] = useState(read);

  useEffect(() => {
    const update = () => setColors(read());
    window.addEventListener("themechange", update);
    return () => window.removeEventListener("themechange", update);
  }, []);

  return colors;
}
