import { useEffect, useRef } from "react";

import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import Lenis from "lenis";

/** Smooth scrolling plus a thin progress bar across the top.
 *
 *  Both are switched off under prefers-reduced-motion: hijacking the scroll
 *  wheel is exactly the kind of thing that setting exists to refuse.
 */
export function SmoothScroll() {
  const bar = useRef<HTMLDivElement>(null);

  useEffect(() => {
    gsap.registerPlugin(ScrollTrigger);
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const onScroll = () => {
      const height = document.documentElement.scrollHeight - window.innerHeight;
      if (bar.current) {
        bar.current.style.transform =
          `scaleX(${height > 0 ? Math.min(1, window.scrollY / height) : 0})`;
      }
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();

    let lenis: Lenis | null = null;
    let tick: ((time: number) => void) | null = null;
    if (!reduced) {
      lenis = new Lenis({ lerp: 0.1 });
      lenis.on("scroll", ScrollTrigger.update);
      tick = (time: number) => lenis?.raf(time * 1000);
      gsap.ticker.add(tick);
      gsap.ticker.lagSmoothing(0);
    }

    return () => {
      window.removeEventListener("scroll", onScroll);
      if (tick) gsap.ticker.remove(tick);
      lenis?.destroy();
    };
  }, []);

  return (
    <div aria-hidden className="pointer-events-none fixed inset-x-0 top-0 z-50 h-[3px]">
      <div ref={bar} className="h-full origin-left bg-lime" style={{ transform: "scaleX(0)" }} />
    </div>
  );
}
