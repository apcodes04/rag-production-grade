import { useEffect } from "react";

import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

const FROM: Record<string, gsap.TweenVars> = {
  up: { y: 40, opacity: 0 },
  left: { x: -80, opacity: 0 },
  right: { x: 80, opacity: 0 },
  zoom: { scale: 0.86, opacity: 0, rotate: -2 },
  pop: { scale: 0.5, opacity: 0 },
};

/** Reveals anything marked `data-rv` as it scrolls into view.
 *
 *  A MutationObserver drives this, not the page prop alone. The CSS starts
 *  every `data-rv` element at opacity 0, so anything that appears AFTER a scan
 *  stays invisible for good: that is what blanked the Evaluation page, whose
 *  cards arrive asynchronously, and the Evidence panel, which only exists once
 *  an answer comes back. Content that appears late must be scanned late.
 */
export function PageMotion({ page }: { page: string }) {
  useEffect(() => {
    gsap.registerPlugin(ScrollTrigger);
    const calm = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const triggers: ScrollTrigger[] = [];

    const scan = () => {
      const fresh = document.querySelectorAll<HTMLElement>("[data-rv]:not([data-rv-done])");
      if (!fresh.length) return;

      for (const element of fresh) {
        element.dataset.rvDone = "1";
        if (calm) {
          element.style.opacity = "1";
          continue;
        }
        const from = FROM[element.dataset.rv || "up"] ?? FROM.up;
        const tween = gsap.fromTo(element, from, {
          x: 0, y: 0, scale: 1, rotate: 0, opacity: 1,
          duration: 0.6, ease: "power3.out",
          scrollTrigger: {
            trigger: element,
            // Anything already on screen reveals at once. Without this, an
            // element that mounts above the fold never crosses its start line
            // and simply never appears.
            start: "top 95%",
            once: true,
          },
        });
        if (tween.scrollTrigger) triggers.push(tween.scrollTrigger);
      }
      ScrollTrigger.refresh();
    };

    scan();
    const observer = new MutationObserver(scan);
    observer.observe(document.body, { childList: true, subtree: true });

    return () => {
      observer.disconnect();
      triggers.forEach((trigger) => trigger.kill());
    };
  }, [page]);

  return null;
}

/** Moves the lime spotlight on `.spot-card` elements to follow the cursor. */
export function useSpotlight() {
  useEffect(() => {
    const onMove = (event: PointerEvent) => {
      const card = (event.target as HTMLElement)?.closest<HTMLElement>(".spot-card");
      if (!card) return;
      const box = card.getBoundingClientRect();
      card.style.setProperty("--mx", `${event.clientX - box.left}px`);
      card.style.setProperty("--my", `${event.clientY - box.top}px`);
    };
    window.addEventListener("pointermove", onMove, { passive: true });
    return () => window.removeEventListener("pointermove", onMove);
  }, []);
}
