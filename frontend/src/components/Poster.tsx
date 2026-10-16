import { hash01 as rnd } from "../world/rand";
import type { WorldId } from "../world/worlds";

// The still version of each world: a drawn poster in the world's own colours (CSS variables). It is what shows
// when WebGL is missing or lost, and when the browser asks to save data. It carries no text and no numbers:
// the page's real text sits above it.
const stroke = { fill: "none", strokeLinecap: "round" as const };

function art(world: WorldId) {
  switch (world) {
    case "voice": // contour map of a terrain
      return [0, 1, 2, 3, 4, 5, 6, 7].map((i) => (
        <path
          key={i}
          d={`M0 ${520 - i * 46} C 220 ${420 - i * 40}, 360 ${640 - i * 50}, 600 ${500 - i * 44} S 980 ${380 - i * 36}, 1200 ${480 - i * 44}`}
          stroke="var(--thread)"
          strokeOpacity={0.25 + i * 0.07}
          strokeWidth="2"
          {...stroke}
        />
      ));
    case "graph": // copper threads with beads
      return [0, 1, 2, 3, 4, 5, 6].map((i) => (
        <g key={i}>
          <path d={`M0 ${200 + i * 70} L1200 ${180 + i * 74}`} stroke="var(--thread)" strokeWidth="2" strokeOpacity="0.6" {...stroke} />
          {Array.from({ length: 12 }, (_, j) => (
            <circle key={j} cx={80 + j * 95 + rnd(i * 12 + j) * 40} cy={200 + i * 70 + j * 2} r={5 + rnd(j + i) * 4} fill="var(--accent)" fillOpacity="0.8" />
          ))}
        </g>
      ));
    case "case": // balance
      return (
        <g stroke="var(--thread)" strokeWidth="3" {...stroke}>
          <path d="M600 140 V620 M440 620 H760 M300 300 L900 340" />
          <path d="M300 300 L250 460 H350 Z M900 340 L850 500 H950 Z" strokeOpacity="0.7" />
          <path d="M600 140 L560 100 H640 Z" stroke="var(--accent)" />
        </g>
      );
    case "evidence": // wall of drawers
      return Array.from({ length: 24 }, (_, i) => (
        <g key={i}>
          <rect x={120 + (i % 6) * 170} y={150 + Math.floor(i / 6) * 130} width="150" height="110" stroke="var(--thread)" strokeWidth="2" strokeOpacity="0.55" strokeDasharray={i % 7 === 3 ? "6 6" : undefined} {...stroke} />
          <rect x={170 + (i % 6) * 170} y={190 + Math.floor(i / 6) * 130} width="50" height="10" fill="var(--accent-2)" fillOpacity="0.6" />
        </g>
      ));
    case "about": // blueprint grid with an exploded box
      return (
        <g stroke="var(--thread)" strokeWidth="1.5" {...stroke} strokeOpacity="0.7">
          {Array.from({ length: 13 }, (_, i) => (
            <path key={i} d={`M${i * 100} 0 V800`} strokeOpacity="0.15" />
          ))}
          {Array.from({ length: 9 }, (_, i) => (
            <path key={`h${i}`} d={`M0 ${i * 100} H1200`} strokeOpacity="0.15" />
          ))}
          <path d="M420 300 L600 220 L780 300 L600 380 Z M420 300 V420 L600 500 V380 M780 300 V420 L600 500" />
          <path d="M420 560 L600 480 L780 560 L600 640 Z" strokeDasharray="8 8" />
        </g>
      );
    default: // home and the 404: a mesh
      return Array.from({ length: 60 }, (_, i) => {
        const x = 80 + rnd(i) * 1040, y = 80 + rnd(i + 99) * 640, x2 = 80 + rnd(i + 7) * 1040, y2 = 80 + rnd(i + 55) * 640;
        return (
          <g key={i}>
            <path d={`M${x} ${y} L${x2} ${y2}`} stroke="var(--thread)" strokeOpacity="0.28" strokeWidth="1.2" {...stroke} />
            <circle cx={x} cy={y} r={2 + rnd(i + 3) * 3} fill="var(--accent)" fillOpacity="0.8" />
          </g>
        );
      });
  }
}

export function Poster({ world }: { world: WorldId }) {
  return (
    <svg aria-hidden="true" focusable="false" viewBox="0 0 1200 800" preserveAspectRatio="xMidYMid slice" className="fixed inset-0 -z-[1] h-full w-full" data-poster={world}>
      {art(world)}
    </svg>
  );
}
