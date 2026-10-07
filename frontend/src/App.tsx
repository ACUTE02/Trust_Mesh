import { useEffect, useState } from "react";
import { NavLink, Route, Routes, useLocation } from "react-router-dom";
import { Legend } from "./components/evidence";
import datasets from "./data/datasets.json";
import { GraphPage } from "./pages/GraphPage";
import { Landing } from "./pages/Landing";
import { VoicePage } from "./pages/VoicePage";

function ThemeToggle() {
  const [theme, setTheme] = useState(() => document.documentElement.dataset.theme ?? "light");
  const next = theme === "dark" ? "light" : "dark";
  return (
    <button
      type="button"
      className="cursor-pointer border-0 bg-transparent p-0 text-sm underline"
      onClick={() => {
        document.documentElement.dataset.theme = next;
        try {
          localStorage.setItem("tm-theme", next);
        } catch {
          /* private mode: the choice just isn't remembered */
        }
        setTheme(next);
      }}
    >
      Switch to {next} theme
    </button>
  );
}

function Header() {
  const link = ({ isActive }: { isActive: boolean }) =>
    `py-1 no-underline ${isActive ? "border-b-2 border-ink font-semibold" : "border-b-2 border-transparent hover:border-ink-soft"}`;
  return (
    <header>
      <div className="mx-auto flex max-w-[76rem] flex-wrap items-center gap-x-8 gap-y-2 px-5 py-4 sm:px-8">
        <NavLink to="/" className="text-lg font-semibold no-underline">
          TrustMesh
        </NavLink>
        <nav aria-label="Pages" className="flex gap-6 text-sm">
          <NavLink to="/" end className={link}>
            Overview
          </NavLink>
          <NavLink to="/graph" className={link}>
            Graph Engine
          </NavLink>
          <NavLink to="/voice" className={link}>
            Voice Engine
          </NavLink>
        </nav>
        <div className="ml-auto">
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}

export function DatasetCitations() {
  return (
    <div>
      <table className="stack-sm w-full border-collapse text-sm">
        <caption className="sr-only">Datasets used, with licence and citation</caption>
        <thead>
          <tr className="text-left text-xs text-ink-soft">
            <th scope="col" className="pb-2 pr-4 font-normal">
              Dataset
            </th>
            <th scope="col" className="pb-2 pr-4 font-normal">
              Licence
            </th>
            <th scope="col" className="pb-2 font-normal">
              Citation
            </th>
          </tr>
        </thead>
        <tbody>
          {datasets.datasets.map((d) => (
            <tr key={d.name} className="rule-t align-top">
              <th scope="row" className="py-3 pr-4 text-left font-semibold">
                <a href={d.link} rel="noreferrer">
                  {d.name}
                </a>
                <span className="block font-normal text-ink-soft">{d.usedFor}</span>
              </th>
              <td className="py-3 pr-4" data-label="Licence">{d.licence}</td>
              <td className="py-3" data-label="Citation">{d.citation}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="measure mt-3 mb-0 text-xs text-ink-soft">
        Licences as stated on each dataset's official page when read in October 2026. No dataset file, clip, per-sample
        prediction or trained weight is part of this site. Typefaces:{" "}
        {datasets.fonts.map((f, i) => (
          <span key={f.name}>
            {i > 0 && " and "}
            <a href={f.link} rel="noreferrer">
              {f.name}
            </a>
          </span>
        ))}
        , both under the SIL Open Font License.
      </p>
    </div>
  );
}

function Footer() {
  return (
    <footer className="rule-t mt-24">
      <div className="mx-auto max-w-[76rem] px-5 py-12 sm:px-8">
        <h2 className="mb-4 text-lg">How to read the marks</h2>
        <Legend />
        <h2 className="mt-16 mb-5 border-t border-rule pt-10 text-lg">Datasets and citations</h2>
        <DatasetCitations />
        <p className="mt-12 mb-0 text-sm">
          Built by Ayushmaan Gupta, B.Tech CSE (AI &amp; ML), VIT Bhopal.{" "}
          <a href="https://github.com/ACUTE02" rel="noreferrer">
            GitHub
          </a>{" "}
          and{" "}
          <a href="https://www.linkedin.com/in/ayushmaan-gupta" rel="noreferrer">
            LinkedIn
          </a>
          . A portfolio project, not a product.
        </p>
      </div>
    </footer>
  );
}

function ScrollToTop() {
  const { pathname, hash } = useLocation();
  useEffect(() => {
    if (hash) document.getElementById(hash.slice(1))?.scrollIntoView();
    else window.scrollTo(0, 0);
  }, [pathname, hash]);
  return null;
}

export default function App() {
  return (
    <>
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:bg-sheet focus:p-2">
        Skip to the content
      </a>
      <ScrollToTop />
      <Header />
      <main id="main" className="mx-auto max-w-[76rem] px-5 sm:px-8">
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/graph" element={<GraphPage />} />
          <Route path="/voice" element={<VoicePage />} />
          <Route path="*" element={<p className="py-24">There is no page at this address. Use the links above.</p>} />
        </Routes>
      </main>
      <Footer />
    </>
  );
}
