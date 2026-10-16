import { Link } from "react-router-dom";
import datasets from "../data/datasets.json";
import { NAV_ORDER, WORLDS, type WorldId } from "../world/worlds";
import { Legend } from "./evidence";

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
              <td className="py-3 pr-4" data-label="Licence">
                {d.licence}
              </td>
              <td className="py-3" data-label="Citation">
                {d.citation}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="measure mt-3 mb-0 text-xs text-ink-soft">
        Licences as stated on each dataset&apos;s official page when read in October 2026. No dataset file, clip, per-sample
        prediction or trained weight is part of this site. Typefaces:{" "}
        {datasets.fonts.map((f, i) => (
          <span key={f.name}>
            {i > 0 && (i === datasets.fonts.length - 1 ? " and " : ", ")}
            <a href={f.link} rel="noreferrer">
              {f.name}
            </a>
          </span>
        ))}
        , all under the SIL Open Font License.
      </p>
    </div>
  );
}

/** The footer ends with the thread to the next page, so the pages read as one path. */
export function Footer({ world }: { world: WorldId }) {
  const i = NAV_ORDER.indexOf(world);
  const next = NAV_ORDER[(i + 1) % NAV_ORDER.length];
  return (
    <footer className="rule-t mt-24 scrim">
      <div className="mx-auto max-w-[76rem] px-5 py-12 sm:px-8">
        {world !== "lost" && (
          <p className="mb-12 mt-0 text-lg" data-cursor="Follow the thread">
            <Link to={WORLDS[next].path} className="font-semibold">
              Next: {WORLDS[next].nav} →
            </Link>{" "}
            <span className="text-ink-soft">{WORLDS[next].idea}.</span>
          </p>
        )}
        <h2 className="mb-4 text-lg">How to read the marks</h2>
        <Legend />
        {/* The Archive and the Workbench carry their own datasets-and-citations section, so the footer does not repeat it there. */}
        {world !== "evidence" && world !== "about" && (
          <>
            <h2 className="mt-16 mb-5 border-t border-rule pt-10 text-lg">Datasets and citations</h2>
            <DatasetCitations />
          </>
        )}
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
