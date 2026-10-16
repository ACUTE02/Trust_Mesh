import { DatasetCitations } from "../components/Footer";
import { EngineerLayer, EngineerToggle, ForEngineers } from "../components/engineers";
import { KineticText } from "../components/KineticText";
import { Chapter, useResetStage } from "../world/chapters";

const STEPS: { title: string; body: string; command?: string }[] = [
  {
    title: "Get the files the repository does not hold",
    body: "A fresh clone holds the code, the specs, aggregate results and the site. It holds no dataset, no model weights and no speech encoder, because of their licences and size. The README lists each file, where it comes from and whether it is required.",
  },
  {
    title: "Start everything",
    body: "One command starts the graph database, loads it, starts the scoring service and serves the site on port 8080. On Windows, start.bat does the same.",
    command: "docker compose up --build",
  },
  {
    title: "Open the site and the API docs",
    body: "The site is at http://localhost:8080. The scoring service documents itself at http://localhost:8000/docs.",
  },
  {
    title: "Check every endpoint against the running stack",
    body: "This script calls each endpoint and reports what passed.",
    command: "python scripts/verify_serving.py",
  },
  {
    title: "Check that every number on the site has a source",
    body: "This re-reads each number's source file and compares it with what the site shows.",
    command: "python scripts/check_frontend_numbers.py",
  },
];

export default function AboutPage() {
  useResetStage();
  return (
    <EngineerLayer page="about">
      <Chapter stage={0} className="hero" label="about-h">
        <div className="scrim chapter-card" style={{ maxWidth: "42rem" }}>
          <h1 id="about-h" style={{ fontSize: "var(--text-hero)" }}>
            <KineticText text="How to rebuild TrustMesh yourself." mode="wipe" />
          </h1>
          <p className="measure mt-5 mb-0 text-lg">The drawing behind this page is the whole system, taken apart: data, models, the scoring service, the graph database and this site.</p>
          <p className="measure mt-2 mb-0">Scroll, and it goes back together.</p>
          <div className="mt-6">
            <EngineerToggle />
          </div>
        </div>
      </Chapter>

      <Chapter stage={1} id="run" className="chapter" label="run-h">
        <div className="panel" style={{ maxWidth: "50rem" }}>
          <h2 id="run-h" className="mb-5 text-xl">
            <KineticText text="Run it yourself" mode="wipe" />
          </h2>
          <ol className="m-0 grid list-none gap-5 p-0">
            {STEPS.map((s, i) => (
              <li key={s.title} className="grid gap-1 sm:grid-cols-[2.5rem_minmax(0,1fr)]">
                <span className="marker self-start" aria-hidden>
                  {String(i + 1).padStart(2, "0")}
                </span>
                <div>
                  <p className="m-0 font-semibold">
                    <span className="sr-only">Step {i + 1}. </span>
                    {s.title}
                  </p>
                  <p className="measure m-0 mt-1 text-sm">{s.body}</p>
                  {s.command && (
                    <p className="m-0 mt-2">
                      <code className="font-mono text-sm">{s.command}</code>
                    </p>
                  )}
                </div>
              </li>
            ))}
          </ol>
          <p className="measure mt-6 mb-0 text-sm text-ink-soft">
            Not tested: a start from an empty machine on someone else&apos;s computer, and any public deployment. The stack needs about
            four gigabytes of memory.
          </p>
          <ForEngineers title="without Docker">
            <p className="m-0">
              Start only the database with <code className="font-mono text-xs">docker compose up -d neo4j</code>, load it with{" "}
              <code className="font-mono text-xs">python -m serving.load_neo4j</code>, run the service with{" "}
              <code className="font-mono text-xs">uvicorn serving.api:app</code>, and in the frontend folder run{" "}
              <code className="font-mono text-xs">npm install</code> and <code className="font-mono text-xs">npm run dev</code>.
            </p>
          </ForEngineers>
        </div>
      </Chapter>

      <Chapter stage={2} id="who" className="chapter" label="who-h">
        <div className="scrim chapter-card" style={{ maxWidth: "42rem" }}>
          <h2 id="who-h" className="text-xl">
            <KineticText text="Who built it" mode="wipe" />
          </h2>
          {/* TODO(owner): a short paragraph in your own words about why you built this and what you learned. */}
          <p className="measure mt-4 mb-3" data-todo="owner: short paragraph in your own words">
            TrustMesh is a portfolio project by Ayushmaan Gupta, a student of computer science (AI and machine learning). It is a
            study of how far two fraud checks can be trusted, not a product.
          </p>
          <ul className="m-0 grid list-none gap-1 p-0 text-sm">
            <li>
              <a href="https://github.com/ACUTE02" rel="noreferrer">
                GitHub
              </a>
            </li>
            <li>
              <a href="https://www.linkedin.com/in/ayushmaan-gupta" rel="noreferrer">
                LinkedIn
              </a>
            </li>
            {/* TODO(owner): link the resume file here when it exists. */}
            <li className="text-ink-soft" data-todo="owner: resume link">
              Resume: coming soon.
            </li>
          </ul>
        </div>
      </Chapter>

      <Chapter stage={3} id="checked" className="chapter" label="checked-h">
        <div className="scrim chapter-card" style={{ maxWidth: "42rem" }}>
          <h2 id="checked-h" className="text-xl">
            <KineticText text="Built to be checked" mode="wipe" />
          </h2>
          <ul className="measure m-0 mt-4 grid list-none gap-2 p-0">
            <li>Every number on this site is read from one data file, and each entry names the file it came from and the run it belongs to.</li>
            <li>The build fails if a page shows a result number that is not in that file, or comes live from the scoring service.</li>
            <li>The scoring service tests its own inputs, and refuses what it cannot score instead of guessing.</li>
            <li>What was not tested is listed on the Evidence page, in dashed drawers.</li>
          </ul>
        </div>
      </Chapter>

      <Chapter stage={4} id="datasets" className="below-fold mt-20" label="datasets-h">
        <div className="panel">
          <h2 id="datasets-h" className="mb-5 text-xl">
            Datasets, licences and citations
          </h2>
          <DatasetCitations />
        </div>
      </Chapter>
    </EngineerLayer>
  );
}
