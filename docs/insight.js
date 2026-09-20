"use strict";

(() => {
  const scene = document.querySelector(".insight-scene");
  if (!scene) return;
  const play = document.getElementById("insight-play");
  const buttons = [...document.querySelectorAll("[data-insight-stage]")];
  const sourceLines = [...document.querySelectorAll("[data-paragraph]")];
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  const content = document.getElementById("operation-content");
  const criticalIds = [12, 87, 203];
  const contextIds = [86, 88];
  const packetIds = [...criticalIds, ...contextIds].sort((a, b) => a - b);
  const paragraphs = new Map(
    sourceLines.map((line) => [
      Number(line.dataset.paragraph),
        line.querySelector("p").textContent.replace(/\s+/g, " ").trim(),
    ]),
  );
  const rationales = [
    "[12] establishes the lock requirement.",
    "[87] transfers the key to Leon; [86] and [88] preserve the local context.",
    "[203] establishes continued possession after Mara leaves.",
  ];
  const packetText =
    packetIds.map((id) => `[${id}] ${paragraphs.get(id)}`).join("\n") +
    "\nFinder rationale: " +
    rationales.join(" ");
  const budget = 900;
  const icon = (name) => `<i data-lucide="${name}" aria-hidden="true"></i>`;
  const cite = (id) => `<span class="cite-ref">[${id}]</span>`;
  const escape = (text) =>
    text
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  const windows = [
    {
      id: "S1",
      range: "[202-204]",
      kind: "Retrieval anchor",
      subject: "Later possession",
      yes: true,
      reason: "Leon still has the key after Mara leaves.",
      ref: 203,
    },
    {
      id: "S2",
      range: "[11-13]",
      kind: "Retrieval anchor",
      subject: "The attic lock",
      yes: true,
      reason: "The brass key is required to open the attic.",
      ref: 12,
    },
    {
      id: "S3",
      range: "[40-42]",
      kind: "Local window",
      subject: "Evening weather",
      yes: false,
      reason: "Weather gives no evidence about access.",
      ref: null,
    },
    {
      id: "S4",
      range: "[86-88]",
      kind: "Retrieval anchor",
      subject: "The key changes hands",
      yes: true,
      reason: "Mara gives the key to Leon before leaving.",
      ref: 87,
    },
    {
      id: "S5",
      range: "[87-89]",
      kind: "Overlapping window",
      subject: "The same handover",
      yes: true,
      reason: "The same transfer supports possession.",
      ref: 87,
    },
  ];
  const answerReason =
    "The attic opens with the brass key [12]. Mara gave it to Leon [87], who still had it after she left [203]. The claim is supported.";
  const states = [
    {
      operation: "INPUT",
      title: "One claim. Clues hundreds of paragraphs apart.",
      description:
        "The answer depends on the lock, the key changing hands, and who still has it later.",
      stamp: "A whole story",
      output: "Input to the pipeline",
      meta: "Narrative + claim",
      html: `<p>A paragraph-indexed narrative and a binary claim enter together.</p><dl class="input-record"><div><dt>SOURCE</dt><dd>Original narrative with stable paragraph IDs.</dd></div><div><dt>CLAIM</dt><dd>Leon could unlock the attic after Mara left.</dd></div><div><dt>ANSWER SPACE</dt><dd>TRUE / FALSE</dd></div></dl><p class="output-note">No gold answer is supplied at inference. Only six selected excerpts are shown here.</p>`,
    },
    {
      operation: "RETRIEVAL + SEGMENTATION",
      title: "Find anchors, then preserve their surroundings.",
      description:
        "Lexical and BGE-M3 scores guide short anchor segments. Local windows cover the remaining narrative; paragraph IDs stay attached.",
      stamp: "Where are the clues?",
      output: "Illustrative candidate windows",
      meta: "Five windows shown",
      html: `<ol class="candidate-list">${windows.map((w) => `<li><span class="window-id">${w.id}</span><span class="range">${w.range}</span><span>${w.subject}<br><em>${w.kind}</em></span></li>`).join("")}</ol><p class="output-note">Retrieval guides segmentation within the given story. It does not retrieve from the web or determine the final evidence set.</p>`,
    },
    {
      operation: "FINDER / SELECT",
      title: "Keep useful clues, even when each is only partial.",
      description:
        "The Finder returns YES/NO for each segment, a brief rationale, and paragraph citations. The weather-only segment is rejected.",
      stamp: "Keep the useful bits!",
      output: "Finder decisions",
      meta: "4 YES / 1 NO",
      html: `<ol class="finder-decisions">${windows.map((w) => `<li><span class="window-id">${w.id}</span><span class="decision ${w.yes ? "" : "no"}">${w.yes ? "YES" : "NO"}</span><span>${w.reason}</span>${w.ref ? cite(w.ref) : '<span class="cite-ref"></span>'}</li>`).join("")}</ol><p class="output-note">Two selected windows cite [87]. Its repeated appearance is resolved during packing, not treated as two independent facts.</p>`,
    },
    {
      operation: "EVIDENCE / PACK",
      title: "Turn overlapping clues into one ordered packet.",
      description:
        "Merge repeated [87], retain the nearby context [86, 88], then sort by narrative order. Evidence text and Finder rationales travel together.",
      stamp: "A tidy clue bundle",
      output: "Packed evidence E",
      meta: "5 paragraphs / 3 groups",
      html: `<div class="pack-actions"><span>${icon("copy-minus")}Deduplicate</span><span>${icon("between-horizontal-start")}Keep neighbors</span><span>${icon("list-ordered")}Order</span><span>${icon("ruler")}Budget</span></div><div class="dedup-trace"><span>Cited clues</span>${cite(203)} ${cite(12)} ${cite(87)} <del title="Duplicate citation removed">[87]</del><span class="dedup-note">1 duplicate removed</span></div><ol class="packet-lines">${packetIds.map((id) => `<li data-packet-id="${id}" data-kind="${contextIds.includes(id) ? "context" : "clue"}"><span class="paragraph-id">[${id}]</span><span>${paragraphs.get(id)}</span></li>`).join("")}</ol><div class="budget-meter" role="meter" aria-label="Illustrative evidence character budget" aria-valuemin="0" aria-valuemax="${budget}" aria-valuenow="${packetText.length}"><span style="width:${(packetText.length / budget) * 100}%"></span></div><div class="packet-summary"><span>${packetText.length} / ${budget} characters, including rationales</span><span>No clipping needed</span></div><p class="output-note">Finder notes: ${rationales.join(" ")}</p>`,
    },
    {
      operation: "INTERPRETER / REASON",
      title: "Connect the facts across narrative distance.",
      description:
        "The Interpreter receives the claim and packed evidence, and forms a provisional answer with a paragraph-grounded rationale.",
      stamp: "These clues connect!",
      output: "Evidence-grounded rationale",
      meta: "Same packet E",
      html: `<ol class="reason-chain"><li><strong>REQUIREMENT</strong>The attic needs the brass key. ${cite(12)}</li><li><strong>TRANSFER</strong>Mara hands that key to Leon. ${cite(87)}</li><li><strong>TIME + POSSESSION</strong>Leon still carries it after Mara leaves. ${cite(203)}</li></ol><div class="provisional-answer">Provisional answer <strong>TRUE</strong></div><p class="output-note">The rationale cites original paragraph IDs, not retrieval ranks. Nearby context keeps the handover and pronoun references intact.</p>`,
    },
    {
      operation: "INTERPRETER / SELF-CALIBRATE",
      title: "Re-check the claim against the same evidence.",
      description:
        "This binary verification task triggers an internal self-calibration pass. The Interpreter re-checks its provisional answer without new retrieval.",
      stamp: "Does it all hold up?",
      output: "Self-calibration",
      meta: "Same Interpreter",
      html: `<ol class="check-list"><li>${icon("check")}<span>Object and requirement match.<small>The brass key opens the attic.</small></span>${cite(12)}</li><li>${icon("check")}<span>The recipient is Leon.<small>The handover happens before Mara leaves.</small></span>${cite(87)}</li><li>${icon("check")}<span>The timing supports the claim.<small>Leon still has the key after her departure.</small></span>${cite(203)}</li></ol><div class="evidence-lock">${icon("lock-keyhole")}Unchanged evidence E. No third agent.</div><div class="provisional-answer">Self-calibrated answer <strong>TRUE, retained</strong></div><p class="output-note">The checks shown are a scripted example of a grounded re-check, not a guarantee that self-calibration always corrects an answer.</p>`,
    },
    {
      operation: "OUTPUT / ANSWER + CITATIONS",
      title: "A final answer with a path back to the story.",
      description:
        "The final answer is parsed from the structured output. The rationale preserves [12], [87] and [203] so its supporting evidence can be inspected.",
      stamp: "Clues checked!",
      output: "Final structured output",
      meta: "Claim verification",
      html: `<div class="final-verdict">${icon("circle-check")}<strong>TRUE</strong><span>Supported by the selected evidence</span></div><p>Leon has the required key after Mara leaves.</p><pre class="answer-xml"><code>${escape(`<reason>\n${answerReason}\n</reason>\n<answer>TRUE</answer>`)}</code></pre><p class="output-note">The insight: preserve answer-critical clues before asking a compact model to connect them.</p>`,
    },
  ];
  let stage = 0;
  let renderedStage = -1;
  let playing = false;
  let visible = false;
  let started = false;
  let timer;

  function render() {
    scene.dataset.stage = String(stage);
    scene.dataset.playing = String(playing && visible && !document.hidden);
    if (renderedStage !== stage) {
      const state = states[stage];
      buttons.forEach((button, index) => {
        button.setAttribute("aria-pressed", String(stage === index));
        button.dataset.complete = String(index < stage);
      });
      document.getElementById("insight-counter").textContent =
        `${String(stage + 1).padStart(2, "0")} / 07`;
      document.getElementById("insight-stage-title").textContent = state.title;
      document.getElementById("insight-stage-description").textContent =
        state.description;
      document.getElementById("insight-operation").textContent =
        state.operation;
      document.getElementById("cast-stamp").textContent = state.stamp;
      document.getElementById("output-title").textContent = state.output;
      document.getElementById("output-meta").textContent = state.meta;
      content.innerHTML = state.html;
      // Preserve the exact packet across reasoning and self-calibration.
      content.dataset.evidence = stage >= 3 ? packetText : "";
      sourceLines.forEach((line) => {
        const id = Number(line.dataset.paragraph);
        let state = "unread";
        if (stage === 1) state = "candidate";
        if (stage >= 2)
          state = criticalIds.includes(id)
            ? "keep"
            : contextIds.includes(id)
              ? "context"
              : "drop";
        line.dataset.state = state;
        line.querySelector(".source-state").textContent = {
          unread: "",
          candidate: "WINDOW",
          keep: "KEEP",
          context: "CONTEXT",
          drop: "DROP",
        }[state];
      });
      renderedStage = stage;
    }
    const label = playing ? "Pause animation" : "Play animation";
    if (
      play.getAttribute("aria-label") !== label ||
      !play.querySelector("svg")
    ) {
      play.setAttribute("aria-label", label);
      play.title = label;
      play.innerHTML = icon(playing ? "pause" : "play");
    }
    if (window.lucide) window.lucide.createIcons();
  }

  function schedule() {
    clearTimeout(timer);
    if (!playing || !visible || document.hidden) return;
    timer = setTimeout(() => {
      if (stage < states.length - 1) stage += 1;
      else playing = false;
      render();
      schedule();
    }, 6500);
  }
  play.addEventListener("click", () => {
    started = true;
    if (!playing && stage === states.length - 1) stage = 0;
    playing = !playing;
    render();
    schedule();
  });
  document.getElementById("insight-restart").addEventListener("click", () => {
    started = true;
    stage = 0;
    playing = !reducedMotion.matches;
    render();
    schedule();
  });
  buttons.forEach((button, index) =>
    button.addEventListener("click", () => {
      started = true;
      playing = false;
      stage = index;
      render();
      schedule();
    }),
  );
  new IntersectionObserver(
    (entries) => {
      visible = entries[0].isIntersecting;
      if (visible && !started && !reducedMotion.matches) {
        started = true;
        playing = true;
      }
      render();
      schedule();
    },
    { threshold: 0.15 },
  ).observe(scene);
  document.addEventListener("visibilitychange", () => {
    render();
    schedule();
  });
  reducedMotion.addEventListener("change", () => {
    if (reducedMotion.matches) {
      playing = false;
      render();
      schedule();
    }
  });
  render();
})();
