"use strict";

(() => {
  const scene = document.querySelector(".insight-scene");
  const play = document.getElementById("insight-play");
  const stages = [...document.querySelectorAll("[data-insight-stage]")];
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  const descriptions = [
    [
      "The answer is spread across the story.",
      "The attic lock, a key changing hands, and a detail hundreds of paragraphs later all matter.",
    ],
    [
      "Preserve the clues before reasoning.",
      "The Finder retains the three relevant passages, including the small detail that Leon still has the key. Paragraph IDs stay attached.",
    ],
    [
      "Connect evidence, not just keywords.",
      "The attic requires the brass key [12]. Mara gives it to Leon [87]. He still has it [203]. The Interpreter connects these facts in narrative order.",
    ],
    [
      "A grounded answer, with a trace back to the story.",
      "Leon could unlock the attic [12, 87, 203]. The core insight: a compact reasoner benefits when answer-critical evidence survives selection.",
    ],
  ];
  let stage = 0,
    playing = false,
    visible = false,
    started = false,
    timer;
  function icons() {
    if (window.lucide) window.lucide.createIcons();
  }
  function render() {
    scene.dataset.stage = String(stage);
    scene.dataset.playing = String(playing && visible);
    stages.forEach((button, index) =>
      button.setAttribute("aria-pressed", String(stage === index)),
    );
    document.getElementById("insight-stage-title").textContent =
      descriptions[stage][0];
    document.getElementById("insight-stage-description").textContent =
      descriptions[stage][1];
    play.setAttribute(
      "aria-label",
      playing ? "Pause animation" : "Play animation",
    );
    play.title = playing ? "Pause animation" : "Play animation";
    play.innerHTML = `<i data-lucide="${playing ? "pause" : "play"}" aria-hidden="true"></i>`;
    icons();
  }
  function schedule() {
    clearTimeout(timer);
    if (!playing || !visible || document.hidden) return;
    timer = setTimeout(() => {
      if (stage < 3) stage += 1;
      else playing = false;
      render();
      schedule();
    }, 4200);
  }
  play.addEventListener("click", () => {
    started = true;
    if (!playing && stage === 3) stage = 0;
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
  stages.forEach((button, index) =>
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
    { threshold: 0.25 },
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
