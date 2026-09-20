# Reproduction and Verification

The reference for reported numbers is [Table 1 of the paper](https://arxiv.org/abs/2608.25531). The repository contains the core pipeline, reward plugins, and runnable examples; both agent weights are available on [Hugging Face](https://huggingface.co/Ameame1002/ClueWeaver).

## What Is Checked

Run from the repository root after installing the dependencies:

```bash
python -m unittest discover -s tests -v
python -m scripts.run_question --input examples/multiple_choice.json --validate-only
python -m scripts.run_question --input examples/binary_claim.json --validate-only
```

These offline checks cover input validation, multiple-choice and binary prompt routing, separation of gold labels from model prompts, reuse of evidence during binary self-calibration, and the vLLM thinking parameter. They use mocked model responses and do not establish model accuracy or GPU compatibility.

The offline checks were run with Python 3.12.3, OpenAI SDK 2.32.0, NumPy 2.4.4, and Pydantic 2.12.5. This is an offline-check environment record, not a validated GPU inference lockfile. `requirements.txt` lists direct dependencies without claiming a fully tested version matrix. Install a Qwen3-compatible Transformers and vLLM release with a PyTorch/CUDA combination supported by your machine.

## Running the Models

Follow the README to download and serve the weights. Then run the toy examples with the dense retriever enabled. `--no-retrieval` is available only for a small endpoint smoke check; it changes the pipeline and must not be used to report paper results.

Binary inputs use `options={}`, `binary_mode=true`, and optional gold labels `TRUE` or `FALSE`. Multiple-choice inputs use options `A`, `B`, `C`, and `D`. Gold labels are only used to score outputs. Without a gold label, the command-line output reports `correct: null`.

The README serving example uses two GPUs and CPU retrieval. The paper's serial latency measurement uses a single GPU and GPU retrieval. These are different execution configurations; timings from the example are not directly comparable to the paper's efficiency figure.

## Reproducing Table 1

Exact reproduction requires the original evaluation examples and preprocessing, both released model weights, the retrieval setup, dataset-specific packing, generation settings, and the same answer parser. The paper evaluates 104 DetectiveQA, 69 InfiniteBench, 26 LongBench v2, and 111 NoCha questions. These are the paper's evaluation subsets, not the complete benchmark datasets.

Dataset files, frozen evaluation splits, a full benchmark runner, and per-question reference predictions are not included in this release. The four packing parameters in the README do not constitute a complete experiment configuration. Full fresh-environment GPU inference and reproduction of the reported scores have not been verified as part of this repository cleanup.

Implementation defaults for inference are 192 generation tokens for the Finder, 768 for the Interpreter, and 512 for its self-calibration pass, overridable with `FILM_CS_FINDER_MAX_TOKENS`, `FILM_CS_INTERPRETER_MAX_TOKENS`, and `FILM_CS_VERIFIER_MAX_TOKENS`. These are distinct from the paper's 128-token cap for end-to-end reader baselines and the 256-token training rollout setting. Do not substitute baseline limits for the agent inference settings.
