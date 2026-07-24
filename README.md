# RSI

## A1 — ReST-EM/STaR self-improvement with a pass@k diagnostic

This project reproduces a bounded language-model self-improvement loop:

1. sample multiple solutions from the current model;
2. verify their final answers with an external correctness checker;
3. keep correct solutions;
4. supervised fine-tune on those model-generated solutions; and
5. repeat for a small, predeclared number of iterations.

The main research question is not merely whether iterative self-training raises
single-sample accuracy. It is:

> Does iterative verifier-filtered self-training expand the set of problems the
> starting model can solve, or mainly increase the probability of sampling
> solutions that were already present in the starting model's distribution?

The primary diagnostic is a matched `pass@k` comparison, for every integer
`k = 1, ..., 256`, between the iteration-0 checkpoint and every iterated
checkpoint. The expected pattern is that `pass@1` improves and then plateaus.
The large-`k` curve determines whether that gain is accompanied by broader
sampled problem coverage or by a narrower, better-weighted distribution.

## Execution boundary: local editing only

Local work is restricted to reading, creating, and editing repository files,
scripts, configurations, and documentation.

Do **not** run any of the following on the local machine:

- dataset download or preparation;
- model download, loading, or conversion;
- sample generation or any other inference;
- SFT, RL, or any other model training;
- benchmark evaluation, verifier sweeps, or `pass@k` computation;
- smoke tests that execute the experimental pipeline.

All experiments—including data preparation that executes project code,
training, inference, verification at scale, and evaluation—must run on the
cluster. The known Explorer configuration, local-to-cluster Git workflow, and
VS Code Remote-SSH diagnosis are recorded in
[`docs/cluster_workflow.md`](docs/cluster_workflow.md). Environment package
pins and launch files are prepared locally, but their installation, validation,
and execution are permitted only through cluster compute-node jobs.

The cluster implementation and exact first-run sequence are staged in
[`docs/experiment_runbook.md`](docs/experiment_runbook.md). Begin with its
environment and preflight jobs, then use only the labeled smoke configuration;
do not jump directly to a full scientific run.

## Current reproduction progress

Last updated: 2026-07-23

Environment and preflight validation completed at commit
`c77baefeea097796b3015e36a7b11e1e71774b67`. The first smoke chain ran at
commit `784d8849000d24aec207eb7d924c2f9ff6a73445`:

| Stage | Status | Recorded evidence |
| --- | --- | --- |
| Repository experiment scaffold | Complete | The pinned configuration, environment setup, preflight checks, unit tests, staged Slurm pipeline, resumable artifact contracts, and runbook are committed. |
| Isolated cluster environment | Passed | Setup job `8629227` created `/home/zha.j/.conda/envs/rsi-restem`, installed the pinned stack, passed `pip check`, detected an NVIDIA H200 with CUDA 12.8, validated scratch storage, and ended with `Environment ready`. |
| Explicit cluster preflight | Passed | Job `8635965` ran on host `d4052`. The fail-fast job reached the final successful preflight message, so the preceding unit-test command exited successfully. Imports resolved inside `rsi-restem`, the H200 and CUDA 12.8 were visible, and all required scratch roots were writable. |
| Smoke compute stages | Passed | Jobs `8636989` through `8636996` completed with exit code `0:0`. The chain produced pinned data, M0 and M1 evaluation scores, 56 retained SFT samples from 128 generations, and an M1 checkpoint after the declared two optimizer steps. |
| Matched smoke report | Passed after repair | The immutable repaired report at `report_m0_m1/` was produced by commit `ad85be3ae95897e6b07807e48a4ea11af9dc2680`. It contains rounds M0 and M1, matched 8-sample curves on 16 problems, both score contracts, an `M1_vs_M0` coverage partition, and the required smoke-only warning. The original M0-only report remains preserved as incomplete. |
| Full-run timing calibration | Next | Before selecting a shard count, time one isolated full-configuration M0 evaluation shard under a separate calibration root. The smoke workload is too small to justify the runbook's original eight-shard placeholder safely. |
| Full ReST-EM study | Not started | Do not launch until the timing calibration establishes a shard count with a conservative eight-hour margin. |

The validated cluster artifact root is `/scratch/zha.j/rsi`, with data,
artifacts, and checkpoints under its corresponding subdirectories. The smoke
gate is complete. The next cluster action is the isolated sizing calibration
documented in the runbook; do not launch the full study with an unmeasured shard
count.

Follow the recovery and validation instructions in
[`docs/experiment_runbook.md`](docs/experiment_runbook.md).

## What is and is not being reproduced

| Source | Element adopted here | Important difference |
| --- | --- | --- |
| [STaR](https://arxiv.org/abs/2203.14465) | Generate rationales, retain answer-correct traces, SFT, and iterate; optionally retry failures while revealing the correct answer. | STaR used older models and reasoning datasets. Answer-conditioned rationalization is an ablation here, not part of the primary loop. |
| [ReST-EM](https://arxiv.org/abs/2312.06585) | Offline Generate/E-step and Improve/M-step, binary external feedback, many samples per problem, per-problem caps, and repeated rounds. | The paper used PaLM 2 on MATH/APPS. This project adapts the mechanism to Qwen2.5 and GSM8K/MATH. |
| [Yue et al.](https://arxiv.org/abs/2504.13837) | Large-`k` evaluation as a test of sampling efficiency versus sampled problem coverage. | Yue et al. studied online RLVR models, not ReST-EM. Their conclusion is a hypothesis to test here, not a result to assume. |
| [ProRL](https://arxiv.org/abs/2505.24864) | The competing hypothesis that sufficiently long, stable training can improve both low- and high-`k` performance on some tasks. | ProRL used prolonged GRPO, a distilled 1.5B starting model, 136K mixed-domain tasks, and far more compute. It is not a direct recipe for this offline loop. |

This is therefore a **mechanism reproduction and diagnostic study**, not an
attempt to match any paper's headline numbers. The intellectual contribution is
the controlled `pass@k` comparison across self-training iterations; the sampling,
filtering, and SFT components are established techniques.

The correctness checker plays the role of a verifiable reward, but the primary
update is supervised learning on accepted samples. The core method should be
described as offline reinforced self-training, rejection-sampling fine-tuning,
or self-distillation—not as online policy-gradient RLVR. TinyZero/GRPO is an
optional infrastructure warm-up only and does not answer the main question.

## Pre-registered hypotheses

Let `M0` be the exact starting checkpoint and `Mt` the checkpoint after
self-training iteration `t`.

- **H1 — reweighting/elicitation:** `Mt` improves `pass@1`, but `M0` catches up
  with or exceeds `Mt` at large `k`. The update made existing successful paths
  easier to sample without expanding sampled coverage.
- **H2 — expanded sampled coverage:** `Mt` improves both small- and large-`k`
  performance and repeatedly solves held-out problems that `M0` does not solve
  under the same substantial sampling budget.
- **H3 — plateau or contraction:** gains saturate after one or two iterations,
  or high-`k` performance declines as the accepted-data distribution narrows.
- **H4 — scale dependence:** the 7B model has a higher initial acceptance rate
  and may benefit more from self-training than the 3B model because useful
  solution paths are easier to discover.

H1–H3 are not mutually exclusive across datasets or difficulty groups. ProRL,
for example, reports task-dependent diminish, plateau, and sustained-gain
regimes. Report the observed regime rather than forcing one project-wide label.

## Experimental scope

### Starting checkpoints

The primary study should use the base checkpoints:

- `Qwen/Qwen2.5-3B`
- `Qwen/Qwen2.5-7B`

Base checkpoints best match the capability-boundary question. If instruction-
tuned checkpoints are used for an operationally easier pilot, treat the exact
instruction checkpoint as `M0`, label the run separately, and never compare it
as though it were the pretrained base model. Pin and record the exact model
revision, tokenizer revision, dtype, and chat/prompt template for every run.

Begin with 3B on GSM8K to validate the complete cluster pipeline. Advance to
MATH and then 7B only after generation, verification, SFT, resumption, and
evaluation artifacts pass the acceptance checks below.

The target cluster resource is one NVIDIA H200. The 3B model is the pipeline
shake-down target and the 7B model is the scaled confirmation target. The staged
implementation uses completion-only LoRA SFT followed by a merged checkpoint,
with a conservative single-H200 configuration. Actual memory, throughput, and
wall-time fit must be confirmed by the cluster smoke run and must not be tested
locally.

### Datasets and split discipline

- Use only the official GSM8K and MATH training splits to construct synthetic
  SFT data.
- Create one fixed validation subset from each training split before any
  generation. Use it for checkpoint selection and training diagnostics only.
- Use the official GSM8K test split for held-out evaluation.
- Use a fixed, versioned MATH test manifest. MATH-500 may be the primary
  budgeted evaluation subset; use the full MATH test split only as a declared
  confirmation run.
- Never generate training data from validation or test prompts. Never use test
  results to choose an iteration, prompt, sampling temperature, or checkpoint.
- Store problem IDs and content hashes in immutable manifests so train/eval
  leakage can be checked automatically.

GSM8K is the pipeline and easier-domain study; MATH is the primary harder-domain
test. Results must be reported separately by dataset, and MATH results should
also be broken down by subject and difficulty when metadata permits.

### Implementation stance

There is no assumed modern turnkey ReST-EM repository for this Qwen-based study.
The repository implements a thin, auditable pipeline around Hugging Face
Transformers/TRL for SFT and vLLM for offline sampling. It reuses paper ideas,
not an opaque end-to-end codebase. Exact package, model, and dataset revisions
are pinned in `requirements.txt` and `configs/restem.yaml`.

## Primary ReST-EM loop

For a predeclared three iterations (`t = 0, 1, 2`):

### 1. Generate (offline E-step)

For every eligible training problem, sample multiple complete reasoning traces
and final answers from `Mt` using one fixed prompt and decoding configuration.
The paper-faithful starting configuration is:

- 32 samples per MATH problem;
- temperature `0.7`;
- top-k `40`;
- a fixed maximum completion length; and
- a few-shot prompt containing step-by-step math examples.

For GSM8K or a budgeted pilot, a smaller sample count may be predeclared, but it
must not be changed after looking at evaluation results. Save every sample,
including rejected and unparsable outputs. Generation must be deterministic
with respect to the recorded configuration and seed schedule, sharded,
resumable, and idempotent.

### 2. Verify and filter

Extract the final answer and apply a deterministic dataset-specific verifier:

- GSM8K: normalize the final numeric answer while preserving sign, decimal,
  fraction, and comma semantics.
- MATH: compare normalized symbolic answers with a conservative equivalence
  procedure and explicit timeout/error handling.

An output is accepted only when parsing succeeds and the verifier returns true.
Parsing failures, timeouts, ambiguous equivalence, and verifier exceptions count
as incorrect. Record the raw extraction, normalized prediction, normalized gold
answer, verifier result, and failure reason.

Keep at most 10 accepted solutions per problem, sampled deterministically from
the accepted set. This follows ReST-EM's balancing strategy so easy problems do
not dominate the SFT corpus. Do not fabricate targets for problems with zero
accepted samples; their absence is itself a key coverage statistic.

Final-answer correctness does not prove that a rationale is valid. Before any
capability-expansion claim, manually or independently audit the reasoning for
the small set of decisive “iterated-only” held-out successes, especially when
the verifier can be satisfied by guessing or algebraic accidents.

### 3. Improve (offline M-step)

Train with standard causal-language-model SFT on the accepted synthetic
solutions. The prompt and question are context; apply next-token loss only to
the generated solution target.

The primary, paper-faithful ReST-EM condition is:

- generate round `Dt` with `Mt`;
- initialize the next training run from the same fixed starting checkpoint
  `M0`, not from `Mt`;
- train on the current round's accepted dataset `Dt`; and
- call the resulting checkpoint `M(t+1)`.

Resetting to `M0` each round limits drift and makes each improved model a
distillation of the current generator's accepted behavior. A cumulative-data or
warm-start-from-`Mt` loop changes the algorithm and may be run only as an
explicitly named ablation.

Checkpoint selection uses the fixed training-derived validation split. Preserve
the final checkpoint and the selected checkpoint if they differ. Do not select
using GSM8K test, MATH-500, or MATH test results.

### 4. Iterate and record

Use `M(t+1)` as the generator for the next E-step. For every round, record:

- total, parsed, correct, and retained sample counts;
- number and fraction of problems with at least one correct sample;
- accepted samples per problem and by difficulty;
- duplicate and near-duplicate rates;
- SFT token count, steps, effective batch size, learning-rate schedule, and
  checkpoint source;
- validation `pass@1`, answer-parse rate, completion length, and truncation
  rate; and
- wall-clock time and cluster resource usage.

The planned three iterations should be reported even if they plateau. Stop
early only for a predeclared safety/validity condition such as corrupted data,
verifier failure, non-finite loss, or unusable output-format collapse—not after
inspecting test performance.

## STaR rationalization ablation

The primary loop excludes rationalization. In a separate STaR-style condition,
retry problems with no accepted direct sample by revealing the gold final answer
and asking the current model to produce a rationale leading to it. Re-run the
same verifier and label every retained sample with `source=rationalized`.

Never mix this condition into the primary ReST-EM result. ReST-EM reports that
answer-conditioned rationalization can create false-positive solutions whose
final answer is correct but reasoning is not. Audit rationalized traces at a
higher rate and report direct and rationalized data yields separately.

## Required controls

At minimum, compare:

1. `M0`: untouched starting checkpoint;
2. `M1`: one Generate/Improve round, equivalent to a one-round rejection-
   sampling fine-tuning control;
3. `M2` and `M3`: iterative ReST-EM checkpoints; and
4. the STaR rationalization variant, if budget permits.

Recommended secondary controls are a human-solution SFT baseline with matched
problem count, a single-round synthetic-data baseline with roughly matched
generation budget, and the explicitly labeled warm-start/cumulative-data
variant. These separate the value of self-generated traces, iteration, and
additional sampling compute.

Run at least two independent training seeds for any result that will support a
strong claim. If cluster budget allows only one seed initially, label the result
as a pilot and do not interpret small differences as stable.

## pass@k capability-boundary protocol

Evaluate `M0`, every ReST-EM checkpoint, and every control with the same harness.
For each held-out problem:

1. generate `n = 256` independent samples;
2. use identical prompt text, decoding parameters, maximum tokens, verifier,
   and stop rules for every checkpoint;
3. use temperature `0.6` and top-p `0.95` as the primary Yue/ProRL-aligned
   evaluation setting;
4. count correct samples as `c`; and
5. compute the unbiased estimator for all integer `k` from 1 through 256:

```text
pass@k = mean_problem[1 - C(n - c, k) / C(n, k)]
```

When `n - c < k`, that problem's term is 1. Implement the estimator in a
numerically stable way; do not evaluate large binomial coefficients directly.
Plot the full curve and emphasize `k = 1, 2, 4, 8, 16, 32, 64, 128, 256` in
tables.

Use the same zero-shot evaluation prompt for all checkpoints. If a format-only
demonstration is required for a base checkpoint, declare it as a separate
prompting condition and apply it unchanged to every model. Do not give `M0`
weaker formatting help than the iterated checkpoints.

Report paired bootstrap confidence intervals over problems and bootstrap the
difference curve `pass@k(Mt) - pass@k(M0)`. Also retain the per-problem `c`
counts so the following coverage sets can be inspected:

- solved by both `M0` and `Mt`;
- solved only by `M0` within 256 samples;
- solved only by `Mt` within 256 samples; and
- solved by neither.

### Interpretation rules

- Higher `pass@1` with equal or lower large-`k` performance supports the
  reweighting/elicitation interpretation.
- Higher performance across the curve, especially replicated large-`k` gains
  and audited `Mt`-only successes, supports expanded **sampled coverage**.
- A crossing curve means improved sampling efficiency but reduced breadth at
  the chosen decoding distribution.
- A flat high-`k` result may reflect saturation of an easy benchmark; inspect
  harder subsets rather than declaring no difference.

`M0` failing a problem in 256 samples does not prove that the solution has zero
probability under `M0`. Likewise, an `Mt`-only success does not by itself prove a
new abstract capability. For decisive cases, run a predeclared second batch of
fresh samples and audit the reasoning before using “new capability” language.
The defensible primary claim is about a change in sampled problem coverage under
a fixed prompt, verifier, decoding distribution, and finite budget.

Do not compare curves generated with different temperatures, prompt templates,
token budgets, answer parsers, or checkpoint-selection rules. Such differences
are confounds, not evidence of capability expansion.

## Failure modes to monitor

- **Easy-problem domination:** many accepted traces come from a small easy
  subset. Mitigate with the per-problem cap and report coverage by difficulty.
- **No-support problems:** a model produces no correct sample and therefore
  receives no positive training target. Track this set across iterations.
- **Diversity collapse:** `pass@1` rises while large-`k` coverage, rationale
  diversity, or solution-set overlap contracts.
- **Verifier false positives:** an answer is accepted despite invalid reasoning
  or parser exploitation. Keep verifier tests and audit decisive samples.
- **Train/test leakage:** test prompts or answers enter generation, prompts, or
  checkpoint selection. Enforce immutable manifests and provenance checks.
- **Prompt asymmetry:** the starting model and iterated model receive different
  formatting or reasoning cues during evaluation.
- **Length confounding:** one checkpoint gains a larger effective search budget
  through longer outputs. Fix maximum tokens and report length/truncation.
- **Iteration overfitting:** training acceptance continues to rise while held-
  out performance plateaus or falls.
- **False precision:** one training seed or a small benchmark produces a tiny
  apparent gain. Report uncertainty and replicate important results.

## Artifact and reproducibility contract

Every cluster run must preserve enough information to reconstruct its lineage:

```text
M0 revision
  -> generation config + prompt hash + dataset manifest + seeds
  -> immutable raw generations
  -> verifier version + accepted-sample manifest
  -> SFT config + code revision + checkpoint
  -> evaluation config + raw generations + per-problem scores
  -> pass@k curve + confidence intervals + coverage-set analysis
```

Planned repository responsibilities are:

- `configs/`: versioned model, data, generation, SFT, and evaluation settings;
- `src/generate.py`: sharded and resumable offline sampling;
- `src/verify_math.py`: deterministic answer extraction and verification;
- `src/build_sft.py`: filtering, balancing, provenance, and dataset manifests;
- `src/train_sft.py`: the ReST-EM Improve step;
- `src/eval_passk.py`: matched 256-sample evaluation and stable estimator;
- `src/report.py`: tables, confidence intervals, curves, and coverage sets; and
- `slurm/`: cluster-only setup, preflight, staged pipeline, and dependency-chain
  entry points documented in `docs/experiment_runbook.md`.

Raw generations and checkpoints must be immutable and stored outside Git on
cluster storage. Derived datasets must reference raw sample IDs rather than
copying untraceable text. Every result table must include the code commit,
configuration hash, model revision, dataset-manifest hash, verifier version,
and training/evaluation seeds.

## Acceptance criteria

The reproduction is complete only when:

- the full offline loop runs for three declared iterations on at least one
  model/dataset pair;
- every accepted SFT sample has traceable raw-generation and verifier records;
- `M0` and all iterated checkpoints are evaluated from raw, retained 256-sample
  outputs with one matched harness;
- full `pass@k` curves, paired uncertainty, and per-problem coverage sets are
  reported;
- answer-parser and verifier behavior is tested and decisive traces are audited;
- plateau, regression, and null results are retained rather than hidden; and
- conclusions use “reweighting,” “sampling efficiency,” or “expanded sampled
  coverage” precisely, reserving “new capability” for stronger replicated
  evidence.

## Optional day-1 cluster warm-up

[TinyZero](https://github.com/Jiayi-Pan/TinyZero) may be used on the cluster to
validate an online GRPO/RLVR environment before implementing any future online
control. It is not required for A1, must not delay the offline ReST-EM pipeline,
and must not be presented as evidence for or against the A1 hypothesis.

## References

- Zelikman et al., [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465), arXiv:2203.14465.
- Singh et al., [Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models](https://arxiv.org/abs/2312.06585), arXiv:2312.06585.
- Yue et al., [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837), arXiv:2504.13837.
- Liu et al., [ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models](https://arxiv.org/abs/2505.24864), arXiv:2505.24864.
