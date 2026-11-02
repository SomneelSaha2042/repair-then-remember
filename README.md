# Repair Then Remember (NUS UROP)

Undergraduate Research Opportunities Programme (CP3209), School of Computing,
National University of Singapore. Student researcher: **Somneel Saha**.

This repository contains ongoing UROP work on parser-grounded episodic repair
memory for frozen-LLM assistance in reinforcement-learning alpha discovery. It
continues the *Repair Before Refinement* research direction using the
AlphaGen/HARLA shared alpha-pool environment.

Built on [stxe09/alphagen-feedback](https://github.com/stxe09/alphagen-feedback)
and the AlphaGen work credited below. The original project history, authorship
and citation are retained. The new UROP experiments are on
`research/repair-memory`.

Status: experiment infrastructure and live API feasibility checks are available;
the full GPU/PPO comparison has not yet been completed. No improvement from
episodic repair memory is claimed at this stage.

## Repair Then Remember experiment setup

Work branch: `research/repair-memory`. Clone this branch from
`SomneelSaha2042/repair-then-remember` to deploy the UROP experiments.

The first comparison is HARLA (LLM + PPO, no repair), Syntax-Repair (the strongest
repair-before-refinement condition), and Syntax-Repair with episodic repair-time
retrieval. Semantic refinement stays off in all three. Optional `random` and
`shuffled` arms test whether retrieval relevance matters. These are pilot
comparisons, not results yet.

Live feasibility result (2026-11-02): a fixed sample of 600 unmodified GPT-4o
outputs across six themes and temperatures 0/0.7 produced **600/600 parser-valid
formulas (538 unique)** under the current HARLA-compatible parser. Consequently
there were no natural failures to replay; no memory gain can be estimated from
this sample. A secondary posthoc audit without the permissive operator aliases
rejected 3/600 formulas, all binary uses of `Max`. Parser acceptance does not
establish numerical validity or financial quality. Details and limitations are
in `experiments/natural_feasibility_2026-11-02.json`. The full hybrid comparison
and current-pool prompting have not yet been benchmarked.

Reproduce this bounded, API-only feasibility experiment with:

```bash
python -m scripts.natural_repair_benchmark
```

Each seed/condition starts with empty memory. Records enter memory only after the
repair parses and the parser's operator type validation passes. Retrieval uses
earlier completed repairs, the same error class, then operator and string
similarity; default context is one case. Parser implementation/configuration is
fingerprinted and rechecked when loading source-model memory. No market metrics
enter this structural memory. An observed token edit distance is logged; it is
not a proof that the repair is the smallest possible one. Passing the parser
does not establish numerical validity, preservation of intent, or profitability.

### Data status

`data/qlib_bin.tar.gz` exists locally (569,210,863 bytes) and is ignored by Git.
**A clone will not contain the dataset.** Its calendar spans 2000-01-04 to
2026-10-23 and includes CSI300 constituent intervals and all six required
features. Archive identity and inspection findings are in
`experiments/data_snapshot.json`. The archive's source/adjustment policy and
identity with the paper's original BaoStock data are not established.

Three profiles are available:

| Profile | Nominal training window | Evaluation windows |
| --- | --- | --- |
| `paper` | 2012–2021 | 2022 H1, 2022 H2, 2023 H1 |
| `postcutoff` | 2012–2023 | 2024, 2025 H1, 2025 H2 |
| `postrelease` (comparison default) | 2012–2024 | 2025 H1, 2025 H2, 2026 H1 |

The runner purges the last 20 trading days of training to prevent the forward
20-day return labels crossing into evaluation. Thus `paper` matches the nominal
windows but is not an exact reproduction of the historical paper run. The
snapshot, model and purge differ. All arms use the same snapshot and protocol.
Qlib worker count follows `SLURM_CPUS_PER_TASK` (one worker outside Slurm).
The comparison now pins GPT-4o `gpt-4o-2024-11-20`. Its documented knowledge
cutoff is October 2023. The default `postrelease` evaluation starts in 2025,
after the snapshot's November 2024 release, to strengthen temporal separation.
This does not certify the contents of proprietary pretraining data. Freeze the
protocol before inspecting test results. The older profiles remain explicit
replication/diagnostic options, rather than contamination-free benchmarks.

Extract and inspect on Linux, from the repository root:

```bash
tar -xzf data/qlib_bin.tar.gz -C data
python -m scripts.check_data data/qlib_bin --profile paper --output data/manifest-paper.json
python -m scripts.check_data data/qlib_bin --profile postcutoff --output data/manifest-postcutoff.json
python -m scripts.check_data data/qlib_bin --profile postrelease --output data/manifest-postrelease.json
```

### Cluster environment

Code is deployed through Git. On the cluster login node:

```bash
git clone --branch research/repair-memory https://github.com/SomneelSaha2042/repair-then-remember.git
cd repair-then-remember
git remote add upstream https://github.com/stxe09/alphagen-feedback.git
```

For later code updates, run `git pull --ff-only` from that directory before
submitting a new job. Existing running jobs should finish before their checkout
is updated. The dataset and API credentials are excluded from Git and need
one-time setup separately. From Windows PowerShell, transfer the data archive:

```powershell
scp -J somneel@stfjump.comp.nus.edu.sg C:\Users\somne\Documents\UROP\alphagen-feedback\data\qlib_bin.tar.gz somneel@xlogin2.comp.nus.edu.sg:~/
```

In the cluster checkout, extract the data and prepare a protected key file:

```bash
mkdir -p data
tar -xzf ~/qlib_bin.tar.gz -C data
(umask 077; touch .env)
chmod 600 .env
```

Then copy your local `.env` separately to the protected destination, from Windows:

```powershell
scp -J somneel@stfjump.comp.nus.edu.sg C:\Users\somne\Documents\UROP\alphagen-feedback\.env somneel@xlogin2.comp.nus.edu.sg:~/repair-then-remember/.env
```

The credentials remain separate from Git and are never committed. Use your
existing interactive SSH authentication. If the destination directory already
exists, inspect it before cloning or updating it.

On a login node, create a user-owned environment (no sudo). The `xlogin1`
storage probe failed after about 9 MiB in `/tmp` despite free filesystem space,
so use a private temporary directory in home. Home must have several gigabytes
available for CUDA packages; the 32 MiB write probe does not establish that.
Disabling pip's cache avoids keeping another copy of downloaded wheels.
The block stops on the first failed command:

```bash
(
set -e
python3 -m venv ~/.venvs/alphagen
source ~/.venvs/alphagen/bin/activate
export TMPDIR="$HOME/.tmp/alphagen"
mkdir -p "$TMPDIR"
chmod 700 "$TMPDIR"
python -c 'import os, tempfile; assert os.path.samefile(tempfile.gettempdir(), os.environ["TMPDIR"]); print("Temporary directory:", tempfile.gettempdir())'
python -m pip install --no-cache-dir --upgrade pip
python -m pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install --no-cache-dir -r requirements-cluster.txt
python -c 'import torch, openai, qlib, sb3_contrib; print("Torch:", torch.__version__)'
python -m pip check
python -m unittest discover -s tests -v
)
```

Both Slurm scripts set the same home-based temporary directory on compute nodes.

The CUDA 12.6 wheel is chosen for the Titan V's Volta architecture. The original
`requirements.txt` is a broad environment freeze containing nightly CUDA 12.8
Torch pins; use `requirements-cluster.txt` for this experiment. Python 3.12 is
the intended cluster interpreter. Dependencies are a new controlled environment,
not a reconstruction of the manuscript environment.

The official deprecation schedule lists GPT-4o API shutdown for January 14, 2027.
We use the requested GPT-4o snapshot for all arms and expose `--model` explicitly;
verify access in your API project before starting the GPU job. Do not silently
substitute a different model between conditions.

```bash
python -c "from openai import OpenAI; print(OpenAI().models.retrieve('gpt-4o-2024-11-20').id)"
```

Before a GPU job, set `OPENAI_API_KEY` through your shell, cluster secret
handling, or the ignored repository-root `.env` file. The API pilot, comparison
run and replay load that `.env` automatically; an existing environment variable
takes precedence. Do not put the key in a script, Git, or logs. Use an API project with data
sharing disabled. API data is not used for model training by default unless
opted in; `store=False` is also set on every request. This setting does not imply
zero retention or certify absence of historical public data from pretraining.
See [OpenAI data controls](https://developers.openai.com/api/docs/guides/your-data)
and the [GPT-4o model card](https://developers.openai.com/api/docs/models/gpt-4o).

### Run sequence

The next hybrid-system pilot uses one seed, the actual CSI300 data, and 51,200
PPO steps per condition. With 2,048-step rollouts, periodic LLM refreshes occur
at 25,600 and 51,200 steps. Keep the existing 15-token PPO construction limit
and all original grammar rules for comparability; the GPU does not itself make
LLM formulas longer. The job array runs HARLA, repair, and repair-memory one at
a time, with 20 inner pool updates per invocation:

```bash
sbatch scripts/cluster_pilot.sbatch
```

Submit from the repository root after installing dependencies, extracting data,
and configuring the API key. Each array task requests up to three hours; actual
runtime is not yet measured. To check only HARLA first, use `sbatch --array=0
scripts/cluster_pilot.sbatch` on one line. `candidates.jsonl` records LLM formula
characters, lexical tokens, operator count, nesting depth and PPO timestep.
`runtime_train.jsonl` records training-expression evaluation exceptions separately
from parser failures; neither numerical errors nor financial results enter repair
memory. `final.json` includes the actual number of periodic LLM refreshes.

Start with the real API pilot on the local machine (no GPU or market dataset is
required). It makes one generation call and up to eight repair calls per arm:

```bash
python -m scripts.api_pilot
```

This sends DSL instructions and synthetic invalid formulas only, with the same
eight failures in the stateless and memory arms. Review `out/api-pilot/*/results.json`
for parser validity, memory retrieval and token usage before starting PPO. The
scripted clients in unit tests are distinct from this live command. Successful
repairs in this tiny synthetic pilot are wiring checks, not a research result.

First inspect the commands without API calls (this still validates the data):

```bash
python -m scripts.compare_repairs --data data/qlib_bin --seeds 0 --steps 4096 --llm-updates 1 --dry-run
```

Then submit a small three-condition pilot, from the repository root:

```bash
sbatch scripts/cluster_compare.sbatch --data data/qlib_bin --seeds 0 --steps 4096 --llm-updates 1
```

That tests initialization, PPO and reporting. To exercise periodic memory use,
run 26,624 PPO steps (13 rollouts) with `--llm-updates 1`. Once this passes,
the main run is 250,000 requested PPO steps, seeds 0–9, pool size 20, five dropped
and five proposed alphas, LLM refresh every 25,000 steps, 20 inner updates,
GPT-4o snapshot `gpt-4o-2024-11-20`, temperature 0. Each invalid formula
gets one repair call. PPO rounds up to full rollouts. Submit one seed/condition
per job if a full matrix exceeds the site's wall-time limit:

```bash
sbatch scripts/cluster_compare.sbatch --data data/qlib_bin --profile postrelease --conditions memory --seeds 0
```

`--conditions harla repair memory random shuffled`, `--memory-k`, and
`--llm-max-calls` are explicit controls. The call ceiling defaults to 1,000 per
run and fails the job when exceeded. A frozen snapshot and seed reduce sampling
variation; the remote API is not guaranteed bitwise deterministic. The first
matrix matches PPO steps and generation schedule, and logs actual calls/tokens;
it does not yet establish an equal-token or equal-backtest-budget result.

Every run saves `final.json`, original `generations.jsonl`, `repair_events.jsonl`,
verified `repair_memory.jsonl` for memory arms, pool checkpoints and logs. The
comparison directory adds a shared data manifest, commit/dirty status and
`results.json` retaining each seed rather than only aggregate means. Initial
validity, repair success, unique valid formulas, error recurrence and API usage
are separate counters. Repair-time retrieval cannot directly improve initial
validity for a fixed generation. Financial IC/Rank IC and their daily IRs are
evaluation outputs; held-out diagnostics are never passed to the LLM.

For the controlled mechanism test, reuse exactly the same generation corpus:

```bash
python -m scripts.repair_replay path/to/generations.jsonl --mode none --output out/replay-none
python -m scripts.repair_replay path/to/generations.jsonl --mode nearest --output out/replay-nearest
python -m scripts.repair_replay path/to/generations.jsonl --mode random --output out/replay-random
```

Use a new output directory per run. For source→target transfer, supply
`--memory-input path/to/source/repair_memory.jsonl --model TARGET_MODEL`; use a
disjoint target corpus and record source provenance. Generation-time memory,
grammar masks, context-size sweeps and strict equal-token controls remain later
experiments. No empirical benefit is claimed until these runs are collected.

<p align="center">
    <img src="images/logo.jpg" width=275 />
</p>

Automatic formulaic alpha generation with reinforcement learning.

This repository contains the code for our paper *Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning* accepted by [KDD 2023](https://kdd.org/kdd2023/), Applied Data Science (ADS) track, publically available on [ACM DL](https://dl.acm.org/doi/10.1145/3580305.3599831). Some extensions upon this work are also included in this repo.

## Repository Structure

- `/alphagen` contains the basic data structures and the essential modules for starting an alpha mining pipeline;
- `/alphagen_qlib` contains the qlib-specific APIs for data preparation;
- `/alphagen_generic` contains data structures and utils designed for our baselines, which basically follow [gplearn](https://github.com/trevorstephens/gplearn) APIs, but with modifications for quant pipeline;
- `/alphagen_llm` contains LLM client abstractions and a set of prompts useful for LLM-based alpha generation, and also provides some LLM-based automatic iterative alpha-generation routines.
- `/gplearn` and `/dso` contains modified versions of our baselines;
- `/scripts` contains several scripts for running the experiments.

## Result Reproduction

Note that you can either use our builtin alpha calculation pipeline (see Choice 1), or implement an adapter to your own pipeline (see Choice 2).

### Choice 1: Stock data preparation

Builtin pipeline requires Qlib library and local-storaged stock data.

- READ THIS! We need some of the metadata (but not the actual stock price/volume data) given by Qlib, so follow the data preparing process in [Qlib](https://github.com/microsoft/qlib#data-preparation) first.
- The actual stock data we use are retrieved from [baostock](http://baostock.com/baostock/index.php/%E9%A6%96%E9%A1%B5), due to concerns on the timeliness and truthfulness of the data source used by Qlib.
- The data can be downloaded by running the script `data_collection/fetch_baostock_data.py`. The newly downloaded data is saved into `~/.qlib/qlib_data/cn_data_baostock_fwdadj` by default. This path can be customized to fit your specific needs, but make sure to use the correct path when loading the data (In `alphagen_qlib/stock_data.py`, function `StockData._init_qlib`, the path should be passed to qlib with `qlib.init(provider_uri=path)`).

### Choice 2: Adapt to external pipelines

Maybe you have better implements of alpha calculation, you can implement an adapter of `alphagen.data.calculator.AlphaCalculator`. The interface is defined as follows:

```python
class AlphaCalculator(metaclass=ABCMeta):
    @abstractmethod
    def calc_single_IC_ret(self, expr: Expression) -> float:
        'Calculate IC between a single alpha and a predefined target.'

    @abstractmethod
    def calc_single_rIC_ret(self, expr: Expression) -> float:
        'Calculate Rank IC between a single alpha and a predefined target.'

    @abstractmethod
    def calc_single_all_ret(self, expr: Expression) -> Tuple[float, float]:
        'Calculate both IC and Rank IC between a single alpha and a predefined target.'

    @abstractmethod
    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        'Calculate IC between two alphas.'

    @abstractmethod
    def calc_pool_IC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        'First combine the alphas linearly,'
        'then Calculate IC between the linear combination and a predefined target.'

    @abstractmethod
    def calc_pool_rIC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        'First combine the alphas linearly,'
        'then Calculate Rank IC between the linear combination and a predefined target.'

    @abstractmethod
    def calc_pool_all_ret(self, exprs: List[Expression], weights: List[float]) -> Tuple[float, float]:
        'First combine the alphas linearly,'
        'then Calculate both IC and Rank IC between the linear combination and a predefined target.'
```

Reminder: the values evaluated from different alphas may have drastically different scales, we recommend that you should normalize them before combination.

### Before running

All principle components of our expriment are located in [train_maskable_ppo.py](train_maskable_ppo.py).

These parameters may help you build an `AlphaCalculator`:

- instruments (Set of instruments)
- start_time & end_time (Data range for each dataset)
- target (Target stock trend, e.g., 20d return rate)

These parameters will define a RL run:

- batch_size (PPO batch size)
- features_extractor_kwargs (Arguments for LSTM shared net)
- device (PyTorch device)
- save_path (Path for checkpoints)
- tensorboard_log (Path for TensorBoard)

### Run the experiments

Please run the individual scripts at the root directory of this project as modules, i.e. `python -m scripts.NAME ARGS...`.
Use `python -m scripts.NAME -h` for information on the arguments.

- `scripts/rl.py`: Main experiments of AlphaGen/HARLA
- `scripts/llm_only.py`: Alpha generator based solely on iterative interactions with an LLM.
- `scripts/llm_test_validity.py`: Tests on how the system prompt affects the valid alpha rate of an LLM.

### After running

- Model checkpoints and alpha pools are located in `save_path`;
    - The model is compatiable with [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
    - Alpha pools are formatted in human-readable JSON.
- Tensorboard logs are located in `tensorboard_log`.

## Baselines

### GP-based methods

[gplearn](https://github.com/trevorstephens/gplearn) implements Genetic Programming, a commonly used method for symbolic regression. We maintained a modified version of gplearn to make it compatiable with our task. The corresponding experiment scipt is [gp.py](gp.py)

### Deep Symbolic Regression

[DSO](https://github.com/brendenpetersen/deep-symbolic-optimization) is a mature deep learning framework for symbolic optimization tasks. We maintained a minimal version of DSO to make it compatiable with our task. The corresponding experiment scipt is [dso.py](dso.py)

## Trading (Experimental)

We implemented some trading strategies based on Qlib. See [backtest.py](backtest.py) and [trade_decision.py](trade_decision.py) for demos.

## Citing our work

```bibtex
@inproceedings{alphagen,
    author = {Yu, Shuo and Xue, Hongyan and Ao, Xiang and Pan, Feiyang and He, Jia and Tu, Dandan and He, Qing},
    title = {Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning},
    year = {2023},
    doi = {10.1145/3580305.3599831},
    booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
}
```

## Contributing

Feel free to submit Issues or Pull requests.

## Contributors

This work is maintained by the MLDM research group, [IIP, ICT, CAS](http://iip.ict.ac.cn/).

Maintainers include:

- [Hongyan Xue](https://github.com/xuehongyanL)
- [Shuo Yu](https://github.com/Chlorie)

Thanks to the following contributors:

- [@yigaza](https://github.com/yigaza)

Thanks to the following in-depth research on our project:

- *因子选股系列之九十五: DFQ强化学习因子组合挖掘系统*
