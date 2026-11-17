# MLIRBench — CGO 2027 Artifact Evaluation

Run the default reviewer workflow without paid model calls:

```bash
./ae.sh quickstart
```

The single Docker image, `mlirbench-ae:reviewer`, uses the host's native ARM64 or AMD64 CPU. No model credentials or paid API calls are needed. The workflow verifies published inputs, tests the implementation, replays **mutation → execution → reasoning → judge**, recomputes score totals and paper tables, reproduces downstream prediction metrics, and runs the execution agent locally on one Linalg and one Torch pair.

Open the printed **START_HERE.md**. All new run artifacts go outside the repository into **../MLIRBench-AE-runs/** by default. Console logs are not saved. Set AE_OUTPUT_ROOT to choose another destination.

The paper route uses archived model responses and subjective judge ratings, as permitted by the evaluation guidance. They are identified as replayed inputs. Objective correctness, weighted score arithmetic and aggregate tables are recomputed; printed manuscript values are used only for comparison. Fresh model generation is optional code for additional experiments and may differ from the paper. Reviewers are not asked to obtain API keys or buy credits for the archived reproduction route; author-provided API credentials, credits and an API access server are not part of this submission plan. See [the email guidance and scope](docs/woodruff-guidance.md).

## Individual agents

```bash
./ae.sh pipeline --stage mutation --output /results/01-mutation
./ae.sh pipeline --stage execution --input /results/01-mutation --output /results/02-execution
./ae.sh pipeline --stage reasoning --input /results/02-execution --output /results/03-reasoning
./ae.sh pipeline --stage judge --input /results/03-reasoning --output /results/04-judge
./ae.sh pipeline --stage metrics --input /results/04-judge --output /results/05-metrics
```

These commands cost nothing and explicitly replay archived stages. Inside Docker, /results maps to the external run folder. Each output retains its verified prerequisites and can be resumed independently.

To rerun the execution agent using the real CPU compiler/runtime:

```bash
./ae.sh pipeline --stage execution --mode fresh --id seed1_linalg_prog1_L1_R1
```

Fresh execution uses valid shared inputs, five seeds and two fresh-process repeats per program/input. EQUIV means agreement on that declared suite. Unsupported cases remain unscored. An archived downstream response/score cannot silently attach to a fresh upstream stage.

If local scientific records and bundled copies are missing, restore them with:

```bash
./ae.sh fetch-data
```

The resolver asks yes/no before downloading missing files from the pinned Hugging Face revision and verifies their published hashes. Declining makes no network request. Replay and CPU execution run with Docker networking disabled.

See [ARTIFACT.md](ARTIFACT.md) for explicit platform selection, optional fresh model calls and new test cases; [STATUS.md](STATUS.md) for remaining AE work; and [repository layout](docs/repository-layout.md) for the source map. The sibling ../MLIRBench remains a preserved reference. Published data, labels and manuscript expectations are unchanged.

Compiler/backend reproduction is available through `./ae.sh compiler`; see [the commands](ARTIFACT.md#compiler-and-risc-v-experiments). See [release scope](docs/release-scope.md) for required versus optional experiments, local checkpoint assets and remaining evidence. Code and data licenses are described in [DATA_LICENSE.md](DATA_LICENSE.md).
