# Optional live-agent developer reference

Archived replay needs no credentials and never constructs model clients. Fresh CPU execution also needs no model credentials. Fresh model stages require `--allow-paid-api`. Before data preparation or any agent runs, the maintained runner resolves the selected model keys, checks the credentials needed by every requested fresh stage, validates request budgets, and constructs the required SDK clients without making requests. An invalid model, missing later-stage credential, invalid budget or SDK setup error stops the run before an earlier stage can incur charges.

This local preflight does not authenticate with providers or establish model availability, quota, endpoint access or output quality. A syntactically valid provider model ID can still be rejected at request time. There is no automatic substitute model. Fresh results are new observations and can differ from archived paper results.

## Optional local configuration and exact identities

The required AE route uses archived responses and needs none of these settings. Author-provided credentials, credits and an API access service are not included or promised. This table documents how the optional code works for users who independently choose to run it; evaluators are not asked to buy access. Such users may set their own configuration in the environment or use `AE_ENV_FILE` pointing to a private Docker environment file outside the repository. Do not commit credentials. The launcher forwards the named settings below only when `--allow-paid-api` is present.

| Stage | Provider/model sent | Required variable | Selection or override |
|---|---|---|---|
| Mutation | OpenAI `gpt-5-mini` | `OPENAI_API_KEY` | `MLIRBENCH_MUTATION_MODEL` (exact OpenAI model ID) |
| Reasoning | Selected registry entry below | Variable in registry table | `--model MODEL_KEY` |
| Judge | OpenAI `gpt-5.4-mini` | `OPENAI_API_KEY` | `OPENAI_MODEL` (exact OpenAI model ID) |

A judge-only fresh run needs the judge credential, not the evaluated reasoning model's credential. A reasoning-only fresh run needs only that reasoning provider's credential. A full fresh run needs mutation, reasoning and judge credentials before mutation starts.

| Reasoning key | Provider | Exact request model ID | Credential |
|---|---|---|---|
| `gemini-3.1-pro-preview` | Google | `gemini-3.1-pro-preview` | `GOOGLE_API_KEY` |
| `gpt-4.1-mini` | OpenAI | `gpt-4.1-mini` | `OPENAI_API_KEY` |
| `claude-sonnet-4-5` | Anthropic | `claude-sonnet-4-5` | `ANTHROPIC_API_KEY` |
| `claude-haiku-4-5` | Anthropic | `claude-haiku-4-5` | `ANTHROPIC_API_KEY` |
| `qwen3-coder-next` | OpenRouter | `qwen/qwen3-next-80b-a3b-instruct` | `OPENROUTER_API_KEY` |
| `llama-4` | OpenRouter | `meta-llama/llama-4-maverick` | `OPENROUTER_API_KEY` |
| `glm-5.1` | OpenRouter | `z-ai/glm-5.1` | `OPENROUTER_API_KEY` |
| `kimi-k2.6` | OpenRouter | `moonshotai/kimi-k2.6` | `OPENROUTER_API_KEY` |
| `gpt-oss-120b` | OpenRouter | `openai/gpt-oss-120b` | `OPENROUTER_API_KEY` |
| `gemini-3-flash` | Google | `gemini-3-flash-preview` | `GOOGLE_API_KEY` |
| `gemini-2.5-flash` | Google | `gemini-2.5-flash` | `GOOGLE_API_KEY` |
| `gemini-2.5-pro` | Google | `gemini-2.5-pro` | `GOOGLE_API_KEY` |

OpenRouter entries use `https://openrouter.ai/api/v1`. The table records the current preserved registry mapping; model keys alone do not prove historical provider identity. In particular, `qwen3-coder-next` currently sends the listed Qwen3-Next Instruct ID. Its historical identity still needs author confirmation; this change does not silently replace that mapping. Provider ID/access checks for other existing entries remain necessary before a large live experiment.

Legacy aliases remain `openai` → `gpt-4.1-mini`, `claude` → `claude-sonnet-4-5`, and `gemini` → `gemini-3-flash`. Use `--model gemini-3.1-pro-preview` for the paper's Gemini model. The generic `gemini` alias selects a different model. Google documents the exact Gemini ID in its [model documentation](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview); this is not evidence of access through a particular reviewer's credentials. `GEMINI_API_KEY` alone does not satisfy this registry's `GOOGLE_API_KEY` requirement.

## Request limits

These are finite request budgets for new runs, not historical generation settings or monetary caps. Tokens can include provider reasoning tokens. A truncated or failed response remains a failure; it is never replaced with an archived score.

| Environment variable | Default | Allowed range |
|---|---:|---|
| `MLIRBENCH_API_TIMEOUT_SECONDS` | 180 | finite, greater than 0, at most 3600 |
| `MLIRBENCH_API_MAX_RETRIES` | 0 | integer 0–5, SDK retries per request |
| `MLIRBENCH_MUTATION_MAX_OUTPUT_TOKENS` | 8192 | integer 1–131072 |
| `MLIRBENCH_REASONING_MAX_OUTPUT_TOKENS` | registry default | integer 1–131072 |
| `MLIRBENCH_JUDGE_MAX_OUTPUT_TOKENS` | 4096 | integer 1–131072 |
| `MLIRBENCH_MUTATION_MAX_ATTEMPTS` | 3 | integer 1–10 |

Registry reasoning defaults remain 4096 tokens, except `llama-4` at 2048. SDK retries default to zero; reasoning and judging each make one agent attempt. Mutation retains the original maximum of three generation/validation attempts. The maximum request attempts for one mutation case are `mutation_max_attempts × (sdk_retries + 1)`; for one reasoning or judge case, `sdk_retries + 1`. Timeout applies to each SDK request, not the entire experiment. The full workload can grow with `--all`.

Each fresh run saves `live_configuration` in its `manifest.json`: resolved model IDs, provider names, credential variable names and budgets, without credential values. Archived replay's records and metric calculations are unchanged. Tests construct every registered provider SDK and intercept request payloads with dummy credentials and networking disabled. Real provider authentication and paid inference have not been verified by those tests.

For durable case recovery, see [continuing a stopped fresh run](../ARTIFACT.md#continue-a-stopped-fresh-run). Continuation requires the same resolved settings and source fingerprint and an explicit `--resume-incomplete`; changing model identities or budgets starts a separate experiment.

Self-contained fresh checkpoints can continue without published dataset views at runtime. If host bundles are missing and a matching local image already exists, `AE_SKIP_BUILD=1` reuses it without a build or image pull; see the reviewer instructions. Historical replay retains its frozen-source checks.

Optional live judging validates raw responses before computing any score. Required fields are a scores object with result_accuracy in [0,10] and an integer-valued explanation_rating in [0,4], an evidence object and nonempty reasoning text. Booleans, strings, NaN/infinities, duplicate JSON keys, malformed/truncated responses and out-of-range ratings are rejected. Supplied component maxima must match the actual mutation rubric. The failed case journal retains raw response text, its SHA-256 and schema errors, without a successful score. No API headers or credential values are captured. Valid ratings keep the existing scoring formula; archived observations are not passed through this new live-response validator.

The live adapter passes the original recorded rule and inspection metadata to the preserved judge. Archived pairs resolve their explicit rule ID from the packaged catalog and retain pipeline_metadata; provenance identifies this lookup and does not claim the current catalog proves the historical prompt. Missing metadata is identified as not recorded. Unknown or conflicting rule identities fail before the judge call. This is a correction to future optional judging, not a retroactive change to published ratings.

## Fresh failure handling versus the historical paper protocol

The accepted paper describes fallback zero subjective scores after judge request/parse failure (Appendix E, archived text lines 1189–1193), and clamping parsed judge scores (lines 1287–1292). New optional live judging instead rejects invalid responses and records an **unscored failed attempt**. This is an intentional validation change for new experiments, not an exact rerun of the historical failure protocol. A fresh experiment must report attempted, completed, failed and unscored counts; an average over successful cases alone cannot be presented as the paper's full-dataset score. Frozen historical judge observations, including any recorded fallback values, are replayed unchanged. No new fallback scores are invented.

New mutation generation uses ordinary compiler verification, without `--verify-diagnostics`: expected-error comments must not turn a failing program into a successful validation. Levels 1–4 require compiler acceptance; failed attempts are retained and the pipeline stops if none passes. Level 5 requires an explicit catalog rule, a compiler-valid original, a different complete module envelope and a positive compiler error return code with a recognized verification diagnostic. The dependency-breaking R4 rules also accept a recognized undefined-SSA diagnostic. Other parser/unknown-operation failures are conservatively rejected; a balanced module alone is insufficient. Accepted Level 5 candidates carry `EXPECTED_INVALID`; this validates the declared error-test shape, not full semantic compliance with every rule. Review the saved candidate and diagnostics for rule adherence. Compiler infrastructure failures stop immediately.

Fresh failed reasoning attempts retain full returned text, its pre-redaction SHA-256, parse/schema status and error details in the case journal, without attaching a fabricated label or score downstream. Credentials and authorization headers are redacted from new diagnostic records; a completed result containing a configured credential is rejected before persistence. A hash of original text will therefore differ from a redacted displayed response. Never share environment files or provider credentials. These protections do not rewrite any frozen historical records.

OpenRouter-compatible clients explicitly use Chat Completions and send registry controls via `extra_body`. OpenRouter's `reasoning.exclude` hides returned reasoning; it does not disable computation or guarantee lower token usage, latency or charges. Its Responses API does not honor that setting. Model availability/authentication remains unverified without independently arranged access; no automatic substitute model is selected.
