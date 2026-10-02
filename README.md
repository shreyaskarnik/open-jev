# open-jev

`open-jev` is a browser-focused TypeScript library for **typed decisions**: one piece of text (the _state_) plus any number of typed questions go in, and one forward pass returns a calibrated probability distribution per question. Nothing is generated, so an answer is always one of the options you provided.

It runs open reproductions of the _shape_ of TypeSafe AI's [Jev "System One" model](https://typesafe.ai/blog/introducing-system-one-models-and-jev) via [`@huggingface/transformers` (Transformers.js)](https://huggingface.co/docs/transformers.js/en/index). Everything happens on-device: WebGPU when available, WebAssembly otherwise.

## Models

| Alias                  | Repo                                                                                                                  | Base             | Weights (q4f16 / q4)        | Notes                                                                                                                |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------- | ---------------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `kev-0.6b`             | [onnx-community/kev-0.6b-ONNX](https://huggingface.co/onnx-community/kev-0.6b-ONNX)                                   | Qwen3-0.6B-Base  | 0.34 GB / 0.38 GB           | Default. Small and fast. 8192-token context.                                                                         |
| `kev-4b`               | [onnx-community/kev-4b-ONNX](https://huggingface.co/onnx-community/kev-4b-ONNX)                                       | Qwen3-4B-Base    | 2.3 GB / 2.5 GB             | Most accurate. 8192-token context. Needs a capable GPU.                                                              |
| `open-jev`             | [onnx-community/open-jev-deberta-v3-large-ONNX](https://huggingface.co/onnx-community/open-jev-deberta-v3-large-ONNX) | DeBERTa-v3-large | 0.35 GB / 0.48 GB           | Also ships `fp16` (0.88 GB) and `fp32` (1.75 GB). 512-token context.                                                 |
| `julia-1`              | [SupersonicLabs/Julia-1-ONNX](https://huggingface.co/SupersonicLabs/Julia-1-ONNX)                                     | mmBERT-small     | fp32 only: 0.58 GB          | Multilingual. One sequence per question, 2–20 options. 1024-token default.                                           |
| `laya`                 | [onnx-community/laya-ONNX](https://huggingface.co/onnx-community/laya-ONNX)                                           | ModernBERT-large | fp16 0.85 GB / fp32 1.69 GB | Convai's [Laya](https://huggingface.co/convaiinnovations/laya), English. Calibrated temperatures. 512-token default. |
| `laya-typed-decisions` | [onnx-community/laya-typed-decisions-ONNX](https://huggingface.co/onnx-community/laya-typed-decisions-ONNX)           | ModernBERT-large | fp16 0.85 GB / fp32 1.69 GB | Laya fine-tuned on the four typed-decisions workflows. 1024-token default.                                           |
| `laya-multilingual`    | [onnx-community/laya-multilingual-ONNX](https://huggingface.co/onnx-community/laya-multilingual-ONNX)                 | mmBERT-base      | fp16 0.65 GB / fp32 1.29 GB | Laya for 100+ languages. 1024-token default (the model reads up to 8192).                                            |
| `strands-decider-2b`   | [onnx-community/strands-decider-2B-hobson-v19-ONNX](https://huggingface.co/onnx-community/strands-decider-2B-hobson-v19-ONNX) | Qwen3.5-2B-Base | q8 1.80 GB / q4f16 1.09 GB | Strands Labs' [decider](https://github.com/strands-labs/strands-decider) (v19), a pointer head on a Qwen3.5 torso. Calibrated temperatures. One prompt per question. 4096-token default. |

Pass the alias as `model`, or any Hugging Face repo id whose `config.json` carries an `open_jev`, `kev`, `laya` or `decider` section. The encoding family is detected from that config. Julia 1's export ships without a `config.json`, so `SupersonicLabs/Julia-1-ONNX` is recognized by its repo id and loaded as published.

## Install

```bash
npm install open-jev @huggingface/transformers
```

`@huggingface/transformers` is a peer dependency so your app and this library share one copy of Transformers.js and its model cache.

## Quick start

```ts
import { OpenJev, choice, score, noul } from "open-jev";

const info = await OpenJev.info({ dtype: "q4f16" });
console.log(info.isCached, info.downloadSize, info.device, info.dtype);

const jev = await OpenJev.load({
  model: "kev-0.6b", // default; or "kev-4b", "open-jev", "julia-1", "laya"
  dtype: "q4f16",
  onProgress: ({ progress }) =>
    console.log(`Model download: ${Math.round(progress * 100)}%`),
});

const state =
  "I was charged twice for the same order and nobody answers my emails. I want my money back now.";

const [area, sentiment, refund] = await jev.decide(state, [
  choice("Which product area is the message about?", [
    "fees & charges",
    "refund & dispute",
    "card",
    "other",
  ]),
  score("How positive is the sentiment of this message?", [
    "very negative",
    "negative",
    "neutral",
    "positive",
    "very positive",
  ]),
  noul("The customer is asking for a refund."),
]);

area.choice; // "fees & charges" | "refund & dispute" | "card" | "other"
area.confidence; // 0.59
sentiment.level; // "very negative" | ... | "very positive"
sentiment.score; // 0.91 (expected level index, may fall between levels)
refund.answer; // true
refund.probability; // 0.88 (p(yes))
```

Answers are fully typed: option literals flow from the question into the answer, so `area.choice` is a union of exactly the strings you passed.

## Keyed questions

Pass an object instead of an array and get the answers back under the same keys:

```ts
const answers = await jev.decide(state, {
  area: choice("Which product area?", [
    "fees & charges",
    "refund & dispute",
    "other",
  ]),
  refund: noul("The customer is asking for a refund."),
});

answers.area.choice; // "fees & charges" | "refund & dispute" | "other"
answers.refund.answer; // boolean
```

## Question types

| Builder                                        | Type     | Answer                                                         |
| ---------------------------------------------- | -------- | -------------------------------------------------------------- |
| `choice(instructions, options, descriptions?)` | `choice` | Pick one option. Descriptions render as `option: description`. |
| `score(instructions, levels)`                  | `score`  | Rate on an ordered scale of levels (first = lowest).           |
| `noul(statement, descriptions?)`               | `noul`   | Does the statement hold for the state? (yes/no)                |

Limits per model:

| Model      | `choice` options | `score` levels |
| ---------- | ---------------- | -------------- |
| `open-jev` | 2 to 255         | 2 to 10        |
| `kev-*`    | 1 to 255         | 2 to 255       |
| `julia-1`  | 2 to 20          | 2 to 20        |
| `laya*`    | 2 to 64          | 2 to 20        |
| `strands-decider-2b` | 2 to 255 | 2 to 255     |

`noul` can say what each outcome means: `noul("The problem is already solved.", { false: "The customer still has the problem.", true: "Nothing is left to do." })`. julia-1 and the laya models read these descriptions; julia-1 was trained with them (80.5% on its `noul` benchmark with them, 65.2% with the literal false/true it gets otherwise). On other text they may not help: on the demo app's short messages they made its yes/no answers worse, so test them on your own questions. The other models answer with their fixed no/yes options and ignore them.

The builders are optional sugar. Plain objects work too:

```ts
await jev.decide(state, [
  {
    type: "choice",
    instructions: "Which team should handle this?",
    options: ["billing", "shipping", "other"],
    descriptions: { billing: "Charges, invoices, payment problems" },
  },
  {
    type: "score",
    instructions: "How urgent?",
    options: ["low", "medium", "high"],
  },
  { type: "noul", instructions: "The customer is angry." },
]);
```

### Answer shapes

```ts
type ChoiceAnswer<O> = {
  type: "choice";
  choice: O; // option with the highest probability
  confidence: number; // probability of `choice`
  probabilities: Record<O, number>;
};

type ScoreAnswer<L> = {
  type: "score";
  score: number; // expected level index, 0..levels-1
  normalized: number; // score rescaled to 0..1
  level: L; // level label closest to `score`
  confidence: number; // highest single-level probability
  probabilities: Record<L, number>;
};

type NoulAnswer = {
  type: "noul";
  answer: boolean; // probability >= 0.5
  probability: number; // p(yes)
  confidence: number; // max(p(yes), p(no))
};
```

## API

### `OpenJev.load(options?): Promise<OpenJev>`

Downloads (or reads from cache) the tokenizer and model and resolves to a ready instance. All options are optional:

- `model` (default `"kev-0.6b"`)
  - `"kev-0.6b"`, `"kev-4b"`, `"open-jev"`, `"julia-1"`, `"laya"`, `"laya-multilingual"`, `"laya-typed-decisions"`, `"strands-decider-2b"`, or a Hugging Face repo id / path Transformers.js understands.
- `dtype` (default `"auto"`)
  - `fp32`, `fp16`, `q8`, `q4` or `q4f16` (the kev models only ship `q4` and `q4f16`; `julia-1` only ships `fp32`; the laya models ship `fp32` and `fp16`; `strands-decider-2b` ships `q8`, its default, and `q4f16`).
  - `auto` picks the model's best WebGPU variant (`q4f16` for kev, `fp16` for open-jev) when `shader-f16` is supported, `q4` everywhere else. `julia-1` always uses `fp32`.
- `device` (default `"auto"`)
  - `webgpu`, `wasm`, or `cpu` (Node.js).
  - `auto` picks `webgpu` when available, `cpu` in Node.js, otherwise `wasm`.
- `onProgress`
  - Called with `{ progress, loaded, total }` while files download. `progress` is `0..1`, `loaded` and `total` are bytes. Only fires when the rounded value changes.
- `maxLength` (default `512` for open-jev and laya, `1024` for julia-1, laya-multilingual and laya-typed-decisions, `4096` for strands-decider-2b, `8192` for kev)
  - Context limit. For open-jev the whole sequence; for kev the state plus one question branch; for julia-1, laya and strands-decider-2b each question's own sequence (laya takes up to 8192).
- `temperature`, `maxStateTokens`, `truncation`
  - Defaults for `decide()`, see below.

### `OpenJev.info(options?): Promise<OpenJevInfo>`

Returns model cache/download metadata for a configuration (`model`, `device`, `dtype`) without loading anything.

- `isCached`: whether every required file is in the browser cache.
- `downloadSize`: sum of all required file sizes (bytes).
- `files`: the files Transformers.js will fetch.
- `model`, `family`, `device`, `dtype`: the resolved runtime.

### `jev.decide(state, questions, options?)`

One forward pass, returns typed answers. Per-call options override the defaults given to `load()`:

- `temperature` (default: open-jev's calibrated `1.05`, `1` for kev, julia-1, laya and strands-decider-2b; laya and the decider apply their own calibrated temperatures first)
  - Softmax temperature applied to each question's logits.
- `maxStateTokens` (default `256` for open-jev, `8192` for kev; julia-1, laya and strands-decider-2b give the state whatever each question leaves of `maxLength`)
  - Token budget for the state. It is cut further if the questions would not fit in `maxLength`.
- `truncation` (default `"cut"`)
  - `"cut"` drops trailing state tokens, `"error"` throws when the state does not fit.

`decide()` throws if the questions alone exceed `maxLength`. Concurrent calls are queued and run one after another.

### `jev.countTokens(text): number`

Number of tokens `text` occupies, without markers. Use it to check a state against `maxStateTokens` up front.

### `jev.runtime`

The `{ model, family, device, dtype }` that were loaded. `MODELS` maps each alias to its repo id.

### `jev.dispose(): Promise<void>`

Releases the ONNX session. Pending `decide()` calls finish first; the instance cannot be used afterwards.

## How it works

open-jev and kev read the state once and score every option of every question in a single pass. Julia 1 reads the state once per question, and all questions of a call run as one batch. The library builds the model-specific sequence and reads the right logits back.

**open-jev** (DeBERTa-v3-large):

```
[CLS] [STATE] state [Q] instructions [OPT] option_1 [OPT] option_2 … [Q] … [SEP]
```

together with a span-slot tensor (`seg`) and per-pair slot ids (`pair_q`, `pair_opt`). The graph returns one logit per (question, option) pair.

**kev** (Qwen3):

```
<state> state <q> instructions <opt> option_1 </opt> <opt> option_2 </opt> … <decide> <q> …
```

The graph takes only `input_ids` and `attention_mask`, derives a block-causal mask from the delimiters so each question sees the state and itself only, and returns one logit per token. The library reads the value at every option's `</opt>` position. Caller text is escaped (`<|name|>` becomes `<¦name¦>`) so it can never forge a delimiter.

In both cases a temperature-scaled softmax within each question's group is that question's distribution. `noul` questions use the fixed options `["no", "yes"]` the models were trained with.

**julia-1** (mmBERT-small, [Supersonic Labs](https://huggingface.co/SupersonicLabs/Julia-1)):

```
<bos> {type} question: instructions <eos> <mask> option_1 <mask> option_2 … <eos> state <eos>
```

one sequence per question, batched. The graph takes the `<mask>` positions (`marker_pos`, `marker_mask`) and the question type (`qtype`) and returns one logit per option. A described choice is fed as its description alone, and `noul` uses its `descriptions` when given and `["false", "true"]` otherwise, as in Julia's own typed API. Options may be 48 tokens each and the question with its options 512, as in the model's own benchmark. Longer questions or options throw rather than being cut. Text is tokenized one space-free word at a time, because Transformers.js splits runs of spaces differently from the Rust `tokenizers` library the model was trained with; this reproduces the original token ids exactly.

**laya** ([Convai Innovations](https://huggingface.co/convaiinnovations/laya); ModernBERT-large, or mmBERT-base for `laya-multilingual`):

```
[CLS] {type} question: instructions [SEP] [MASK] option_1 [MASK] option_2 … [SEP] state [SEP]
```

one sequence per question, batched, with the same graph inputs as Julia. This follows laya's own `build_sequence`: a described choice is `name: description`, a score level `level i: text`, and `noul` is `false: …` / `true: …` with its `descriptions` or laya's defaults. Options are cut to 48 tokens, and when options and question crowd the head budget (`head_max_len`, 192 or 256) they are shortened as laya does, rather than rejected. Each question's logits are divided by the checkpoint's calibrated temperature for its type and option count (clamped to 0.5–5), as `laya.Agent` does. The multilingual tokenizer is fed one word at a time, like Julia's. Against laya's Python `Agent`, token ids are identical and fp32 probabilities agree within 1e-4 (see the ONNX repos' cards).

**decider** ([Strands Labs](https://github.com/strands-labs/strands-decider); Qwen3.5-2B-Base with a merged LoRA adapter):

```
<state>
state
</state>
<question type="choice">
Select exactly one option.
instructions
<options>
1. name — description
2. name
</options>
</question>
<answer>
```

one prompt per question, batched with right padding (both attention types are causal, so a row never sees its pads). The graph takes the `<answer>` position (`answer_pos`) and the last token of each option's line (`option_pos`) and returns the pointer head's score per option. This follows `strands_decider/prompting.py` and `infer.py`: score levels are numbered from 0 with the level text as description, `noul` lists `false` and `true` with its `descriptions` or the decider's defaults, the question claims the window first (cut from the front if it must), and each question's logits are divided by the checkpoint's calibrated temperature for its type. Option positions come from UTF-8 byte offsets of the byte-level BPE tokens. Against the upstream engine, the same graph gives identical probabilities from Python and from open-jev.

## Development

```bash
pnpm install
pnpm dev
```

`pnpm dev` builds the library once, then runs the library watch build and the example app's Vite dev server side by side. Use `pnpm dev:lib` or `pnpm dev:example` to run only one of them.

Build for publish:

```bash
pnpm build
```

Type-check:

```bash
pnpm typecheck
```

Publish to npm (checks login, bumps the version, type-checks, builds, publishes):

```bash
pnpm release <otp> [patch|minor|major]
```

## Example app

A minimal vanilla demo is included at `examples/simple`.

```bash
cd examples/simple
pnpm install --ignore-workspace
pnpm dev
```

## Notes

- Designed for browser environments; works in Node.js with `device: "cpu"` (or `auto`).
- The models are English only. open-jev was trained on three public domains (banking support, movie reviews, Wikipedia yes/no), the kev models on ten. Questions outside these domains work but are less accurate; measure before relying on them. See the model cards linked above for numbers and limitations.
- Loading a kev model logs one Transformers.js warning ("assuming encoder-only architecture"). It is expected: the graph is a custom pointer-head export, not a text generator.
- Julia 1 is multilingual. Through this library it reproduces the model card's CPU run of its typed-decisions benchmark exactly (426/600 choice, 542/800 score, 483/600 noul) and matches 100/100 predictions of its parity set on WebGPU. Give `noul` questions `descriptions` for Julia, and measure on your own questions: answers can depend on option order.
- Loading Julia 1 requests `config.json` and `onnx/model.onnx` once and gets 404s: Transformers.js probes the standard layout to size its progress bar. Loading is unaffected; open-jev reports progress from the files the export actually has.
- Model weights are Apache-2.0, this library is MIT.
