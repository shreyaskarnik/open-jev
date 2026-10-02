import { PreTrainedModel, Tensor } from "@huggingface/transformers";
import type { PreTrainedTokenizer } from "@huggingface/transformers";
import type { FamilyAdapter, JevModel } from "./models";
import type { Question } from "./types";

/**
 * strands-decider by Strands Labs (Qwen3.5-2B-Base, a LoRA adapter and a
 * pointer head).
 *
 * Each question is its own prompt, the state followed by the rendered
 * question:
 *
 *   <state>
 *   {state}
 *   </state>
 *   <question type="choice">
 *   Select exactly one option.
 *   {instructions}
 *   <options>
 *   1. billing — {description}
 *   2. sales
 *   </options>
 *   </question>
 *   <answer>
 *
 * The graph scores option k from the hidden state at the last token of its
 * line against the one at `<answer>`. This mirrors `render_state`,
 * `render_question` and `_option_block` in `strands_decider/prompting.py`,
 * and `_fit` and `_option_token_index` in `strands_decider/infer.py`: the
 * question claims the window first and loses its front if it must, then the
 * state gets the rest.
 */

type ConfigJson = Record<string, unknown>;

const HEADERS = {
  noul: "Decide whether the statement is true of the state.",
  choice: "Select exactly one option.",
  score: "Rate the state against the ordered levels below (lowest first).",
} as const;

/** `NOUL_DEFAULT_CRITERIA` of the original. */
const NOUL_DEFAULTS = {
  false: "the statement does not hold for this state",
  true: "the statement holds for this state",
};

type Rendered = {
  text: string;
  /** UTF-8 byte span of each option's line within `text`. */
  spans: [number, number][];
};

type Prompt = {
  ids: number[];
  answer: number;
  options: number[];
  stateTokens: number;
  stateTruncated: boolean;
};

function num(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

const utf8 = new TextEncoder();
const byteLength = (text: string): number => utf8.encode(text).length;

/** Python's `" ".join(text.split())`. */
const collapse = (text: string | undefined): string =>
  (text ?? "").split(/\s+/).filter(Boolean).join(" ");

/** `render_question` of the original, with byte spans instead of characters. */
function renderQuestion(question: Question): Rendered {
  let pairs: [string, string][];
  if (question.type === "noul") {
    pairs = [
      ["false", question.descriptions?.false ?? NOUL_DEFAULTS.false],
      ["true", question.descriptions?.true ?? NOUL_DEFAULTS.true],
    ];
  } else if (question.type === "choice") {
    const descriptions = (question.descriptions ?? {}) as Record<
      string,
      string | undefined
    >;
    pairs = question.options.map((option) => [
      option,
      descriptions[option] ?? "",
    ]);
  } else {
    // Score levels are numbered from 0; the level text is the description.
    pairs = question.options.map((level, index) => [String(index), level]);
  }

  const prefix = `<question type="${question.type}">\n${HEADERS[question.type]}\n${question.instructions.trim()}\n<options>\n`;
  let text = prefix;
  const spans: [number, number][] = [];
  pairs.forEach(([name, description], index) => {
    const desc = collapse(description);
    const line = `${index + 1}. ${name}${desc ? ` — ${desc}` : ""}`;
    const start = byteLength(text);
    text += line;
    spans.push([start, byteLength(text)]);
    text += index < pairs.length - 1 ? "\n" : "";
  });
  text += "\n</options>\n</question>\n<answer>";
  return { text, spans };
}

export function deciderFamily(section: ConfigJson): FamilyAdapter {
  const byKind = (section.temperature_by_kind ?? {}) as Record<string, unknown>;
  const fallbackTemperature = num(section.temperature, 1);
  const questionFraction = num(section.max_question_fraction, 0.75);

  let tokenizer: PreTrainedTokenizer | null = null;
  let tokenBytes: (id: number) => number = () => 0;
  let pad = 0;

  const encode = (text: string): number[] => {
    const { input_ids } = tokenizer!(text, { add_special_tokens: false }) as {
      input_ids: Tensor;
    };
    return Array.from(input_ids.data as ArrayLike<bigint | number>, Number);
  };

  /**
   * `_option_token_index` of the original: the last token that lies wholly
   * inside each option's line. Offsets are in UTF-8 bytes, accumulated from
   * the tokens, which a byte-level BPE covers exactly.
   */
  const optionTokens = (
    ids: number[],
    cut: number,
    spans: [number, number][],
  ): number[] => {
    const ends: number[] = [];
    let offset = 0;
    for (const id of ids) {
      offset += tokenBytes(id);
      ends.push(offset);
    }
    return spans.map(([a, b]) => {
      let last = -1;
      for (let j = cut; j < ids.length; j += 1) {
        const lo = j === 0 ? 0 : ends[j - 1];
        const hi = ends[j];
        if (hi > lo && lo >= a && hi <= b) {
          last = j - cut;
        }
      }
      if (last < 0) {
        throw new Error(
          "An option has no tokens left; the question was truncated through its options. Raise maxLength.",
        );
      }
      return last;
    });
  };

  return {
    name: "decider",
    // q8: 8-bit block weights (fp16 activations) in `model_quantized.onnx`,
    // within 0.03 of the original; q4f16 is smaller but moves near-ties.
    webgpuDtype: "q8",
    fallbackDtype: "q8",
    dtypes: ["q8", "q4f16"],
    limits: {
      minChoiceOptions: 2,
      maxChoiceOptions: 255,
      minScoreLevels: 2,
      maxScoreLevels: 255,
    },
    defaults: {
      // Calibrated temperatures are applied per question type below; this
      // one scales on top of them.
      temperature: 1,
      maxStateTokens: num(section.max_length, 4096),
      maxLength: num(section.max_length, 4096),
    },
    // Like kev: the base class takes the single-session path and feeds the
    // graph's inputs by name.
    loadModel: async (modelId, options) =>
      (await PreTrainedModel.from_pretrained(modelId, {
        config: options.config,
        dtype: options.dtype,
        device: options.device as "webgpu",
        progress_callback: options.progress_callback,
      })) as unknown as JevModel,
    prepare(loaded) {
      tokenizer = loaded;
      const inner = (
        loaded as unknown as {
          _tokenizer: { id_to_token(id: number): string | undefined };
          _tokenizerJSON: { added_tokens?: { id: number; content: string }[] };
        }
      );
      const added = new Map(
        (inner._tokenizerJSON.added_tokens ?? []).map((t) => [t.id, t.content]),
      );
      const cache = new Map<number, number>();
      tokenBytes = (id) => {
        let size = cache.get(id);
        if (size === undefined) {
          const content = added.get(id);
          // Byte-level BPE: one vocab character per byte.
          size =
            content !== undefined
              ? byteLength(content)
              : [...(inner._tokenizer.id_to_token(id) ?? "")].length;
          cache.set(id, size);
        }
        return size;
      };
      pad = (loaded as unknown as { pad_token_id?: number }).pad_token_id ?? 0;
    },
    escape: (text) => text,
    encode() {
      throw new Error(
        "strands-decider scores each question in its own prompt; use decide().",
      );
    },
    countTokens: (text) => encode(text).length,
    async decide({ model, state, questions, maxLength, maxStateTokens }) {
      if (!tokenizer) {
        throw new Error("Model family not prepared.");
      }
      const rendered = questions.map(renderQuestion);
      const questionIds = rendered.map((r) => encode(r.text));

      // `_fit`: the question gets first claim on the window, cut from the front.
      const longest = Math.max(...questionIds.map((ids) => ids.length));
      const reserve = Math.min(
        longest,
        Math.max(1, Math.floor(maxLength * questionFraction)),
      );
      const stateBudget = Math.min(maxStateTokens, Math.max(1, maxLength - reserve));
      const allState = encode(`<state>\n${state.trim()}\n</state>\n`);
      const stateIds = allState.slice(0, stateBudget);

      const prompts: Prompt[] = rendered.map((r, index) => {
        const full = questionIds[index];
        const cut = Math.max(0, full.length - reserve);
        const options = optionTokens(full, cut, r.spans).map(
          (position) => stateIds.length + position,
        );
        const ids = [...stateIds, ...full.slice(cut)];
        return {
          ids,
          answer: ids.length - 1,
          options,
          stateTokens: stateIds.length,
          stateTruncated: stateIds.length < allState.length,
        };
      });

      // One right-padded batch: both attention types are causal, so the pads
      // never reach a row's real tokens.
      const batch = prompts.length;
      const length = Math.max(...prompts.map((p) => p.ids.length));
      const count = Math.max(...prompts.map((p) => p.options.length));
      const ids = new BigInt64Array(batch * length).fill(BigInt(pad));
      const attention = new BigInt64Array(batch * length);
      const answer = new BigInt64Array(batch);
      const options = new BigInt64Array(batch * count);
      prompts.forEach((prompt, row) => {
        prompt.ids.forEach((id, column) => {
          ids[row * length + column] = BigInt(id);
          attention[row * length + column] = 1n;
        });
        answer[row] = BigInt(prompt.answer);
        prompt.options.forEach((position, column) => {
          options[row * count + column] = BigInt(position);
        });
      });

      const { logits } = await model({
        input_ids: new Tensor("int64", ids, [batch, length]),
        attention_mask: new Tensor("int64", attention, [batch, length]),
        answer_pos: new Tensor("int64", answer, [batch]),
        option_pos: new Tensor("int64", options, [batch, count]),
      });
      const values = Array.from(logits.to("float32").data as ArrayLike<number>);

      return {
        logits: prompts.map((prompt, row) => {
          const t = num(byKind[questions[row].type], fallbackTemperature);
          return values
            .slice(row * count, row * count + prompt.options.length)
            .map((value) => value / t);
        }),
        stateTokens: stateIds.length,
        stateTruncated: prompts.some((p) => p.stateTruncated),
      };
    },
  };
}
