import { PreTrainedModel, Tensor } from "@huggingface/transformers";
import type { PreTrainedTokenizer } from "@huggingface/transformers";
import type { FamilyAdapter, JevModel } from "./models";
import type { Question } from "./types";

/**
 * Decision 2.0 by vLLM Semantic Router (Qwen3 / Qwen3.5 with a candidate head).
 *
 * Each question is its own prompt:
 *
 *   Context:
 *   {state}
 *
 *   Task type: choice
 *   Question:
 *   {instructions}
 *   Options:
 *   <option>
 *   {"description":"…","key":"billing"}
 *   </option>
 *   …
 *
 *   Select the single option best supported by the context and instructions.
 *   Decision:
 *
 * The prefix, every option and the suffix are tokenized separately and
 * concatenated; the graph scores each option from the hidden state at the last
 * token of its segment against the one at the final token. This mirrors
 * `segments`, `encode` and `question_to_row` in the package's
 * `decision_model.py` and `infer.py`: options in the order given, score levels
 * keyed "0".."L-1" with the level text as description, `noul` as false/true
 * with its descriptions or "No"/"Yes", and no truncation. Per-level score
 * offsets (`score_bias`) are added before the softmax; the temperature is 1.
 */

type ConfigJson = Record<string, unknown>;

/** Python's json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False). */
const canonical = (value: unknown): string => JSON.stringify(sortKeys(value));

function sortKeys(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(sortKeys);
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value as object)
        .sort()
        .map((key) => [key, sortKeys((value as Record<string, unknown>)[key])]),
    );
  }
  return value;
}

function num(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

type Option = { key: string; description: string | null };

function options(question: Question): Option[] {
  if (question.type === "noul") {
    return [
      { key: "false", description: question.descriptions?.false ?? "No" },
      { key: "true", description: question.descriptions?.true ?? "Yes" },
    ];
  }
  if (question.type === "score") {
    return question.options.map((level, index) => ({
      key: String(index),
      description: level,
    }));
  }
  const descriptions = (question.descriptions ?? {}) as Record<
    string,
    string | undefined
  >;
  return question.options.map((option) => ({
    key: option,
    description: descriptions[option] ?? null,
  }));
}

export function decision2Family(section: ConfigJson): FamilyAdapter {
  const maxInput = num(section.max_input_tokens, 8192);
  const scoreBias: Record<string, number[]> = {};
  if (section.score_bias && typeof section.score_bias === "object") {
    for (const [levels, offsets] of Object.entries(
      section.score_bias as Record<string, unknown>,
    )) {
      if (Array.isArray(offsets)) {
        scoreBias[levels] = offsets.map((value) => num(value, 0));
      }
    }
  }

  let tokenizer: PreTrainedTokenizer | null = null;
  let pad = 0;

  const encode = (text: string): number[] => {
    const { input_ids } = tokenizer!(text, { add_special_tokens: false }) as {
      input_ids: Tensor;
    };
    return Array.from(input_ids.data as ArrayLike<bigint | number>, Number);
  };

  /** `encode` of the original: segment by segment, endpoints at each option's end. */
  const prompt = (state: string, question: Question) => {
    const ids = encode(
      `Context:\n${state}\n\nTask type: ${question.type}\nQuestion:\n${question.instructions}\nOptions:`,
    );
    const ends: number[] = [];
    for (const option of options(question)) {
      ids.push(...encode(`\n<option>\n${canonical(option)}\n</option>`));
      ends.push(ids.length - 1);
    }
    ids.push(
      ...encode(
        "\n\nSelect the single option best supported by the context and instructions.\nDecision:",
      ),
    );
    return { ids, ends };
  };

  return {
    name: "decision2",
    webgpuDtype: "q8",
    fallbackDtype: "q8",
    dtypes: Array.isArray(section.dtypes)
      ? (section.dtypes as FamilyAdapter["dtypes"])
      : ["q8"],
    limits: {
      minChoiceOptions: 2,
      maxChoiceOptions: 255,
      minScoreLevels: 2,
      maxScoreLevels: 10,
    },
    defaults: {
      temperature: 1,
      maxStateTokens: maxInput,
      maxLength: maxInput,
    },
    loadModel: async (modelId, options) =>
      (await PreTrainedModel.from_pretrained(modelId, {
        config: options.config,
        dtype: options.dtype,
        device: options.device as "webgpu",
        progress_callback: options.progress_callback,
      })) as unknown as JevModel,
    prepare(loaded) {
      tokenizer = loaded;
      pad = (loaded as unknown as { pad_token_id?: number }).pad_token_id ?? 0;
    },
    escape: (text) => text,
    encode() {
      throw new Error(
        "Decision 2.0 scores each question in its own prompt; use decide().",
      );
    },
    countTokens: (text) => encode(text).length,
    async decide({ model, state, questions, maxLength, maxStateTokens }) {
      if (!tokenizer) {
        throw new Error("Model family not prepared.");
      }
      // The original never truncates; neither do we, beyond an explicit state budget.
      const stateIds = encode(state);
      const kept =
        stateIds.length > maxStateTokens
          ? tokenizer.decode(stateIds.slice(0, maxStateTokens))
          : state;
      const prompts = questions.map((question) => prompt(kept, question));
      for (const p of prompts) {
        if (p.ids.length > maxLength) {
          throw new Error(
            `A question's prompt is ${p.ids.length} tokens; the limit is ${maxLength}. Shorten the state or raise maxLength.`,
          );
        }
      }

      const batch = prompts.length;
      const length = Math.max(...prompts.map((p) => p.ids.length));
      const count = Math.max(...prompts.map((p) => p.ends.length));
      const ids = new BigInt64Array(batch * length).fill(BigInt(pad));
      const attention = new BigInt64Array(batch * length);
      const answer = new BigInt64Array(batch);
      const optionPos = new BigInt64Array(batch * count);
      prompts.forEach((p, row) => {
        p.ids.forEach((id, column) => {
          ids[row * length + column] = BigInt(id);
          attention[row * length + column] = 1n;
        });
        answer[row] = BigInt(p.ids.length - 1);
        p.ends.forEach((position, column) => {
          optionPos[row * count + column] = BigInt(position);
        });
      });

      const { logits } = await model({
        input_ids: new Tensor("int64", ids, [batch, length]),
        attention_mask: new Tensor("int64", attention, [batch, length]),
        answer_pos: new Tensor("int64", answer, [batch]),
        option_pos: new Tensor("int64", optionPos, [batch, count]),
      });
      const values = Array.from(logits.to("float32").data as ArrayLike<number>);

      return {
        logits: prompts.map((p, row) => {
          const z = values.slice(row * count, row * count + p.ends.length);
          const offsets =
            questions[row].type === "score" ? scoreBias[String(z.length)] : undefined;
          return offsets ? z.map((value, i) => value + offsets[i]) : z;
        }),
        stateTokens: Math.min(stateIds.length, maxStateTokens),
        stateTruncated: stateIds.length > maxStateTokens,
      };
    },
  };
}
