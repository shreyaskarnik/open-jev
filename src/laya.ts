import { AutoModel, Tensor } from "@huggingface/transformers";
import type { PreTrainedTokenizer } from "@huggingface/transformers";
import type { FamilyAdapter, JevModel } from "./models";
import type { Question } from "./types";

/**
 * Laya by Convai Innovations (ModernBERT or mmBERT with a decision head).
 *
 * Like Julia, Laya scores each question in its own sequence:
 *
 *   [CLS] {type} question: {question} [SEP] [MASK] {opt 1} … [MASK] {opt n} [SEP] {state} [SEP]
 *
 * and the graph returns one logit per `[MASK]` marker. This mirrors
 * `build_sequence` and `Agent._decode_answers` in `laya/common.py` and
 * `laya/agent.py`: option texts, the 48-token cap per option, the head
 * budget that shortens options and question when they crowd it, and the
 * calibrated temperature per question type and option count.
 */

type ConfigJson = Record<string, unknown>;

const QTYPES = { choice: 0, score: 1, noul: 2 } as const;
const QTYPE_NAMES = ["choice", "score", "noul"] as const;

/** Laya's defaults for `noul` outcomes without a description. */
const NOUL_DEFAULTS = {
  false: "no, the statement does not hold",
  true: "yes, the statement holds",
};

const MAX_OPTION_TOKENS = 48;
/** Laya refuses temperatures outside this range rather than over-sharpen. */
const TEMPERATURE_MIN = 0.5;
const TEMPERATURE_MAX = 5;

type Special = {
  cls: number;
  sep: number;
  mask: number;
  pad: number;
  maskText: string;
  space: number;
};

type Sequence = {
  ids: number[];
  markers: number[];
  qtype: number;
  stateTokens: number;
  stateTruncated: boolean;
};

function num(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function clampTemperature(value: unknown): number {
  const t = typeof value === "number" && Number.isFinite(value) ? value : 1;
  return Math.min(TEMPERATURE_MAX, Math.max(TEMPERATURE_MIN, t));
}

/** `temp_bucket` of the original: question type and option count. */
function bucket(qtype: number, count: number): string {
  const size =
    count <= 2 ? "2" : count <= 5 ? "3-5" : count <= 10 ? "6-10" : "11+";
  return `${QTYPE_NAMES[qtype]}:${size}`;
}

export function layaFamily(section: ConfigJson): FamilyAdapter {
  const headLength = num(section.head_max_len, 192);
  const perType = Array.isArray(section.temperature)
    ? (section.temperature as unknown[]).map(clampTemperature)
    : [1, 1, 1];
  const byOptions: Record<string, number> = {};
  if (
    section.temperature_by_options &&
    typeof section.temperature_by_options === "object"
  ) {
    for (const [key, value] of Object.entries(
      section.temperature_by_options as Record<string, unknown>,
    )) {
      byOptions[key] = clampTemperature(value);
    }
  }
  // mmBERT's Metaspace tokenizer needs Julia's word-at-a-time workaround.
  const splitWords = section.split_words === true;

  let tokenizer: PreTrainedTokenizer | null = null;
  let special: Special | null = null;

  const raw = (text: string): number[] => {
    const { input_ids } = tokenizer!(text, { add_special_tokens: false }) as {
      input_ids: Tensor;
    };
    return Array.from(input_ids.data as ArrayLike<bigint | number>, Number);
  };

  const single = (text: string): number => {
    const ids = raw(text);
    if (ids.length !== 1) {
      throw new Error(`Tokenizer does not know the token ${text}.`);
    }
    return ids[0];
  };

  /**
   * Tokenize like the Rust `tokenizers` library. For byte-level BPE
   * (ModernBERT) Transformers.js already agrees; for Metaspace (mmBERT) it
   * merges runs of spaces differently, so encode one word at a time, as the
   * julia family does.
   */
  const encode = (input: string): number[] => {
    const text = input.split(special!.maskText).join(" ");
    if (text === "") {
      return [];
    }
    if (!splitWords) {
      return raw(text);
    }
    const spaced = text.startsWith(" ") ? text : ` ${text}`;
    const ids: number[] = [];
    for (const piece of spaced.split(/(?= )/)) {
      const word = piece.slice(1);
      if (word === "") {
        ids.push(special!.space);
      } else {
        ids.push(...raw(word));
      }
    }
    return ids;
  };

  /** `render_options` of the original. */
  const optionTexts = (question: Question): string[] => {
    if (question.type === "noul") {
      return [
        `false: ${question.descriptions?.false || NOUL_DEFAULTS.false}`,
        `true: ${question.descriptions?.true || NOUL_DEFAULTS.true}`,
      ];
    }
    if (question.type === "score") {
      return question.options.map((level, index) => `level ${index}: ${level}`);
    }
    const descriptions = (question.descriptions ?? {}) as Record<
      string,
      string | undefined
    >;
    return question.options.map((option) =>
      descriptions[option] ? `${option}: ${descriptions[option]}` : option,
    );
  };

  /** `build_sequence` of the original, plus open-jev's state budget. */
  const sequence = (
    question: Question,
    state: number[],
    maxLength: number,
    maxStateTokens: number,
  ): Sequence => {
    const { cls, sep, mask } = special!;
    let options = optionTexts(question).map((text) => [
      mask,
      ...encode(` ${text}`).slice(0, MAX_OPTION_TOKENS),
    ]);
    const size = () => options.reduce((sum, option) => sum + option.length, 0);
    let budget = headLength - size();
    if (budget < 16) {
      const per = Math.max(4, Math.floor((headLength - 16) / options.length));
      options = options.map((option) => option.slice(0, per));
      budget = headLength - size();
    }
    const head = encode(
      `${question.type} question: ${question.instructions}`,
    ).slice(0, Math.max(8, budget));

    let ids = [cls, ...head, sep];
    let markers: number[] = [];
    for (const option of options) {
      markers.push(ids.length);
      ids.push(...option);
    }
    ids.push(sep);
    const room = Math.max(
      0,
      Math.min(maxStateTokens, maxLength - ids.length - 1),
    );
    const kept = state.slice(0, room);
    ids = [...ids, ...kept, sep].slice(0, maxLength);
    markers = markers.filter((position) => position < maxLength);
    if (markers.length < options.length) {
      throw new Error(
        `"${question.instructions}" and its options do not fit in ${maxLength} tokens; raise maxLength.`,
      );
    }
    return {
      ids,
      markers,
      qtype: QTYPES[question.type],
      stateTokens: kept.length,
      stateTruncated: kept.length < state.length,
    };
  };

  return {
    name: "laya",
    webgpuDtype: "fp16",
    fallbackDtype: "fp32",
    dtypes: ["fp32", "fp16"],
    limits: {
      minChoiceOptions: 2,
      maxChoiceOptions: 64,
      minScoreLevels: 2,
      maxScoreLevels: 20,
    },
    defaults: {
      // Calibrated temperatures are applied per question below; this one
      // scales on top of them.
      temperature: 1,
      // The state gets whatever the question leaves of the sequence.
      maxStateTokens: 8192,
      maxLength: num(section.max_len, 512),
    },
    loadModel: async (modelId, options) =>
      (await AutoModel.from_pretrained(modelId, {
        config: options.config,
        dtype: options.dtype,
        device: options.device as "webgpu",
        progress_callback: options.progress_callback,
      })) as unknown as JevModel,
    prepare(loaded) {
      tokenizer = loaded;
      // `tokenizer_config.json`: ModernBERT uses [CLS]/[SEP], mmBERT <bos>/<eos>.
      const t = ((loaded as unknown as { config?: object }).config ??
        {}) as Record<string, string | undefined>;
      const maskText = t.mask_token ?? "[MASK]";
      special = {
        cls: single(t.cls_token ?? "[CLS]"),
        sep: single(t.sep_token ?? "[SEP]"),
        mask: single(maskText),
        pad: single(t.pad_token ?? "[PAD]"),
        maskText,
        space: raw(" ")[0],
      };
    },
    escape: (text) => text,
    encode() {
      throw new Error(
        "Laya scores each question in its own sequence; use decide().",
      );
    },
    countTokens: (text) => encode(text).length,
    async decide({ model, state, questions, maxLength, maxStateTokens }) {
      if (!special) {
        throw new Error("Model family not prepared.");
      }
      const stateIds = encode(state);
      const items = questions.map((question) =>
        sequence(question, stateIds, maxLength, maxStateTokens),
      );

      const batch = items.length;
      const length = Math.max(...items.map((item) => item.ids.length));
      const count = Math.max(...items.map((item) => item.markers.length));
      const ids = new BigInt64Array(batch * length).fill(BigInt(special.pad));
      const attention = new BigInt64Array(batch * length);
      const positions = new BigInt64Array(batch * count);
      const markerMask = new Uint8Array(batch * count);
      const qtype = new BigInt64Array(batch);
      items.forEach((item, row) => {
        item.ids.forEach((id, column) => {
          ids[row * length + column] = BigInt(id);
          attention[row * length + column] = 1n;
        });
        item.markers.forEach((position, column) => {
          positions[row * count + column] = BigInt(position);
          markerMask[row * count + column] = 1;
        });
        qtype[row] = BigInt(item.qtype);
      });

      const { logits } = await model({
        input_ids: new Tensor("int64", ids, [batch, length]),
        attention_mask: new Tensor("int64", attention, [batch, length]),
        marker_pos: new Tensor("int64", positions, [batch, count]),
        marker_mask: new Tensor("bool", markerMask, [batch, count]),
        qtype: new Tensor("int64", qtype, [batch]),
      });
      const values = Array.from(logits.to("float32").data as ArrayLike<number>);

      return {
        logits: items.map((item, row) => {
          const k = item.markers.length;
          const t =
            byOptions[bucket(item.qtype, k)] ?? perType[item.qtype] ?? 1;
          return values
            .slice(row * count, row * count + k)
            .map((value) => value / t);
        }),
        stateTokens: items.length
          ? Math.min(...items.map((item) => item.stateTokens))
          : 0,
        stateTruncated: items.some((item) => item.stateTruncated),
      };
    },
  };
}
