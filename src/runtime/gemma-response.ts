export interface GemmaParsedToolCall {
  readonly type: "function";
  readonly function: {
    readonly name: string;
    readonly arguments: Readonly<Record<string, unknown>>;
  };
}

export interface GemmaParsedResponse {
  readonly reasoning: string;
  readonly text: string;
}

const TOOL_CALL_PATTERN = /<\|tool_call>call:([A-Za-z_][A-Za-z0-9_]*)\{([\s\S]*?)\}<tool_call\|>/g;
const THOUGHT_CHANNEL_PATTERN = /<\|channel>thought\n?([\s\S]*?)<channel\|>/g;
const RESPONSE_CONTROL_PATTERN = /<\|turn>(?:model)?\n?|<turn\|>|<\|channel>final\n?/g;

export function parseGemmaResponse(rawText: string, decodedText: string): GemmaParsedResponse {
  const reasoning = Array.from(rawText.matchAll(THOUGHT_CHANNEL_PATTERN), (match) => match[1].trim())
    .filter(Boolean)
    .join("\n\n");
  if (!rawText.includes("<|channel>thought")) {
    return Object.freeze({ reasoning: "", text: decodedText.trim() });
  }
  const withoutThought = rawText.replace(THOUGHT_CHANNEL_PATTERN, "");
  if (withoutThought.includes("<|channel>thought")) {
    throw new Error("Gemma emitted an unterminated thought channel");
  }
  if (withoutThought.includes("<channel|>")) {
    throw new Error("Gemma emitted a malformed thought channel");
  }
  const text = withoutThought.replace(RESPONSE_CONTROL_PATTERN, "")
    .trim();
  return Object.freeze({ reasoning, text });
}

export function countGemmaReasoningTokens(rawTokenFragments: readonly string[]): number {
  let inThought = false;
  let count = 0;
  for (const fragment of rawTokenFragments) {
    if (fragment === "<|channel>thought") {
      inThought = true;
      continue;
    }
    if (fragment === "<channel|>") {
      inThought = false;
      continue;
    }
    if (inThought) count += 1;
  }
  return count;
}

export function parseGemmaToolCalls(rawText: string): readonly GemmaParsedToolCall[] {
  const calls: GemmaParsedToolCall[] = [];
  for (const match of rawText.matchAll(TOOL_CALL_PATTERN)) {
    calls.push(Object.freeze({
      type: "function",
      function: Object.freeze({
        name: match[1],
        arguments: Object.freeze(parseGemmaArguments(`{${match[2]}}`)),
      }),
    }));
  }
  if (rawText.includes("<|tool_call>") && calls.length === 0) {
    throw new Error("Gemma emitted a malformed tool call");
  }
  return Object.freeze(calls);
}

function parseGemmaArguments(source: string): Record<string, unknown> {
  const parser = new GemmaArgumentParser(source);
  const value = parser.parseValue();
  parser.finish();
  if (!isRecord(value)) throw new Error("Gemma tool arguments must be an object");
  return value;
}

class GemmaArgumentParser {
  private position = 0;
  private readonly source: string;

  constructor(source: string) {
    this.source = source;
  }

  parseValue(): unknown {
    this.skipWhitespace();
    if (this.source.startsWith('<|"|>', this.position)) return this.parseString();
    const character = this.source[this.position];
    if (character === "{") return this.parseObject();
    if (character === "[") return this.parseArray();
    if (this.consumeKeyword("true")) return true;
    if (this.consumeKeyword("false")) return false;
    if (this.consumeKeyword("null")) return null;
    const number = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/.exec(
      this.source.slice(this.position),
    )?.[0];
    if (!number) throw new Error(`Invalid Gemma tool value at offset ${this.position}`);
    this.position += number.length;
    return Number(number);
  }

  finish(): void {
    this.skipWhitespace();
    if (this.position !== this.source.length) {
      throw new Error(`Unexpected Gemma tool argument at offset ${this.position}`);
    }
  }

  private parseString(): string {
    const delimiter = '<|"|>';
    this.position += delimiter.length;
    const end = this.source.indexOf(delimiter, this.position);
    if (end < 0) throw new Error("Unterminated Gemma tool string");
    const value = this.source.slice(this.position, end);
    this.position = end + delimiter.length;
    return value;
  }

  private parseObject(): Record<string, unknown> {
    this.expect("{");
    const result: Record<string, unknown> = {};
    this.skipWhitespace();
    while (this.source[this.position] !== "}") {
      const key = this.parseKey();
      this.expect(":");
      result[key] = this.parseValue();
      this.skipWhitespace();
      if (this.source[this.position] !== ",") break;
      this.position += 1;
      this.skipWhitespace();
    }
    this.expect("}");
    return result;
  }

  private parseArray(): unknown[] {
    this.expect("[");
    const result: unknown[] = [];
    this.skipWhitespace();
    while (this.source[this.position] !== "]") {
      result.push(this.parseValue());
      this.skipWhitespace();
      if (this.source[this.position] !== ",") break;
      this.position += 1;
      this.skipWhitespace();
    }
    this.expect("]");
    return result;
  }

  private parseKey(): string {
    this.skipWhitespace();
    const start = this.position;
    while (![":", ",", "}", ""].includes(this.source[this.position] ?? "")) {
      this.position += 1;
    }
    const key = this.source.slice(start, this.position).trim();
    if (!key) throw new Error(`Missing Gemma tool argument key at offset ${start}`);
    return key;
  }

  private consumeKeyword(keyword: string): boolean {
    if (!this.source.startsWith(keyword, this.position)) return false;
    this.position += keyword.length;
    return true;
  }

  private expect(character: string): void {
    this.skipWhitespace();
    if (this.source[this.position] !== character) {
      throw new Error(`Expected ${character} in Gemma tool arguments at offset ${this.position}`);
    }
    this.position += 1;
  }

  private skipWhitespace(): void {
    while (/\s/.test(this.source[this.position] ?? "")) this.position += 1;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}