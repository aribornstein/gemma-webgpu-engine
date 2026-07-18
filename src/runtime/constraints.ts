import Ajv, { type ValidateFunction } from "ajv";
import { visitRegExpAST } from "@eslint-community/regexpp";
import { DFA, JS, NFA } from "refa";
import { TokenByteTrie, type TokenByteTrieNode } from "./token-byte-trie";

const DEFAULT_JSON_DEPTH = 4;
const MAX_JSON_DEPTH = 8;
const MAX_SCHEMA_PROPERTIES = 12;
const JSON_STRING = String.raw`"(?:[^"\\\x00-\x1F]|\\["\\/bfnrt]|\\u[0-9A-Fa-f]{4})*"`;
const JSON_NUMBER = String.raw`-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?`;
const JSON_INTEGER = String.raw`-?(?:0|[1-9][0-9]*)`;

export type GenerationConstraint =
  | { type: "regex"; pattern: string }
  | { type: "json"; maxDepth?: number; whitespace?: JsonWhitespace }
  | {
      type: "json-schema";
      schema: object;
      maxDepth?: number;
      whitespace?: JsonWhitespace;
    };

export type JsonWhitespace = "none" | "compact" | "any";

interface ConstraintState {
  node: DFA.Node;
  pendingUtf8: readonly number[];
}

export class CompiledGenerationConstraint {
  private state: ConstraintState;
  private readonly dfa: DFA;
  private readonly validateOutput: (text: string) => void;
  private readonly viableStates: ReadonlySet<DFA.Node>;
  private readonly legalCache = new WeakMap<
    TokenByteTrie,
    WeakMap<DFA.Node, Map<string, readonly number[]>>
  >();

  constructor(
    dfa: DFA,
    validateOutput: (text: string) => void,
  ) {
    this.dfa = dfa;
    this.validateOutput = validateOutput;
    this.state = { node: dfa.initial, pendingUtf8: [] };
    this.viableStates = findViableStates(dfa);
  }

  get accepting(): boolean {
    return this.state.pendingUtf8.length === 0 && this.dfa.finals.has(this.state.node);
  }

  legalTokenIds(trie: TokenByteTrie): readonly number[] {
    let nodeCache = this.legalCache.get(trie);
    if (!nodeCache) {
      nodeCache = new WeakMap();
      this.legalCache.set(trie, nodeCache);
    }
    let pendingCache = nodeCache.get(this.state.node);
    if (!pendingCache) {
      pendingCache = new Map();
      nodeCache.set(this.state.node, pendingCache);
    }
    const pendingKey = this.state.pendingUtf8.join(",");
    const cached = pendingCache.get(pendingKey);
    if (cached) return cached;

    const legal: number[] = [];
    const stack: Array<{ node: TokenByteTrieNode; state: ConstraintState }> = [
      { node: trie.root, state: this.state },
    ];
    while (stack.length > 0) {
      const current = stack.pop()!;
      if (current.node.tokenIds.length > 0 &&
          isViableConstraintState(current.state, this.viableStates)) {
        legal.push(...current.node.tokenIds);
      }
      for (const [byte, child] of current.node.children) {
        const next = transitionByte(current.state, byte);
        if (next && isViableConstraintState(next, this.viableStates)) {
          stack.push({ node: child, state: next });
        }
      }
    }
    legal.sort((left, right) => left - right);
    const result = Object.freeze(legal);
    pendingCache.set(pendingKey, result);
    return result;
  }

  acceptToken(tokenBytes: Uint8Array): void {
    let next: ConstraintState | null = this.state;
    for (const byte of tokenBytes) {
      next = next && transitionByte(next, byte);
      if (!next || !this.viableStates.has(next.node)) {
        throw new Error("Selected token violates the generation constraint");
      }
    }
    this.state = next;
  }

  acceptBytes(bytes: Uint8Array): void {
    this.acceptToken(bytes);
  }

  validateFinal(text: string): void {
    if (!this.accepting) throw new Error("Generated output ended before the constraint accepted it");
    this.validateOutput(text);
  }
}

export function compileGenerationConstraint(
  constraint: GenerationConstraint,
): CompiledGenerationConstraint {
  if (constraint.type === "regex") return compileRegexConstraint(constraint.pattern);
  const maxDepth = resolveDepth(constraint.maxDepth);
  const whitespace = whitespacePattern(constraint.whitespace ?? "compact");
  if (constraint.type === "json") {
    const source = `${whitespace}${jsonValuePattern(maxDepth, whitespace)}${whitespace}`;
    return compilePattern(source, (text) => {
      JSON.parse(text);
    });
  }

  const source = `${whitespace}${schemaPattern(
    constraint.schema as JsonSchema,
    maxDepth,
    whitespace,
    0,
  )}${whitespace}`;
  const ajv = new Ajv({ allErrors: true, strict: true });
  let validator: ValidateFunction;
  try {
    validator = ajv.compile(constraint.schema);
  } catch (error) {
    throw new Error(`Invalid JSON Schema constraint: ${errorMessage(error)}`);
  }
  return compilePattern(source, (text) => {
    const value = JSON.parse(text);
    if (!validator(value)) {
      const detail = ajv.errorsText(validator.errors, { separator: "; " });
      throw new Error(`Generated JSON does not satisfy its schema: ${detail}`);
    }
  });
}

export function maskConstraintLogits(
  rawLogits: ArrayLike<number>,
  legalTokenIds: readonly number[],
  terminationTokenIds: readonly number[],
): Float32Array {
  const logits = new Float32Array(rawLogits.length);
  logits.fill(Number.NEGATIVE_INFINITY);
  for (const tokenId of legalTokenIds) {
    if (tokenId >= 0 && tokenId < logits.length) logits[tokenId] = rawLogits[tokenId];
  }
  for (const tokenId of terminationTokenIds) {
    if (tokenId >= 0 && tokenId < logits.length) logits[tokenId] = rawLogits[tokenId];
  }
  return logits;
}

function compileRegexConstraint(pattern: string): CompiledGenerationConstraint {
  if (pattern.length === 0) throw new Error("Regex constraint pattern must not be empty");
  return compilePattern(pattern, (text) => {
    const expression = new RegExp(pattern, "u");
    const match = expression.exec(text);
    if (!match || match.index !== 0 || match[0].length !== text.length) {
      throw new Error("Generated output does not match its regex constraint");
    }
  });
}

function compilePattern(
  source: string,
  validateOutput: (text: string) => void,
): CompiledGenerationConstraint {
  try {
    const parser = JS.Parser.fromLiteral({ source, flags: "u" });
    visitRegExpAST(parser.ast.pattern, {
      onAssertionEnter() {
        throw new Error("regex assertions are not supported");
      },
      onBackreferenceEnter() {
        throw new Error("regex backreferences are not supported");
      },
    });
    const { expression, maxCharacter } = parser.parse({
      assertions: "throw",
      backreferences: "throw",
    });
    const dfa = DFA.fromFA(NFA.fromRegex(expression, { maxCharacter }));
    dfa.removeUnreachable();
    dfa.minimize();
    if (dfa.finals.size === 0) throw new Error("constraint accepts no output");
    return new CompiledGenerationConstraint(dfa, validateOutput);
  } catch (error) {
    throw new Error(`Unsupported generation constraint: ${errorMessage(error)}`);
  }
}

function transitionByte(state: ConstraintState, byte: number): ConstraintState | null {
  const pending = state.pendingUtf8;
  if (pending.length === 0) {
    if (byte <= 0x7f) return transitionCharacter(state.node, byte);
    if ((byte >= 0xc2 && byte <= 0xdf) ||
        (byte >= 0xe0 && byte <= 0xef) ||
        (byte >= 0xf0 && byte <= 0xf4)) {
      return { node: state.node, pendingUtf8: [byte] };
    }
    return null;
  }
  if (byte < 0x80 || byte > 0xbf) return null;
  const bytes = [...pending, byte];
  const expectedLength = pending[0] <= 0xdf ? 2 : pending[0] <= 0xef ? 3 : 4;
  if (bytes.length < expectedLength) return { node: state.node, pendingUtf8: bytes };
  if ((bytes[0] === 0xe0 && bytes[1] < 0xa0) ||
      (bytes[0] === 0xed && bytes[1] > 0x9f) ||
      (bytes[0] === 0xf0 && bytes[1] < 0x90) ||
      (bytes[0] === 0xf4 && bytes[1] > 0x8f)) return null;
  let codePoint = bytes[0] & (expectedLength === 2 ? 0x1f : expectedLength === 3 ? 0x0f : 0x07);
  for (let index = 1; index < bytes.length; index += 1) {
    codePoint = codePoint << 6 | bytes[index] & 0x3f;
  }
  return transitionCharacter(state.node, codePoint);
}

function transitionCharacter(node: DFA.Node, character: number): ConstraintState | null {
  const next = node.out.get(character);
  return next ? { node: next, pendingUtf8: [] } : null;
}

function isViableConstraintState(
  state: ConstraintState,
  viableStates: ReadonlySet<DFA.Node>,
): boolean {
  if (state.pendingUtf8.length === 0) return viableStates.has(state.node);
  const completion = pendingUtf8CompletionRange(state.pendingUtf8);
  if (!completion) return false;
  for (const [characters, target] of state.node.out.entries()) {
    if (viableStates.has(target) &&
        characters.min <= completion.max && characters.max >= completion.min) return true;
  }
  return false;
}

function pendingUtf8CompletionRange(
  pending: readonly number[],
): { min: number; max: number } | null {
  const lead = pending[0];
  const length = lead <= 0xdf ? 2 : lead <= 0xef ? 3 : 4;
  if (pending.length === 0 || pending.length >= length) return null;
  const secondMin = lead === 0xe0 ? 0xa0 : lead === 0xf0 ? 0x90 : 0x80;
  const secondMax = lead === 0xed ? 0x9f : lead === 0xf4 ? 0x8f : 0xbf;
  if (pending.length > 1 && (pending[1] < secondMin || pending[1] > secondMax)) return null;
  for (let index = 2; index < pending.length; index += 1) {
    if (pending[index] < 0x80 || pending[index] > 0xbf) return null;
  }
  const minimumBytes = [...pending];
  const maximumBytes = [...pending];
  while (minimumBytes.length < length) {
    const index = minimumBytes.length;
    minimumBytes.push(index === 1 ? secondMin : 0x80);
    maximumBytes.push(index === 1 ? secondMax : 0xbf);
  }
  return {
    min: decodeUtf8CodePoint(minimumBytes),
    max: decodeUtf8CodePoint(maximumBytes),
  };
}

function decodeUtf8CodePoint(bytes: readonly number[]): number {
  let codePoint = bytes[0] & (bytes.length === 2 ? 0x1f : bytes.length === 3 ? 0x0f : 0x07);
  for (let index = 1; index < bytes.length; index += 1) {
    codePoint = codePoint << 6 | bytes[index] & 0x3f;
  }
  return codePoint;
}

function findViableStates(dfa: DFA): ReadonlySet<DFA.Node> {
  const reverse = new Map<DFA.Node, Set<DFA.Node>>();
  for (const node of dfa.nodes()) {
    for (const target of node.out.values()) {
      let predecessors = reverse.get(target);
      if (!predecessors) {
        predecessors = new Set();
        reverse.set(target, predecessors);
      }
      predecessors.add(node);
    }
  }
  const viable = new Set<DFA.Node>(dfa.finals);
  const stack = [...dfa.finals];
  while (stack.length > 0) {
    for (const predecessor of reverse.get(stack.pop()!) ?? []) {
      if (viable.has(predecessor)) continue;
      viable.add(predecessor);
      stack.push(predecessor);
    }
  }
  return viable;
}

function resolveDepth(value: number | undefined): number {
  const depth = value ?? DEFAULT_JSON_DEPTH;
  if (!Number.isInteger(depth) || depth < 1 || depth > MAX_JSON_DEPTH) {
    throw new Error(`JSON constraint maxDepth must be an integer between 1 and ${MAX_JSON_DEPTH}`);
  }
  return depth;
}

function whitespacePattern(mode: JsonWhitespace): string {
  if (mode === "none") return "";
  if (mode === "compact") return " *";
  if (mode === "any") return "[\\x20\\x09\\x0A\\x0D]*";
  throw new Error(`Unsupported JSON whitespace mode: ${String(mode)}`);
}

function jsonValuePattern(depth: number, whitespace: string): string {
  const primitive = `(?:${JSON_STRING}|${JSON_NUMBER}|true|false|null)`;
  if (depth <= 0) return primitive;
  const child = jsonValuePattern(depth - 1, whitespace);
  const array = `\\[${whitespace}(?:${child}(?:${whitespace},${whitespace}${child})*)?${whitespace}\\]`;
  const object = `\\{${whitespace}(?:${JSON_STRING}${whitespace}:${whitespace}${child}(?:${whitespace},${whitespace}${JSON_STRING}${whitespace}:${whitespace}${child})*)?${whitespace}\\}`;
  return `(?:${primitive}|${array}|${object})`;
}

type JsonSchema = Record<string, unknown>;

function schemaPattern(
  schema: JsonSchema,
  maxDepth: number,
  whitespace: string,
  depth: number,
): string {
  if (!schema || typeof schema !== "object" || Array.isArray(schema)) {
    throw new Error("JSON Schema constraint must be an object schema");
  }
  assertSchemaKeywords(schema);
  if ("const" in schema) return exactJsonPattern(schema.const);
  if (Array.isArray(schema.enum)) {
    if (schema.enum.length === 0) throw new Error("JSON Schema enum must not be empty");
    return alternate(schema.enum.map(exactJsonPattern));
  }
  const alternatives = schema.oneOf ?? schema.anyOf;
  if (Array.isArray(alternatives)) {
    if (alternatives.length === 0) throw new Error("JSON Schema alternatives must not be empty");
    return alternate(alternatives.map((item) => schemaPattern(
      item as JsonSchema,
      maxDepth,
      whitespace,
      depth,
    )));
  }
  if (depth > maxDepth) throw new Error(`JSON Schema exceeds maxDepth ${maxDepth}`);
  const types = Array.isArray(schema.type) ? schema.type : [schema.type];
  if (types.length === 0 || types[0] === undefined) {
    throw new Error("JSON Schema constraint requires type, const, enum, oneOf, or anyOf");
  }
  return alternate(types.map((type) => {
    if (type === "string") return JSON_STRING;
    if (type === "number") return JSON_NUMBER;
    if (type === "integer") return JSON_INTEGER;
    if (type === "boolean") return "(?:true|false)";
    if (type === "null") return "null";
    if (type === "array") return schemaArrayPattern(schema, maxDepth, whitespace, depth);
    if (type === "object") return schemaObjectPattern(schema, maxDepth, whitespace, depth);
    throw new Error(`Unsupported JSON Schema type: ${String(type)}`);
  }));
}

function schemaArrayPattern(
  schema: JsonSchema,
  maxDepth: number,
  whitespace: string,
  depth: number,
): string {
  if (!schema.items || typeof schema.items !== "object" || Array.isArray(schema.items)) {
    throw new Error("JSON Schema arrays require one object-valued items schema");
  }
  const minimum = schema.minItems === undefined ? 0 : requireCount(schema.minItems, "minItems");
  const maximum = schema.maxItems === undefined
    ? null
    : requireCount(schema.maxItems, "maxItems");
  if (maximum !== null && maximum < minimum) throw new Error("JSON Schema maxItems is below minItems");
  const item = schemaPattern(schema.items as JsonSchema, maxDepth, whitespace, depth + 1);
  const counts = maximum === null
    ? null
    : Array.from({ length: maximum - minimum + 1 }, (_, index) => minimum + index);
  const body = counts
    ? alternate(counts.map((count) => repeatedItems(item, whitespace, count)))
    : `${repeatedItems(item, whitespace, minimum)}${minimum > 0 ? `(?:${whitespace},${whitespace}${item})*` : `(?:${item}(?:${whitespace},${whitespace}${item})*)?`}`;
  return `\\[${whitespace}${body}${whitespace}\\]`;
}

function schemaObjectPattern(
  schema: JsonSchema,
  maxDepth: number,
  whitespace: string,
  depth: number,
): string {
  if (schema.additionalProperties !== false) {
    throw new Error("JSON Schema objects require additionalProperties: false");
  }
  const properties = schema.properties;
  if (!properties || typeof properties !== "object" || Array.isArray(properties)) {
    throw new Error("JSON Schema objects require a properties object");
  }
  const entries = Object.entries(properties as Record<string, unknown>);
  if (entries.length > MAX_SCHEMA_PROPERTIES) {
    throw new Error(`JSON Schema objects support at most ${MAX_SCHEMA_PROPERTIES} properties`);
  }
  const required = schema.required;
  if (!Array.isArray(required) || required.some((key) => typeof key !== "string")) {
    throw new Error("JSON Schema objects require a string-valued required array");
  }
  const requiredSet = new Set(required);
  if (requiredSet.size !== entries.length || entries.some(([key]) => !requiredSet.has(key))) {
    throw new Error("JSON Schema constrained objects require every declared property");
  }
  const members = entries.map(([key, value]) => {
    const valuePattern = schemaPattern(value as JsonSchema, maxDepth, whitespace, depth + 1);
    return `${exactJsonPattern(key)}${whitespace}:${whitespace}${valuePattern}`;
  });
  return `\\{${whitespace}${members.join(`${whitespace},${whitespace}`)}${whitespace}\\}`;
}

function assertSchemaKeywords(schema: JsonSchema): void {
  const supported = new Set([
    "$schema", "title", "description", "default", "examples",
    "type", "const", "enum", "oneOf", "anyOf",
    "properties", "required", "additionalProperties",
    "items", "minItems", "maxItems",
  ]);
  const unsupported = Object.keys(schema).filter((key) => !supported.has(key));
  if (unsupported.length > 0) {
    throw new Error(`Unsupported JSON Schema keywords: ${unsupported.join(", ")}`);
  }
}

function repeatedItems(item: string, whitespace: string, count: number): string {
  return Array.from({ length: count }, (_, index) =>
    `${index > 0 ? `${whitespace},${whitespace}` : ""}${item}`).join("");
}

function requireCount(value: unknown, name: string): number {
  if (!Number.isInteger(value) || (value as number) < 0 || (value as number) > 64) {
    throw new Error(`JSON Schema ${name} must be an integer between 0 and 64`);
  }
  return value as number;
}

function exactJsonPattern(value: unknown): string {
  const text = JSON.stringify(value);
  if (text === undefined) throw new Error("JSON Schema const/enum value is not JSON-serializable");
  return regexEscape(text);
}

function regexEscape(value: string): string {
  return value.replace(/[\\^$.*+?()[\]{}|/]/g, "\\$&");
}

function alternate(patterns: readonly string[]): string {
  if (patterns.length === 1) return patterns[0];
  return `(?:${patterns.join("|")})`;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
