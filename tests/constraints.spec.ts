import { expect, test } from "@playwright/test";
import {
  compileGenerationConstraint,
  maskConstraintLogits,
  type GenerationConstraint,
} from "../src/runtime/constraints";
import { TokenByteTrie, type TokenByteSource } from "../src/runtime/token-byte-trie";

const encoder = new TextEncoder();

function source(tokens: readonly (string | Uint8Array | null)[]): TokenByteSource {
  return {
    vocabularySize: tokens.length,
    tokenBytes(tokenId) {
      const token = tokens[tokenId];
      return typeof token === "string" ? encoder.encode(token) : token;
    },
  };
}

function acceptText(constraint: GenerationConstraint, text: string): void {
  const compiled = compileGenerationConstraint(constraint);
  compiled.acceptBytes(encoder.encode(text));
  compiled.validateFinal(text);
}

test("prunes a token-byte trie through a full-match regex DFA", () => {
  const trie = new TokenByteTrie(source([null, "a", "b", "c", "ab", "ac", "x"]));
  const constraint = compileGenerationConstraint({ type: "regex", pattern: "a(?:b|c)" });

  expect(constraint.legalTokenIds(trie)).toEqual([1, 4, 5]);
  constraint.acceptToken(encoder.encode("a"));
  expect(constraint.accepting).toBe(false);
  expect(constraint.legalTokenIds(trie)).toEqual([2, 3]);
  constraint.acceptToken(encoder.encode("b"));
  expect(constraint.accepting).toBe(true);
  constraint.validateFinal("ab");
});

test("preserves UTF-8 decoder state across token boundaries", () => {
  const arabic = encoder.encode("أ");
  const trie = new TokenByteTrie(source([
    null,
    arabic.slice(0, 1),
    arabic.slice(1),
    arabic,
    Uint8Array.of(0xff),
  ]));
  const constraint = compileGenerationConstraint({ type: "regex", pattern: "أ" });

  expect(constraint.legalTokenIds(trie)).toEqual([1, 3]);
  constraint.acceptToken(arabic.slice(0, 1));
  expect(constraint.accepting).toBe(false);
  expect(constraint.legalTokenIds(trie)).toEqual([2]);
  constraint.acceptToken(arabic.slice(1));
  expect(constraint.accepting).toBe(true);
});

test("enforces bounded JSON syntax incrementally", () => {
  acceptText(
    { type: "json", maxDepth: 3, whitespace: "none" },
    "{\"x\":[true,null,-1.5e2]}",
  );
  expect(() => {
    const constraint = compileGenerationConstraint({ type: "json", maxDepth: 2 });
    constraint.acceptBytes(encoder.encode("{]"));
  }).toThrow("violates the generation constraint");
});

test("enforces the supported closed JSON Schema subset", () => {
  const schema = {
    type: "object",
    properties: {
      kind: { enum: ["move", "wait"] },
      count: { type: "integer" },
    },
    required: ["kind", "count"],
    additionalProperties: false,
  };
  acceptText(
    { type: "json-schema", schema, whitespace: "none" },
    "{\"kind\":\"move\",\"count\":2}",
  );

  expect(() => acceptText(
    { type: "json-schema", schema, whitespace: "none" },
    "{\"kind\":\"invalid\",\"count\":2}",
  )).toThrow("violates the generation constraint");
});

test("rejects unsupported regular and schema constructs", () => {
  expect(() => compileGenerationConstraint({ type: "regex", pattern: "a(?=b)" })).toThrow(
    "Unsupported generation constraint",
  );
  expect(() => compileGenerationConstraint({ type: "regex", pattern: "(a)\\1" })).toThrow(
    "Unsupported generation constraint",
  );
  expect(() => compileGenerationConstraint({
    type: "json-schema",
    schema: {
      type: "object",
      properties: { required: { type: "boolean" }, optional: { type: "string" } },
      required: ["required"],
      additionalProperties: false,
    },
  })).toThrow("require every declared property");
  expect(() => compileGenerationConstraint({
    type: "json-schema",
    schema: { type: "string", pattern: "[a-z]+" },
  })).toThrow("Unsupported JSON Schema keywords: pattern");
});

test("masks logits to legal and accepting-state termination tokens", () => {
  const masked = maskConstraintLogits([1, 2, 3, 4], [1, 3], [0]);
  expect(Array.from(masked)).toEqual([1, 2, Number.NEGATIVE_INFINITY, 4]);
});
