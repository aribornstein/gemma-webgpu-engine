import { expect, test } from "@playwright/test";
import {
  reusableGemmaPromptPrefixLength,
  sameGemmaMultimodalIdentity,
} from "../src/runtime/gemma-prompt-cache";

test("retains only the token prefix that remains valid after a history edit", () => {
  expect(reusableGemmaPromptPrefixLength([2, 10, 20, 30], [2, 10, 20, 40])).toBe(3);
  expect(reusableGemmaPromptPrefixLength([2, 10, 25, 30], [2, 10, 20, 40])).toBe(2);
  expect(reusableGemmaPromptPrefixLength([3, 10, 20, 30], [2, 10, 20, 40])).toBe(0);
});

test("leaves the final prompt token pending even when history is unchanged", () => {
  expect(reusableGemmaPromptPrefixLength([2, 10, 20], [2, 10, 20])).toBe(2);
  expect(reusableGemmaPromptPrefixLength([2], [2])).toBe(0);
  expect(reusableGemmaPromptPrefixLength([], [2])).toBe(0);
});

test("requires exact ordered multimodal identities before reusing prompt rows", () => {
  expect(sameGemmaMultimodalIdentity([], [])).toBe(true);
  expect(sameGemmaMultimodalIdentity(["image-a"], ["image-a"])).toBe(true);
  expect(sameGemmaMultimodalIdentity(["image-a", "image-b"], ["image-a", "image-b"]))
    .toBe(true);
  expect(sameGemmaMultimodalIdentity(["image-a"], [])).toBe(false);
  expect(sameGemmaMultimodalIdentity(["image-a"], ["image-b"])).toBe(false);
  expect(sameGemmaMultimodalIdentity(["image-b", "image-a"], ["image-a", "image-b"]))
    .toBe(false);
});