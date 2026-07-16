import { AutoTokenizer, env, type PreTrainedTokenizer } from "@huggingface/transformers";
import type { GemmaVisionImageSource } from "./gemma-vision-input";

const TOKENIZER_PATH = "gemma-4-e2b-tokenizer";
const MODEL_ROOT = `${new URL(import.meta.url).origin}/models/`;
const EOS_TOKEN_IDS = new Set([1, 50, 106]);
export const GEMMA_IMAGE_TOKEN_ID = 258880;

export type GemmaChatRole = "system" | "developer" | "user" | "assistant";

export interface GemmaChatTextPart {
  type: "text";
  text: string;
}

export interface GemmaChatImagePart {
  type: "image";
}

export type GemmaChatContentPart = GemmaChatTextPart | GemmaChatImagePart;

export interface GemmaChatMessage {
  role: GemmaChatRole;
  content: string | readonly GemmaChatContentPart[];
}

export interface GemmaMultimodalGenerationInput {
  messages: readonly GemmaChatMessage[];
  images: readonly GemmaVisionImageSource[];
}

export type GemmaGenerationInput =
  | string
  | readonly GemmaChatMessage[]
  | GemmaMultimodalGenerationInput;

env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = "/models/";
env.useBrowserCache = false;
const fetchFromEngineOrigin = globalThis.fetch.bind(globalThis);
env.fetch = (input, init) => {
  const resolved = typeof input === "string" && input.startsWith("/models/")
    ? new URL(input, MODEL_ROOT).href
    : input;
  return fetchFromEngineOrigin(resolved, init);
};

export interface GemmaTokenizer {
  readonly vocabularySize: number;
  readonly endTokenIds: readonly number[];
  readonly imageTokenId: number;
  encodePrompt(prompt: string): number[];
  encodeMessages(messages: readonly GemmaChatMessage[]): number[];
  encodeInput(input: GemmaGenerationInput): number[];
  decodeTokens(tokenIds: readonly number[]): string;
  tokenBytes(tokenId: number): Uint8Array | null;
  isEndToken(tokenId: number): boolean;
}

export async function loadGemmaTokenizer(): Promise<GemmaTokenizer> {
  const [tokenizer, templateResponse] = await Promise.all([
    AutoTokenizer.from_pretrained(TOKENIZER_PATH, { local_files_only: true }),
    fetch(new URL(`${TOKENIZER_PATH}/chat_template.jinja`, MODEL_ROOT)),
  ]);
  if (!templateResponse.ok) throw new Error("Pinned Gemma chat template is unavailable");
  return createGemmaTokenizer(tokenizer, await templateResponse.text());
}

function createGemmaTokenizer(
  tokenizer: PreTrainedTokenizer,
  chatTemplate: string,
): GemmaTokenizer {
  const tokenBytes = createTokenByteTable(tokenizer);
  const encodeMessages = (messages: readonly GemmaChatMessage[]): number[] => {
    validateMessages(messages);
    const tokenIds = tokenizer.apply_chat_template(
      messages.map(({ role, content }) => ({
        role,
        content: typeof content === "string"
          ? content
          : content.map((part) => ({ ...part })),
      })),
      {
        chat_template: chatTemplate,
        add_generation_prompt: true,
        tokenize: true,
        return_tensor: false,
        return_dict: false,
      },
    );
    if (!Array.isArray(tokenIds) || tokenIds.some((value) => !Number.isInteger(value))) {
      throw new Error("Gemma tokenizer returned invalid prompt IDs");
    }
    return tokenIds as number[];
  };
  return {
    vocabularySize: tokenBytes.length,
    endTokenIds: Object.freeze([...EOS_TOKEN_IDS]),
    imageTokenId: GEMMA_IMAGE_TOKEN_ID,
    encodePrompt(prompt) {
      return encodeMessages([{ role: "user", content: prompt }]);
    },
    encodeMessages,
    encodeInput(input) {
      if (typeof input === "string") return this.encodePrompt(input);
      return encodeMessages("messages" in input ? input.messages : input);
    },
    decodeTokens(tokenIds) {
      if (tokenIds.length === 0) return "";
      return tokenizer.decode(Array.from(tokenIds), {
        skip_special_tokens: true,
        clean_up_tokenization_spaces: false,
      });
    },
    tokenBytes(tokenId) {
      if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= tokenBytes.length) {
        throw new Error(`Gemma token ID ${tokenId} is outside the vocabulary`);
      }
      return tokenBytes[tokenId];
    },
    isEndToken(tokenId) {
      return EOS_TOKEN_IDS.has(tokenId);
    },
  };
}

function validateMessages(messages: readonly GemmaChatMessage[]): void {
  if (messages.length === 0) throw new Error("Gemma messages must not be empty");
  const validRoles = new Set<GemmaChatRole>(["system", "developer", "user", "assistant"]);
  for (const [index, message] of messages.entries()) {
    if (!validRoles.has(message.role) || !validContent(message.content)) {
      throw new Error(`Gemma message ${index} has an invalid role or empty content`);
    }
    if ((message.role === "system" || message.role === "developer") && index !== 0) {
      throw new Error("Gemma system or developer message must be first");
    }
  }
  if (messages.at(-1)?.role !== "user") {
    throw new Error("Gemma generation messages must end with a user turn");
  }
}

function validContent(content: GemmaChatMessage["content"]): boolean {
  if (typeof content === "string") return content.trim().length > 0;
  if (content.length === 0) return false;
  return content.every((part) => part.type === "image" ||
    (part.type === "text" && part.text.trim().length > 0));
}

function createTokenByteTable(tokenizer: PreTrainedTokenizer): (Uint8Array | null)[] {
  const vocabulary = tokenizer.get_vocab();
  let vocabularySize = 0;
  for (const tokenId of vocabulary.values()) vocabularySize = Math.max(vocabularySize, tokenId + 1);
  const bytes: (Uint8Array | null)[] = Array.from({ length: vocabularySize }, () => null);
  const specialIds = new Set(tokenizer.all_special_ids);
  const encoder = new TextEncoder();
  for (const [token, tokenId] of vocabulary) {
    if (specialIds.has(tokenId)) continue;
    const byteFallback = /^<0x([0-9A-F]{2})>$/.exec(token);
    bytes[tokenId] = byteFallback
      ? Uint8Array.of(Number.parseInt(byteFallback[1], 16))
      : encoder.encode(token.replaceAll("▁", " "));
  }
  return bytes;
}