import { AutoTokenizer, env, type PreTrainedTokenizer } from "@huggingface/transformers";
import type {
  GemmaVisionImageSource,
  GemmaVisionTokenBudget,
} from "./gemma-vision-input";

const TOKENIZER_PATH = "gemma-4-e2b-tokenizer";
const MODEL_ROOT = `${new URL(import.meta.url).origin}/models/`;
const EOS_TOKEN_IDS = new Set([1, 50, 106]);
export const GEMMA_IMAGE_TOKEN_ID = 258880;

export type GemmaChatRole = "system" | "developer" | "user" | "assistant" | "tool";

export interface GemmaChatTextPart {
  type: "text";
  text: string;
}

export interface GemmaChatImagePart {
  type: "image";
}

export type GemmaChatContentPart = GemmaChatTextPart | GemmaChatImagePart;

export interface GemmaChatToolCall {
  id?: string;
  type: "function";
  function: {
    name: string;
    arguments: Readonly<Record<string, unknown>>;
  };
}

export interface GemmaChatMessage {
  role: GemmaChatRole;
  content: string | readonly GemmaChatContentPart[];
  reasoning?: string;
  toolCalls?: readonly GemmaChatToolCall[];
  toolCallId?: string;
  name?: string;
}

export interface GemmaFunctionTool {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: {
      type: "object";
      properties: Readonly<Record<string, Readonly<Record<string, unknown>>>>;
      required: readonly string[];
    };
    response?: Readonly<Record<string, unknown>>;
  };
}

export interface GemmaStructuredGenerationInput {
  messages: readonly GemmaChatMessage[];
  images?: readonly GemmaVisionImageSource[];
  visionTokenBudget?: GemmaVisionTokenBudget;
  tools?: readonly GemmaFunctionTool[];
  enableThinking?: boolean;
  preserveThinking?: boolean;
}

export interface GemmaMultimodalGenerationInput extends GemmaStructuredGenerationInput {
  images: readonly GemmaVisionImageSource[];
}

export type GemmaGenerationInput =
  | string
  | readonly GemmaChatMessage[]
  | GemmaStructuredGenerationInput;

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
  encodeText(text: string): number[];
  encodeMessages(
    messages: readonly GemmaChatMessage[],
    tools?: readonly GemmaFunctionTool[],
    enableThinking?: boolean,
    preserveThinking?: boolean,
  ): number[];
  encodeInput(input: GemmaGenerationInput): number[];
  decodeTokens(tokenIds: readonly number[]): string;
  decodeRawTokens(tokenIds: readonly number[]): string;
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
  const encodeMessages = (
    messages: readonly GemmaChatMessage[],
    tools: readonly GemmaFunctionTool[] = [],
    enableThinking = false,
    preserveThinking = false,
  ): number[] => {
    validateMessages(messages);
    validateTools(tools);
    const templateOptions = {
      chat_template: chatTemplate,
      tools: tools.map((tool) => ({
        type: tool.type,
        function: {
          ...tool.function,
          parameters: {
            ...tool.function.parameters,
            properties: { ...tool.function.parameters.properties },
            required: [...tool.function.parameters.required],
          },
        },
      })),
      add_generation_prompt: true,
      enable_thinking: enableThinking,
      preserve_thinking: preserveThinking,
      tokenize: true as const,
      return_tensor: false,
      return_dict: false,
    } as Parameters<PreTrainedTokenizer["apply_chat_template"]>[1] & {
      enable_thinking: boolean;
      preserve_thinking: boolean;
    };
    const tokenIds = tokenizer.apply_chat_template(
      messages.map(({ role, content, reasoning, toolCalls, toolCallId, name }) => ({
        role,
        content: typeof content === "string"
          ? content
          : content.map((part) => ({ ...part })),
        ...(reasoning ? { reasoning } : {}),
        ...(toolCalls ? {
          tool_calls: toolCalls.map((call) => ({
            ...call,
            function: { ...call.function, arguments: { ...call.function.arguments } },
          })),
        } : {}),
        ...(toolCallId ? { tool_call_id: toolCallId } : {}),
        ...(name ? { name } : {}),
      })),
      templateOptions,
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
    encodeText(text) {
      return tokenizer.encode(text, { add_special_tokens: false });
    },
    encodeMessages,
    encodeInput(input) {
      if (typeof input === "string") return this.encodePrompt(input);
      return "messages" in input
        ? encodeMessages(
            input.messages,
            input.tools,
            input.enableThinking,
            input.preserveThinking,
          )
        : encodeMessages(input);
    },
    decodeTokens(tokenIds) {
      if (tokenIds.length === 0) return "";
      return tokenizer.decode(Array.from(tokenIds), {
        skip_special_tokens: true,
        clean_up_tokenization_spaces: false,
      });
    },
    decodeRawTokens(tokenIds) {
      if (tokenIds.length === 0) return "";
      return tokenizer.decode(Array.from(tokenIds), {
        skip_special_tokens: false,
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
  const validRoles = new Set<GemmaChatRole>(["system", "developer", "user", "assistant", "tool"]);
  for (const [index, message] of messages.entries()) {
    const hasToolCalls = message.toolCalls !== undefined && message.toolCalls.length > 0;
    if (!validRoles.has(message.role) ||
        (!validContent(message.content) && !(message.role === "assistant" && hasToolCalls))) {
      throw new Error(`Gemma message ${index} has an invalid role or empty content`);
    }
    if (message.reasoning !== undefined &&
        (message.role !== "assistant" || !message.reasoning.trim())) {
      throw new Error(`Gemma message ${index} has invalid reasoning content`);
    }
    if ((message.role === "system" || message.role === "developer") && index !== 0) {
      throw new Error("Gemma system or developer message must be first");
    }
    if (message.toolCalls !== undefined) {
      if (message.role !== "assistant" || !hasToolCalls ||
          message.toolCalls.some((call) => call.type !== "function" ||
            !/^[A-Za-z_][A-Za-z0-9_]*$/.test(call.function.name) ||
            !isRecord(call.function.arguments))) {
        throw new Error(`Gemma message ${index} has invalid tool calls`);
      }
    }
    if (message.role === "tool") {
      if (!message.toolCallId?.trim() && !message.name?.trim()) {
        throw new Error(`Gemma tool message ${index} must identify its tool call`);
      }
    } else if (message.toolCallId !== undefined || message.name !== undefined) {
      throw new Error(`Gemma message ${index} has tool response fields on a non-tool role`);
    }
  }
  if (!new Set<GemmaChatRole>(["user", "tool"]).has(messages.at(-1)?.role as GemmaChatRole)) {
    throw new Error("Gemma generation messages must end with a user or tool turn");
  }
}

function validContent(content: GemmaChatMessage["content"]): boolean {
  if (typeof content === "string") return content.trim().length > 0;
  if (content.length === 0) return false;
  return content.every((part) => part.type === "image" ||
    (part.type === "text" && part.text.trim().length > 0));
}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function validateTools(tools: readonly GemmaFunctionTool[]): void {
  const names = new Set<string>();
  for (const [index, tool] of tools.entries()) {
    const declaration = tool.function;
    if (tool.type !== "function" || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(declaration.name) ||
        !declaration.description.trim() || declaration.parameters.type !== "object") {
      throw new Error(`Gemma tool ${index} has an invalid function declaration`);
    }
    if (names.has(declaration.name)) {
      throw new Error(`Gemma tool name ${declaration.name} is duplicated`);
    }
    names.add(declaration.name);
    for (const required of declaration.parameters.required) {
      if (!(required in declaration.parameters.properties)) {
        throw new Error(`Gemma tool ${declaration.name} requires unknown property ${required}`);
      }
    }
  }
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