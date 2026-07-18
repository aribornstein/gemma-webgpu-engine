import {
  GEMMA_VISION_MAX_SOFT_TOKENS,
  type GemmaVisionImageSource,
  type GemmaVisionTokenBudget,
} from "./gemma-vision-input";
import type { GemmaAudioSource } from "./gemma-audio-input";
import type {
  GemmaChatMessage,
  GemmaFunctionTool,
  GemmaGenerationInput,
} from "./gemma-tokenizer";

export interface GemmaConversation {
  readonly messages: readonly GemmaChatMessage[];
  readonly images: readonly GemmaVisionImageSource[];
  readonly audios: readonly GemmaAudioSource[];
  readonly tools: readonly GemmaFunctionTool[];
}

export interface PreparedGemmaConversationTurn {
  readonly input: GemmaGenerationInput;
  readonly userMessage: GemmaChatMessage;
  readonly image?: GemmaVisionImageSource;
  readonly audio?: GemmaAudioSource;
}

export interface PreparedGemmaConversationEdit {
  readonly conversation: GemmaConversation;
  readonly turn: PreparedGemmaConversationTurn;
}

export function createGemmaConversation(
  messages: readonly GemmaChatMessage[] = [],
  images: readonly GemmaVisionImageSource[] = [],
  tools: readonly GemmaFunctionTool[] = [],
  audios: readonly GemmaAudioSource[] = [],
): GemmaConversation {
  return {
    messages: Object.freeze([...messages]),
    images: Object.freeze([...images]),
    audios: Object.freeze([...audios]),
    tools: Object.freeze([...tools]),
  };
}

export function prepareGemmaConversationTurn(
  conversation: GemmaConversation,
  prompt: string,
  image?: GemmaVisionImageSource,
  visionTokenBudget: GemmaVisionTokenBudget = GEMMA_VISION_MAX_SOFT_TOKENS,
  enableThinking = false,
  audio?: GemmaAudioSource,
): PreparedGemmaConversationTurn {
  const content = prompt.trim();
  if (!content) throw new Error("Gemma conversation prompt must not be empty");
  const userMessage: GemmaChatMessage = image || audio
    ? {
        role: "user",
        content: [
          ...(image ? [{ type: "image" } as const] : []),
          ...(audio ? [{ type: "audio" } as const] : []),
          { type: "text", text: content },
        ],
      }
    : { role: "user", content };
  const messages = [...conversation.messages, userMessage];
  const images = image ? [...conversation.images, image] : [...conversation.images];
  const audios = audio ? [...conversation.audios, audio] : [...conversation.audios];
  const tools = [...conversation.tools];
  return {
    input: images.length > 0 || audios.length > 0 || tools.length > 0 || enableThinking
      ? {
          messages,
          ...(images.length > 0 ? { images, visionTokenBudget } : {}),
          ...(audios.length > 0 ? { audios } : {}),
          ...(tools.length > 0 ? { tools } : {}),
          ...(enableThinking ? { enableThinking: true } : {}),
        }
      : messages,
    userMessage,
    image,
    audio,
  };
}

export function commitGemmaConversationTurn(
  conversation: GemmaConversation,
  turn: PreparedGemmaConversationTurn,
  assistantText: string,
  assistantReasoning?: string,
): GemmaConversation {
  const content = assistantText.trim();
  if (!content) throw new Error("Gemma assistant response must not be empty");
  const reasoning = assistantReasoning?.trim();
  return createGemmaConversation(
    [
      ...conversation.messages,
      turn.userMessage,
      { role: "assistant", content, ...(reasoning ? { reasoning } : {}) },
    ],
    turn.image ? [...conversation.images, turn.image] : conversation.images,
    conversation.tools,
    turn.audio ? [...conversation.audios, turn.audio] : conversation.audios,
  );
}

export function prepareGemmaConversationEdit(
  conversation: GemmaConversation,
  messageIndex: number,
  prompt: string,
  replacementImage?: GemmaVisionImageSource,
  visionTokenBudget: GemmaVisionTokenBudget = GEMMA_VISION_MAX_SOFT_TOKENS,
): PreparedGemmaConversationEdit {
  if (!Number.isInteger(messageIndex) || messageIndex < 0 ||
      messageIndex >= conversation.messages.length) {
    throw new Error("Gemma conversation edit index is invalid");
  }
  const message = conversation.messages[messageIndex];
  if (message.role !== "user") throw new Error("Only Gemma user turns can be edited");
  const priorMessages = conversation.messages.slice(0, messageIndex);
  const priorImageCount = countImageParts(priorMessages);
  const messageImageCount = countImageParts([message]);
  if (messageImageCount > 1) {
    throw new Error("Gemma conversation edits support at most one image per user turn");
  }
  const image = replacementImage ?? (messageImageCount === 1
    ? conversation.images[priorImageCount]
    : undefined);
  if (messageImageCount === 1 && !image) {
    throw new Error("Gemma conversation edit is missing its owned image");
  }
  const truncated = createGemmaConversation(
    priorMessages,
    conversation.images.slice(0, priorImageCount),
    conversation.tools,
    conversation.audios,
  );
  return {
    conversation: truncated,
    turn: prepareGemmaConversationTurn(truncated, prompt, image, visionTokenBudget),
  };
}

function countImageParts(messages: readonly GemmaChatMessage[]): number {
  return messages.reduce((total, message) => total + (typeof message.content === "string"
    ? 0
    : message.content.filter((part) => part.type === "image").length), 0);
}