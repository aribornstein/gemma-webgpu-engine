import {
  GEMMA_VISION_MAX_SOFT_TOKENS,
  type GemmaVisionImageSource,
  type GemmaVisionTokenBudget,
} from "./gemma-vision-input";
import type {
  GemmaChatMessage,
  GemmaFunctionTool,
  GemmaGenerationInput,
} from "./gemma-tokenizer";

export interface GemmaConversation {
  readonly messages: readonly GemmaChatMessage[];
  readonly images: readonly GemmaVisionImageSource[];
  readonly tools: readonly GemmaFunctionTool[];
}

export interface PreparedGemmaConversationTurn {
  readonly input: GemmaGenerationInput;
  readonly userMessage: GemmaChatMessage;
  readonly image?: GemmaVisionImageSource;
}

export interface PreparedGemmaConversationEdit {
  readonly conversation: GemmaConversation;
  readonly turn: PreparedGemmaConversationTurn;
}

export function createGemmaConversation(
  messages: readonly GemmaChatMessage[] = [],
  images: readonly GemmaVisionImageSource[] = [],
  tools: readonly GemmaFunctionTool[] = [],
): GemmaConversation {
  return {
    messages: Object.freeze([...messages]),
    images: Object.freeze([...images]),
    tools: Object.freeze([...tools]),
  };
}

export function prepareGemmaConversationTurn(
  conversation: GemmaConversation,
  prompt: string,
  image?: GemmaVisionImageSource,
  visionTokenBudget: GemmaVisionTokenBudget = GEMMA_VISION_MAX_SOFT_TOKENS,
): PreparedGemmaConversationTurn {
  const content = prompt.trim();
  if (!content) throw new Error("Gemma conversation prompt must not be empty");
  const userMessage: GemmaChatMessage = image
    ? {
        role: "user",
        content: [
          { type: "image" },
          { type: "text", text: content },
        ],
      }
    : { role: "user", content };
  const messages = [...conversation.messages, userMessage];
  const images = image ? [...conversation.images, image] : [...conversation.images];
  const tools = [...conversation.tools];
  return {
    input: images.length > 0 || tools.length > 0
      ? {
          messages,
          ...(images.length > 0 ? { images, visionTokenBudget } : {}),
          ...(tools.length > 0 ? { tools } : {}),
        }
      : messages,
    userMessage,
    image,
  };
}

export function commitGemmaConversationTurn(
  conversation: GemmaConversation,
  turn: PreparedGemmaConversationTurn,
  assistantText: string,
): GemmaConversation {
  const content = assistantText.trim();
  if (!content) throw new Error("Gemma assistant response must not be empty");
  return createGemmaConversation(
    [...conversation.messages, turn.userMessage, { role: "assistant", content }],
    turn.image ? [...conversation.images, turn.image] : conversation.images,
    conversation.tools,
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