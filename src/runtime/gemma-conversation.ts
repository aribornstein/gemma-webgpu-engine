import type { GemmaVisionImageSource } from "./gemma-vision-input";
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
          ...(images.length > 0 ? { images } : {}),
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