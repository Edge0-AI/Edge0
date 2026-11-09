import type { Message } from './chat';
export type GenerationPhase = 'idle' | 'generating' | 'stopping';
export interface ConversationState { messages: Message[]; phase: GenerationPhase; }
export type ConversationAction =
  | { type: 'reset' }
  | { type: 'start'; user: string }
  | { type: 'fragment'; message: Message }
  | { type: 'reasoning_expanded'; index: number; expanded: boolean }
  | { type: 'stop' }
  | { type: 'finish'; failed?: boolean };
export const emptyConversation: ConversationState = { messages: [], phase: 'idle' };
export function conversationReducer(state: ConversationState, action: ConversationAction): ConversationState {
  switch (action.type) {
    case 'reset': return emptyConversation;
    case 'start':
      if (state.phase !== 'idle') return state;
      return { messages: [...state.messages, { role: 'user', content: action.user }, { role: 'assistant', content: '' }], phase: 'generating' };
    case 'fragment':
      if (state.phase !== 'generating') return state;
      return { ...state, messages: [...state.messages.slice(0, -1), { ...action.message, reasoningExpanded: state.messages.at(-1)?.reasoningExpanded }] };
    case 'reasoning_expanded':
      return { ...state, messages: state.messages.map((message, index) => index === action.index && message.role === 'assistant' ? { ...message, reasoningExpanded: action.expanded } : message) };
    case 'stop': return state.phase === 'generating' ? { ...state, phase: 'stopping' } : state;
    case 'finish': {
      if (state.phase === 'idle') return state;
      const stopped = state.phase === 'stopping';
      return { phase: 'idle', messages: state.messages.map((message, index) => index === state.messages.length - 1 ? {
        ...message, stopped, ...((stopped || action.failed) ? { metrics: undefined } : {}),
      } : message) };
    }
  }
}
// Return primitive effect dependencies; display metadata never moves the viewport.
export function scrollContent(messages: Message[]): readonly [number, string, string] {
  const last = messages.at(-1);
  return [messages.length - 1, last?.content ?? '', last?.reasoning_content ?? ''];
}
export function canChangeThinking(supported: boolean, conversation: ConversationState): boolean {
  return supported && conversation.phase === 'idle' && conversation.messages.length === 0;
}
export type PortValidation = { valid: true; port: number | null } | { valid: false; error: string };
export function validateServerPort(input: string): PortValidation {
  const value = input.trim();
  if (!value) return { valid: true, port: null };
  if (!/^\d+$/.test(value) || !Number.isSafeInteger(Number(value)) || Number(value) < 1 || Number(value) > 65535) {
    return { valid: false, error: 'Server port must be blank or an integer from 1 to 65535.' };
  }
  return { valid: true, port: Number(value) };
}
