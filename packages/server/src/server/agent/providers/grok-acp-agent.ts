import type { Logger } from "pino";
import type { ModelInfo, PromptResponse, SessionNotification } from "@agentclientprotocol/sdk";
import { z } from "zod";

import type {
  AgentPromptInput,
  AgentRunOptions,
  AgentSessionConfig,
  AgentTimelineItem,
  AgentUsage,
} from "../agent-sdk-types.js";
import { ACPAgentSession, type ACPAgentSessionOptions } from "./acp-agent.js";
import { GenericACPAgentClient } from "./generic-acp-agent.js";

export const GROK_SESSION_NOTIFICATION_METHOD = "_x.ai/session_notification";
export const GROK_SESSION_UPDATE_METHOD = "_x.ai/session/update";
export const GROK_REWIND_POINTS_METHOD = "_x.ai/rewind/points";
export const GROK_REWIND_EXECUTE_METHOD = "_x.ai/rewind/execute";

const GrokCompactionUpdateSchema = z
  .object({
    sessionUpdate: z.string(),
    tokens_before: z.number().finite().optional(),
  })
  .passthrough();

const GrokCompactionNotificationSchema = z
  .object({
    sessionId: z.string(),
    update: GrokCompactionUpdateSchema,
  })
  .passthrough();

function compactionItem(
  status: "loading" | "completed",
  trigger: "auto" | "manual" = "auto",
  preTokens?: number,
): Extract<AgentTimelineItem, { type: "compaction" }> {
  return {
    type: "compaction",
    status,
    trigger,
    ...(preTokens !== undefined ? { preTokens } : {}),
  };
}

export function mapGrokCompactionExtensionNotification(
  method: string,
  params: Record<string, unknown>,
  sessionId: string | null,
  trigger: "auto" | "manual" = "auto",
): AgentTimelineItem[] | null {
  if (method !== GROK_SESSION_NOTIFICATION_METHOD && method !== GROK_SESSION_UPDATE_METHOD) {
    return null;
  }

  const parsed = GrokCompactionNotificationSchema.safeParse(params);
  if (!parsed.success || parsed.data.sessionId !== sessionId) {
    return [];
  }

  const update = parsed.data.update;
  const preTokens =
    typeof update.tokens_before === "number" && update.tokens_before > 0
      ? update.tokens_before
      : undefined;
  switch (update.sessionUpdate) {
    case "auto_compact_started":
      return [compactionItem("loading", trigger)];
    case "auto_compact_completed":
    case "auto_compact_failed":
    case "auto_compact_cancelled":
      return [compactionItem("completed", trigger, preTokens)];
    default:
      return [];
  }
}

interface GrokACPAgentClientOptions {
  logger: Logger;
  command: [string, ...string[]];
  env?: Record<string, string>;
  providerId?: string;
  label?: string;
  providerParams?: unknown;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === "object" && !Array.isArray(value);
}

function positiveTokenCount(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return null;
  }
  return value;
}

export function grokOccupancyFromMeta(meta: unknown): number | null {
  if (!isRecord(meta)) {
    return null;
  }
  return positiveTokenCount(meta.totalTokens);
}

export function grokContextWindowFromModels(
  models: ModelInfo[] | null | undefined,
  currentModelId: string | null,
): number | null {
  if (!models || models.length === 0) {
    return null;
  }

  let current: ModelInfo | undefined;
  if (currentModelId === null) {
    current = models.length === 1 ? models[0] : undefined;
  } else {
    current = models.find((model) => model.modelId === currentModelId);
  }
  if (!current) {
    return null;
  }

  const window = positiveTokenCount(current._meta?.totalContextTokens);
  return window !== null && window > 0 ? window : null;
}

function grokCompactPromptText(prompt: AgentPromptInput): string {
  if (typeof prompt === "string") {
    return prompt.trim();
  }
  const parts: string[] = [];
  for (const block of prompt) {
    if (block.type === "text") {
      parts.push(block.text);
    }
  }
  return parts.join("").trim();
}

function isGrokCompactPrompt(prompt: AgentPromptInput): boolean {
  const text = grokCompactPromptText(prompt);
  return text === "/compact" || text.startsWith("/compact ");
}

function isGrokSlashPrompt(text: string): boolean {
  return text.startsWith("/");
}

const GrokRewindPointSchema = z
  .object({
    prompt_index: z.number().int().nonnegative(),
    prompt_preview: z.string().optional(),
  })
  .passthrough();

const GrokRewindPointsResponseSchema = z
  .object({
    rewind_points: z.array(GrokRewindPointSchema),
  })
  .passthrough();

const GrokRewindExecuteResponseSchema = z
  .object({
    success: z.boolean(),
    error: z.string().nullable().optional(),
  })
  .passthrough();

export interface GrokRewindPoint {
  promptIndex: number;
  promptPreview: string | null;
}

export function parseGrokRewindPoints(payload: unknown): GrokRewindPoint[] {
  const parsed = GrokRewindPointsResponseSchema.safeParse(payload);
  if (!parsed.success) {
    throw new Error("Grok rewind points response was invalid");
  }
  return parsed.data.rewind_points.map((point) => ({
    promptIndex: point.prompt_index,
    promptPreview: point.prompt_preview?.trim() ? point.prompt_preview : null,
  }));
}

export interface GrokRewindExecuteResult {
  success: boolean;
  error: string | null;
}

export function parseGrokRewindExecuteResult(payload: unknown): GrokRewindExecuteResult {
  const parsed = GrokRewindExecuteResponseSchema.safeParse(payload);
  if (!parsed.success) {
    throw new Error("Grok rewind execute response was invalid");
  }
  return {
    success: parsed.data.success,
    error: parsed.data.error ?? null,
  };
}

function normalizeGrokPromptText(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

export const GROK_PROMPT_MESSAGE_ID_PREFIX = "grok-prompt-";

export function grokPromptMessageId(promptIndex: number): string {
  return `${GROK_PROMPT_MESSAGE_ID_PREFIX}${promptIndex}`;
}

export function parseGrokPromptMessageId(messageId: string): number | null {
  if (!messageId.startsWith(GROK_PROMPT_MESSAGE_ID_PREFIX)) {
    return null;
  }
  const suffix = messageId.slice(GROK_PROMPT_MESSAGE_ID_PREFIX.length);
  if (!/^\d+$/.test(suffix)) {
    return null;
  }
  return Number(suffix);
}

export function resolveGrokRewindPromptIndex(input: {
  messageId: string;
  userMessages: Array<{ messageId?: string; text: string }>;
  rewindPoints: GrokRewindPoint[];
}): number {
  const messageId = input.messageId.trim();
  if (!messageId) {
    throw new Error("Grok rewind requires a user message id");
  }
  if (input.rewindPoints.length === 0) {
    throw new Error("Grok rewind has no rewind points for this session");
  }

  const stampedIndex = parseGrokPromptMessageId(messageId);
  if (stampedIndex !== null) {
    const byStamp = input.rewindPoints.find((point) => point.promptIndex === stampedIndex);
    if (byStamp) {
      return byStamp.promptIndex;
    }
  }

  const targetIndex = input.userMessages.findIndex((message) => message.messageId === messageId);
  if (targetIndex >= 0) {
    const byIndex = input.rewindPoints.find((point) => point.promptIndex === targetIndex);
    if (byIndex) {
      return byIndex.promptIndex;
    }
  }

  const targetText =
    targetIndex >= 0 ? normalizeGrokPromptText(input.userMessages[targetIndex].text) : null;
  if (targetText) {
    const matchingIndexes = input.rewindPoints
      .filter((point) => {
        if (!point.promptPreview) {
          return false;
        }
        const preview = normalizeGrokPromptText(point.promptPreview);
        return targetText === preview || targetText.startsWith(preview);
      })
      .map((point) => point.promptIndex);
    if (matchingIndexes.length === 1) {
      return matchingIndexes[0];
    }
    if (matchingIndexes.length > 1 && targetIndex >= 0 && matchingIndexes.includes(targetIndex)) {
      return targetIndex;
    }
  }

  throw new Error(`Grok rewind target ${messageId} was not found in rewind points`);
}

export function assignGrokPromptMessageIds(input: {
  userMessages: Array<{ messageId?: string; text: string }>;
  rewindPoints: GrokRewindPoint[];
}): Array<{ messageId: string; text: string }> {
  const used = new Set<number>();
  const assigned: Array<{ messageId: string; text: string }> = [];
  for (const point of input.rewindPoints) {
    const matchIndex = input.userMessages.findIndex((message, index) => {
      if (used.has(index) || !point.promptPreview) {
        return false;
      }
      const text = normalizeGrokPromptText(message.text);
      const preview = normalizeGrokPromptText(point.promptPreview);
      return text === preview || text.startsWith(preview);
    });
    if (matchIndex < 0) {
      continue;
    }
    used.add(matchIndex);
    const message = input.userMessages[matchIndex];
    message.messageId = grokPromptMessageId(point.promptIndex);
    assigned.push({ messageId: message.messageId, text: message.text });
  }
  return assigned;
}

function grokUsageSnapshot(params: {
  occupancy: number;
  models: ModelInfo[] | null | undefined;
  currentModelId: string | null;
}): AgentUsage | undefined {
  const contextWindowMaxTokens = grokContextWindowFromModels(params.models, params.currentModelId);
  if (contextWindowMaxTokens === null) {
    return undefined;
  }
  return {
    contextWindowUsedTokens: params.occupancy,
    contextWindowMaxTokens,
  };
}

// Grok Build reports occupancy on ACP `_meta.totalTokens` and the window on
// the current model's `_meta.totalContextTokens`. It does not emit usage_update.
export class GrokACPAgentSession extends ACPAgentSession {
  private lastUsedTokens: number | null = null;
  private lastMaxTokens: number | null = null;
  private compactTrigger: "auto" | "manual" | null = null;
  private userPrompts: Array<{ messageId: string; text: string }> = [];

  override async initializeNewSession(): Promise<void> {
    await super.initializeNewSession();
    this.userPrompts = [];
  }

  override async initializeResumedSession(): Promise<void> {
    await super.initializeResumedSession();
    await this.captureUserPromptsFromHistory();
  }

  // Grok /compact never emits auto_compact_started. Show the loading stripe
  // from the slash command itself; auto_compact_completed still settles it.
  override async startTurn(
    prompt: AgentPromptInput,
    options?: AgentRunOptions,
  ): Promise<{ turnId: string }> {
    if (isGrokCompactPrompt(prompt)) {
      this.compactTrigger = "manual";
    }
    const promptText = grokCompactPromptText(prompt);
    const submittedMessageId = options?.clientMessageId;
    if (submittedMessageId && promptText.length > 0 && !isGrokSlashPrompt(promptText)) {
      this.userPrompts.push({ messageId: submittedMessageId, text: promptText });
    }
    const result = await super.startTurn(prompt, options);
    if (isGrokCompactPrompt(prompt)) {
      this.pushEvent({
        type: "timeline",
        provider: this.provider,
        item: compactionItem("loading", "manual"),
        turnId: result.turnId,
      });
    }
    return result;
  }

  override async extNotification(method: string, params: Record<string, unknown>): Promise<void> {
    const items = mapGrokCompactionExtensionNotification(
      method,
      params,
      this.id,
      this.compactTrigger ?? "auto",
    );
    if (items) {
      if (items.some((item) => item.type === "compaction" && item.status === "loading")) {
        this.compactTrigger = this.compactTrigger ?? "auto";
      }
      for (const item of items) {
        this.pushEvent(this.wrapTimeline(item));
      }
      if (items.some((item) => item.type === "compaction" && item.status !== "loading")) {
        this.compactTrigger = null;
      }
    }
    await super.extNotification(method, params);
  }

  override async sessionUpdate(params: SessionNotification): Promise<void> {
    await super.sessionUpdate(params);
    if (this.id === null || params.sessionId !== this.id) {
      return;
    }
    this.publishOccupancy(grokOccupancyFromMeta(params._meta));
  }

  protected override handlePromptResponse(response: PromptResponse, turnId: string): void {
    const occupancy = grokOccupancyFromMeta(response._meta);
    if (occupancy !== null) {
      const usage = grokUsageSnapshot({
        occupancy,
        models: this.availableModels,
        currentModelId: this.currentModel,
      });
      if (usage) {
        this.currentTurnUsage = { ...this.currentTurnUsage, ...usage };
        this.lastUsedTokens = usage.contextWindowUsedTokens ?? null;
        this.lastMaxTokens = usage.contextWindowMaxTokens ?? null;
      }
    }
    super.handlePromptResponse(response, turnId);
  }

  private publishOccupancy(occupancy: number | null): void {
    if (occupancy === null) {
      return;
    }
    const usage = grokUsageSnapshot({
      occupancy,
      models: this.availableModels,
      currentModelId: this.currentModel,
    });
    if (!usage) {
      return;
    }

    const used = usage.contextWindowUsedTokens ?? null;
    const max = usage.contextWindowMaxTokens ?? null;
    if (used === this.lastUsedTokens && max === this.lastMaxTokens) {
      return;
    }
    this.lastUsedTokens = used;
    this.lastMaxTokens = max;
    this.pushEvent({
      type: "usage_updated",
      provider: this.provider,
      usage,
    });
  }

  async revertConversation(input: { messageId: string }): Promise<void> {
    if (!this.id) {
      throw new Error("Grok session is not ready for rewind");
    }

    const pointsPayload = await this.callAcpExtensionMethod(GROK_REWIND_POINTS_METHOD, {
      sessionId: this.id,
    });
    const rewindPoints = parseGrokRewindPoints(pointsPayload);
    const userMessages =
      this.userPrompts.length > 0 ? this.userPrompts : this.persistedUserMessages();
    const promptIndex = resolveGrokRewindPromptIndex({
      messageId: input.messageId,
      userMessages,
      rewindPoints,
    });
    // Grok's confirm-before-rewind default returns success:false with no
    // error unless force is set. conversation_only matches Paseo's conversation rewind.
    const executeResult = parseGrokRewindExecuteResult(
      await this.callAcpExtensionMethod(GROK_REWIND_EXECUTE_METHOD, {
        sessionId: this.id,
        targetPromptIndex: promptIndex,
        force: true,
        mode: "conversation_only",
      }),
    );
    if (!executeResult.success) {
      throw new Error(executeResult.error ?? "Grok rewind did not truncate the conversation");
    }
    await this.reloadSessionHistory();
    await this.captureUserPromptsFromHistory();
  }

  private async captureUserPromptsFromHistory(): Promise<void> {
    const sessionId = this.id;
    const messages = this.persistedUserMessages();
    if (!sessionId) {
      this.userPrompts = [];
      return;
    }
    const rewindPoints = parseGrokRewindPoints(
      await this.callAcpExtensionMethod(GROK_REWIND_POINTS_METHOD, { sessionId }),
    );
    this.userPrompts = assignGrokPromptMessageIds({
      userMessages: messages,
      rewindPoints,
    });
  }
}

export class GrokACPAgentClient extends GenericACPAgentClient {
  constructor(options: GrokACPAgentClientOptions) {
    super({
      logger: options.logger,
      command: options.command,
      env: options.env,
      providerId: options.providerId ?? "grok",
      label: options.label ?? "Grok",
      providerParams: options.providerParams,
      capabilities: {
        supportsRewindConversation: true,
      },
    });
  }

  protected override createAgentSession(
    config: AgentSessionConfig,
    options: ACPAgentSessionOptions,
  ): ACPAgentSession {
    return new GrokACPAgentSession(config, options);
  }
}
