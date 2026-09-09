import { describe, expect, test, vi } from "vitest";
import type { ModelInfo, PromptResponse, SessionNotification } from "@agentclientprotocol/sdk";

import { asInternals } from "../../test-utils/class-mocks.js";
import { createTestLogger } from "../../../test-utils/test-logger.js";
import type { AgentStreamEvent, AgentUsage } from "../agent-sdk-types.js";
import {
  GROK_REWIND_EXECUTE_METHOD,
  GROK_REWIND_POINTS_METHOD,
  GROK_SESSION_NOTIFICATION_METHOD,
  GROK_SESSION_UPDATE_METHOD,
  GrokACPAgentClient,
  GrokACPAgentSession,
  assignGrokPromptMessageIds,
  grokContextWindowFromModels,
  grokOccupancyFromMeta,
  grokPromptMessageId,
  mapGrokCompactionExtensionNotification,
  parseGrokRewindExecuteResult,
  parseGrokRewindPoints,
  resolveGrokRewindPromptIndex,
} from "./grok-acp-agent.js";

const GROK_CAPABILITIES = {
  supportsStreaming: true,
  supportsSessionPersistence: true,
  supportsDynamicModes: true,
  supportsMcpServers: true,
  supportsReasoningStream: true,
  supportsToolInvocations: true,
  supportsRewindConversation: false,
  supportsRewindFiles: false,
  supportsRewindBoth: false,
};

function grokModel(modelId: string, totalContextTokens: number | undefined): ModelInfo {
  return {
    modelId,
    name: modelId,
    _meta: totalContextTokens === undefined ? undefined : { totalContextTokens },
  };
}

function createGrokSession(): GrokACPAgentSession {
  return new GrokACPAgentSession(
    {
      provider: "grok",
      cwd: "/tmp/paseo-grok-test",
    },
    {
      provider: "grok",
      logger: createTestLogger(),
      defaultCommand: ["grok", "agent", "stdio"],
      defaultModes: [],
      capabilities: GROK_CAPABILITIES,
    },
  );
}

interface GrokSessionInternals {
  sessionId: string | null;
  connection: { prompt: () => Promise<PromptResponse> } | null;
  availableModels: ModelInfo[] | null;
  currentModel: string | null;
  currentTurnUsage: AgentUsage | undefined;
  handlePromptResponse(response: PromptResponse, turnId: string): void;
}

function readyGrokSession(models: ModelInfo[], currentModelId: string): GrokACPAgentSession {
  const session = createGrokSession();
  const internals = asInternals<GrokSessionInternals>(session);
  internals.sessionId = "session-1";
  internals.connection = {
    prompt: async () => ({ stopReason: "end_turn" }),
  };
  internals.availableModels = models;
  internals.currentModel = currentModelId;
  return session;
}

describe("grokOccupancyFromMeta", () => {
  test("reads a finite non-negative totalTokens value", () => {
    expect(grokOccupancyFromMeta({ totalTokens: 12332 })).toBe(12332);
    expect(grokOccupancyFromMeta({ totalTokens: 0 })).toBe(0);
  });

  test("ignores missing or invalid occupancy", () => {
    expect(grokOccupancyFromMeta(undefined)).toBeNull();
    expect(grokOccupancyFromMeta({ totalTokens: -1 })).toBeNull();
    expect(grokOccupancyFromMeta({ totalTokens: Number.NaN })).toBeNull();
    expect(grokOccupancyFromMeta({ totalTokens: "12332" })).toBeNull();
  });
});

describe("grokContextWindowFromModels", () => {
  test("reads totalContextTokens from the current model", () => {
    expect(
      grokContextWindowFromModels(
        [grokModel("grok-4.6", 200000), grokModel("or-grok", 500000)],
        "or-grok",
      ),
    ).toBe(500000);
  });

  test("does not fall back to another model's window", () => {
    expect(
      grokContextWindowFromModels(
        [grokModel("grok-4.6", 200000), grokModel("or-grok", undefined)],
        "or-grok",
      ),
    ).toBeNull();
  });

  test("uses the sole advertised model when no current model is selected", () => {
    expect(grokContextWindowFromModels([grokModel("or-grok", 500000)], null)).toBe(500000);
  });
});

describe("Grok ACP session occupancy", () => {
  test("emits usage_updated from session/update _meta.totalTokens and the current model window", async () => {
    const session = readyGrokSession(
      [grokModel("grok-4.6", 200000), grokModel("or-grok", 500000)],
      "or-grok",
    );
    const events: AgentStreamEvent[] = [];
    session.subscribe((event) => {
      events.push(event);
    });

    const notification: SessionNotification = {
      sessionId: "session-1",
      update: {
        sessionUpdate: "agent_message_chunk",
        content: { type: "text", text: "pong" },
      },
      _meta: { totalTokens: 1747 },
    };
    await session.sessionUpdate(notification);

    expect(events.filter((event) => event.type === "usage_updated")).toEqual([
      {
        type: "usage_updated",
        provider: "grok",
        usage: {
          contextWindowUsedTokens: 1747,
          contextWindowMaxTokens: 500000,
        },
      },
    ]);
  });

  test("does not emit a meter snapshot when the current model has no window", async () => {
    const session = readyGrokSession([grokModel("or-grok", undefined)], "or-grok");
    const events: AgentStreamEvent[] = [];
    session.subscribe((event) => {
      events.push(event);
    });

    await session.sessionUpdate({
      sessionId: "session-1",
      update: {
        sessionUpdate: "agent_message_chunk",
        content: { type: "text", text: "pong" },
      },
      _meta: { totalTokens: 1747 },
    });

    expect(events.filter((event) => event.type === "usage_updated")).toEqual([]);
  });

  test("does not re-emit identical Grok occupancy", async () => {
    const session = readyGrokSession([grokModel("or-grok", 500000)], "or-grok");
    const events: AgentStreamEvent[] = [];
    session.subscribe((event) => {
      events.push(event);
    });

    const notification: SessionNotification = {
      sessionId: "session-1",
      update: {
        sessionUpdate: "agent_message_chunk",
        content: { type: "text", text: "pong" },
      },
      _meta: { totalTokens: 1747 },
    };
    await session.sessionUpdate(notification);
    await session.sessionUpdate(notification);

    expect(events.filter((event) => event.type === "usage_updated")).toHaveLength(1);
  });

  test("keeps prompt-response occupancy on turn_completed", () => {
    const session = readyGrokSession([grokModel("or-grok", 500000)], "or-grok");
    const events: AgentStreamEvent[] = [];
    session.subscribe((event) => {
      events.push(event);
    });

    asInternals<GrokSessionInternals>(session).handlePromptResponse(
      {
        stopReason: "end_turn",
        _meta: { totalTokens: 12332 },
      },
      "turn-1",
    );

    expect(events.filter((event) => event.type === "turn_completed")).toEqual([
      {
        type: "turn_completed",
        provider: "grok",
        turnId: "turn-1",
        usage: {
          contextWindowUsedTokens: 12332,
          contextWindowMaxTokens: 500000,
        },
      },
    ]);
  });
});

const compactionSessionId = "session-1";

describe("mapGrokCompactionExtensionNotification", () => {
  test("ignores extension methods and sessions it does not own", () => {
    expect(
      mapGrokCompactionExtensionNotification(
        "_other.ai/session/update",
        { sessionId: "session-1" },
        compactionSessionId,
      ),
    ).toBeNull();
    expect(
      mapGrokCompactionExtensionNotification(
        GROK_SESSION_NOTIFICATION_METHOD,
        {
          sessionId: "session-2",
          update: { sessionUpdate: "auto_compact_started" },
        },
        compactionSessionId,
      ),
    ).toEqual([]);
  });

  test("maps Grok auto-compaction start and completion into the shared marker", () => {
    expect(
      mapGrokCompactionExtensionNotification(
        GROK_SESSION_NOTIFICATION_METHOD,
        {
          sessionId: "session-1",
          update: {
            sessionUpdate: "auto_compact_started",
            tokens_used: 423_901,
          },
        },
        compactionSessionId,
      ),
    ).toEqual([
      {
        type: "compaction",
        status: "loading",
        trigger: "auto",
      },
    ]);

    expect(
      mapGrokCompactionExtensionNotification(
        GROK_SESSION_NOTIFICATION_METHOD,
        {
          sessionId: "session-1",
          update: {
            sessionUpdate: "auto_compact_completed",
            tokens_before: 12304,
            tokens_after: 12304,
            summary_preview: null,
          },
        },
        compactionSessionId,
      ),
    ).toEqual([
      {
        type: "compaction",
        status: "completed",
        trigger: "auto",
        preTokens: 12304,
      },
    ]);
  });

  test("accepts the older _x.ai/session/update method name", () => {
    expect(
      mapGrokCompactionExtensionNotification(
        GROK_SESSION_UPDATE_METHOD,
        {
          sessionId: "session-1",
          update: { sessionUpdate: "auto_compact_started" },
        },
        compactionSessionId,
      ),
    ).toEqual([
      {
        type: "compaction",
        status: "loading",
        trigger: "auto",
      },
    ]);
  });

  test("settles failed and canceled compact without leaving a loading marker", () => {
    expect(
      mapGrokCompactionExtensionNotification(
        GROK_SESSION_NOTIFICATION_METHOD,
        {
          sessionId: "session-1",
          update: { sessionUpdate: "auto_compact_failed" },
        },
        compactionSessionId,
      ),
    ).toEqual([
      {
        type: "compaction",
        status: "completed",
        trigger: "auto",
      },
    ]);
    expect(
      mapGrokCompactionExtensionNotification(
        GROK_SESSION_NOTIFICATION_METHOD,
        {
          sessionId: "session-1",
          update: { sessionUpdate: "auto_compact_cancelled" },
        },
        compactionSessionId,
      ),
    ).toEqual([
      {
        type: "compaction",
        status: "completed",
        trigger: "auto",
      },
    ]);
  });

  test("ignores checkpoints and unknown updates", () => {
    expect(
      mapGrokCompactionExtensionNotification(
        GROK_SESSION_NOTIFICATION_METHOD,
        {
          sessionId: "session-1",
          update: { sessionUpdate: "compaction_checkpoint" },
        },
        compactionSessionId,
      ),
    ).toEqual([]);
    expect(
      mapGrokCompactionExtensionNotification(
        GROK_SESSION_NOTIFICATION_METHOD,
        { sessionId: "session-1", update: { sessionUpdate: "turn_completed" } },
        compactionSessionId,
      ),
    ).toEqual([]);
  });
});

describe("Grok ACP session compaction", () => {
  test("emits loading compaction from /compact because Grok omits auto_compact_started", async () => {
    const session = readyGrokSession([grokModel("or-grok", 500000)], "or-grok");
    const events: AgentStreamEvent[] = [];
    session.subscribe((event) => {
      events.push(event);
    });

    const { turnId } = await session.startTurn("/compact");
    await session.extNotification(GROK_SESSION_NOTIFICATION_METHOD, {
      sessionId: "session-1",
      update: {
        sessionUpdate: "auto_compact_completed",
        tokens_before: 37418,
        tokens_after: 16437,
        summary_preview: null,
      },
    });

    expect(
      events.filter((event) => event.type === "timeline" && event.item.type === "compaction"),
    ).toEqual([
      {
        type: "timeline",
        provider: "grok",
        item: { type: "compaction", status: "loading", trigger: "manual" },
        turnId,
      },
      {
        type: "timeline",
        provider: "grok",
        item: {
          type: "compaction",
          status: "completed",
          trigger: "manual",
          preTokens: 37418,
        },
      },
    ]);
  });

  test("emits compaction timeline items from _x.ai/session_notification", async () => {
    const session = readyGrokSession([grokModel("or-grok", 500000)], "or-grok");
    const events: AgentStreamEvent[] = [];
    session.subscribe((event) => {
      events.push(event);
    });

    await session.extNotification(GROK_SESSION_NOTIFICATION_METHOD, {
      sessionId: "session-1",
      update: { sessionUpdate: "auto_compact_started" },
    });
    await session.extNotification(GROK_SESSION_NOTIFICATION_METHOD, {
      sessionId: "session-1",
      update: {
        sessionUpdate: "auto_compact_completed",
        tokens_before: 12304,
        tokens_after: 12304,
        summary_preview: null,
      },
    });

    expect(events.filter((event) => event.type === "timeline")).toEqual([
      {
        type: "timeline",
        provider: "grok",
        item: { type: "compaction", status: "loading", trigger: "auto" },
      },
      {
        type: "timeline",
        provider: "grok",
        item: {
          type: "compaction",
          status: "completed",
          trigger: "auto",
          preTokens: 12304,
        },
      },
    ]);
  });
});

describe("Grok rewind mapping", () => {
  test("parses rewind execute success and optional error", () => {
    expect(
      parseGrokRewindExecuteResult({
        success: true,
        target_prompt_index: 0,
        mode: "conversation_only",
        prompt_text: "first",
        error: null,
      }),
    ).toEqual({ success: true, error: null });
    expect(parseGrokRewindExecuteResult({ success: false, error: null })).toEqual({
      success: false,
      error: null,
    });
  });

  test("parses rewind points from Grok's snake_case payload", () => {
    expect(
      parseGrokRewindPoints({
        rewind_points: [
          {
            prompt_index: 0,
            prompt_preview: "list files in ~/Developer",
            has_file_changes: false,
          },
        ],
      }),
    ).toEqual([
      {
        promptIndex: 0,
        promptPreview: "list files in ~/Developer",
      },
    ]);
  });

  test("maps a live client message id by prompt index", () => {
    expect(
      resolveGrokRewindPromptIndex({
        messageId: "client-2",
        userMessages: [
          { messageId: "client-1", text: "first" },
          { messageId: "client-2", text: "second" },
        ],
        rewindPoints: [
          { promptIndex: 0, promptPreview: "first" },
          { promptIndex: 1, promptPreview: "second" },
        ],
      }),
    ).toBe(1);
  });

  test("skips slash-command rows by matching the rewind-point preview", () => {
    expect(
      resolveGrokRewindPromptIndex({
        messageId: "client-2",
        userMessages: [
          { messageId: "client-1", text: "first" },
          { messageId: "compact-1", text: "/compact" },
          { messageId: "client-2", text: "second prompt after compact" },
        ],
        rewindPoints: [
          { promptIndex: 0, promptPreview: "first" },
          { promptIndex: 1, promptPreview: "second prompt" },
        ],
      }),
    ).toBe(1);
  });

  test("maps a resumed grok-prompt id to the rewind-point index", () => {
    expect(
      resolveGrokRewindPromptIndex({
        messageId: grokPromptMessageId(0),
        userMessages: [{ messageId: grokPromptMessageId(0), text: "list files in ~/Developer" }],
        rewindPoints: [{ promptIndex: 0, promptPreview: "list files in ~/Developer" }],
      }),
    ).toBe(0);
  });

  test("stamps history rows from rewind-point previews and skips slash commands", () => {
    const userMessages = [
      { text: "list files in ~/Developer\nand then compact" },
      { text: "/compact" },
      { text: "follow up" },
    ];

    expect(
      assignGrokPromptMessageIds({
        userMessages,
        rewindPoints: [
          { promptIndex: 0, promptPreview: "list files in ~/Developer" },
          { promptIndex: 1, promptPreview: "follow up" },
        ],
      }),
    ).toEqual([
      { messageId: grokPromptMessageId(0), text: "list files in ~/Developer\nand then compact" },
      { messageId: grokPromptMessageId(1), text: "follow up" },
    ]);
    expect(userMessages[0]?.messageId).toBe(grokPromptMessageId(0));
    expect(userMessages[1]?.messageId).toBeUndefined();
    expect(userMessages[2]?.messageId).toBe(grokPromptMessageId(1));
  });
});

describe("Grok conversation rewind", () => {
  test("advertises conversation rewind on the Grok ACP client", () => {
    const client = new GrokACPAgentClient({
      logger: createTestLogger(),
      command: ["grok", "agent", "stdio"],
    });

    expect(client.capabilities.supportsRewindConversation).toBe(true);
    expect(client.capabilities.supportsRewindFiles).toBe(false);
    expect(client.capabilities.supportsRewindBoth).toBe(false);
  });

  test("executes conversation rewind through Grok's vendor ACP methods", async () => {
    const extMethod = vi.fn(async (method: string) => {
      if (method === GROK_REWIND_POINTS_METHOD) {
        return {
          rewind_points: [
            { prompt_index: 0, prompt_preview: "first" },
            { prompt_index: 1, prompt_preview: "second" },
          ],
        };
      }
      return { success: true, error: null };
    });
    const loadSession = vi.fn(async () => ({}));
    const session = readyGrokSession([grokModel("or-grok", 500000)], "or-grok");
    const internals = asInternals<{
      sessionId: string | null;
      connection: {
        prompt: () => Promise<PromptResponse>;
        extMethod: typeof extMethod;
        loadSession: typeof loadSession;
      } | null;
      agentCapabilities: { loadSession?: boolean } | null;
      userPrompts: Array<{ messageId: string; text: string }>;
    }>(session);
    internals.agentCapabilities = { loadSession: true };
    internals.userPrompts = [
      { messageId: "client-1", text: "first" },
      { messageId: "client-2", text: "second" },
    ];
    internals.connection = {
      prompt: async () => ({ stopReason: "end_turn" }),
      extMethod,
      loadSession,
    };

    await session.revertConversation({ messageId: "client-1" });

    expect(extMethod).toHaveBeenCalledWith(GROK_REWIND_EXECUTE_METHOD, {
      sessionId: "session-1",
      targetPromptIndex: 0,
      force: true,
      mode: "conversation_only",
    });
    expect(loadSession).toHaveBeenCalledWith({
      sessionId: "session-1",
      cwd: "/tmp/paseo-grok-test",
      mcpServers: [],
    });
  });

  test("treats Grok rewind execute success:false as a failed rewind", async () => {
    const extMethod = vi.fn(async (method: string) => {
      if (method === GROK_REWIND_POINTS_METHOD) {
        return {
          rewind_points: [{ prompt_index: 0, prompt_preview: "first" }],
        };
      }
      return {
        success: false,
        target_prompt_index: 0,
        mode: "all",
        prompt_text: null,
        error: null,
      };
    });
    const loadSession = vi.fn(async () => ({}));
    const session = readyGrokSession([grokModel("or-grok", 500000)], "or-grok");
    const internals = asInternals<{
      sessionId: string | null;
      connection: {
        prompt: () => Promise<PromptResponse>;
        extMethod: typeof extMethod;
        loadSession: typeof loadSession;
      } | null;
      agentCapabilities: { loadSession?: boolean } | null;
      userPrompts: Array<{ messageId: string; text: string }>;
    }>(session);
    internals.agentCapabilities = { loadSession: true };
    internals.userPrompts = [{ messageId: "client-1", text: "first" }];
    internals.connection = {
      prompt: async () => ({ stopReason: "end_turn" }),
      extMethod,
      loadSession,
    };

    await expect(session.revertConversation({ messageId: "client-1" })).rejects.toThrow(
      "Grok rewind did not truncate the conversation",
    );
    expect(loadSession).not.toHaveBeenCalled();
  });
});
