/**
 * MIS-E2E-120 / -126 — reconnects duplicated every handler.
 *
 * The `connect` handler re-attached every entry in `eventHandlersRef`, under a
 * comment reading "Re-attach existing handlers FIRST (for reconnections)".
 * socket.io does NOT detach listeners on disconnect: the same Socket instance
 * keeps them across the whole reconnect cycle. So each reconnect added a SECOND
 * registration of every handler already attached, and after N reconnects one
 * server message ran every handler N+1 times.
 *
 * Reconnects are routine — a pod restart, a laptop waking. For a progress event
 * the duplication is noise; for `extraction:completed`, or any store action
 * that appends, it is N+1 duplicate effects from a single event.
 *
 * And `unsubscribe` never cleared `pendingSubscriptionsRef`, so a channel the
 * user abandoned while disconnected was subscribed on the next connect and
 * every reconnect after that — compounding the above.
 *
 * This file did not exist. Neither behaviour was pinned by anything.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, act } from '@testing-library/react';
import { useEffect } from 'react';

/** A socket.io double that behaves the way the real one does on reconnect. */
class FakeSocket {
  id = 'fake-1';
  connected = false;
  /** event -> handlers, exactly as socket.io keeps them: NOT cleared on disconnect. */
  listeners = new Map<string, Array<(...a: unknown[]) => void>>();
  emitted: Array<{ event: string; payload: unknown }> = [];

  on(event: string, handler: (...a: unknown[]) => void) {
    if (!this.listeners.has(event)) this.listeners.set(event, []);
    this.listeners.get(event)!.push(handler);
  }

  off(event: string, handler?: (...a: unknown[]) => void) {
    if (!handler) this.listeners.delete(event);
    else {
      const list = this.listeners.get(event) ?? [];
      this.listeners.set(event, list.filter((h) => h !== handler));
    }
  }

  emit(event: string, payload?: unknown) {
    this.emitted.push({ event, payload });
  }

  disconnect() {
    this.connected = false;
  }

  /** Drive a (re)connect the way socket.io does: fire 'connect' again. */
  fireConnect() {
    this.connected = true;
    (this.listeners.get('connect') ?? []).slice().forEach((h) => h());
  }

  /** Deliver a server message. */
  deliver(event: string, payload: unknown) {
    (this.listeners.get(event) ?? []).slice().forEach((h) => h(payload));
  }

  countFor(event: string) {
    return (this.listeners.get(event) ?? []).length;
  }
}

let fake: FakeSocket;

vi.mock('socket.io-client', () => ({
  io: () => fake,
}));

// Imported after the mock so the provider picks up the double.
const { WebSocketProvider, useWebSocketContext } = await import('./WebSocketContext');

function Consumer({ onEvent }: { onEvent: (p: unknown) => void }) {
  const { on, subscribe } = useWebSocketContext();
  useEffect(() => {
    subscribe('extraction/e1');
    on('extraction:completed', onEvent);
    // Intentionally no cleanup: the provider owns handler lifetime, and this
    // mirrors how the panels actually use it.
  }, [on, subscribe, onEvent]);
  return null;
}

beforeEach(() => {
  fake = new FakeSocket();
  vi.restoreAllMocks();
  vi.spyOn(console, 'log').mockImplementation(() => {});
});

describe('WebSocketContext across reconnects', () => {
  it('registers a handler exactly once, however many reconnects happen', () => {
    const seen = vi.fn();
    render(
      <WebSocketProvider>
        <Consumer onEvent={seen} />
      </WebSocketProvider>,
    );

    act(() => fake.fireConnect());
    expect(fake.countFor('extraction:completed')).toBe(1);

    // Three reconnects, the way a flaky network or a pod restart produces them.
    act(() => fake.fireConnect());
    act(() => fake.fireConnect());
    act(() => fake.fireConnect());

    expect(fake.countFor('extraction:completed')).toBe(1);
  });

  it('runs a handler once per server message after reconnects', () => {
    // The consequence, stated as behaviour rather than as a listener count.
    const seen = vi.fn();
    render(
      <WebSocketProvider>
        <Consumer onEvent={seen} />
      </WebSocketProvider>,
    );

    act(() => fake.fireConnect());
    act(() => fake.fireConnect());
    act(() => fake.fireConnect());

    act(() => fake.deliver('extraction:completed', { id: 'e1' }));

    expect(seen).toHaveBeenCalledTimes(1);
  });

  it('resubscribes each active channel once per connect, not cumulatively', () => {
    render(
      <WebSocketProvider>
        <Consumer onEvent={() => {}} />
      </WebSocketProvider>,
    );

    act(() => fake.fireConnect());
    const afterFirst = fake.emitted.filter((e) => e.event === 'subscribe').length;

    act(() => fake.fireConnect());
    const afterSecond = fake.emitted.filter((e) => e.event === 'subscribe').length;

    // One more subscribe per connect for the one active channel — not two, and
    // not growing.
    expect(afterSecond - afterFirst).toBe(1);
  });
});

describe('unsubscribe clears the pending queue (MIS-E2E-126)', () => {
  function Abandoner() {
    const { subscribe, unsubscribe } = useWebSocketContext();
    useEffect(() => {
      // Subscribed while DISCONNECTED, so it lands in the pending queue...
      subscribe('steering/task-abandoned');
      // ...and abandoned before the socket ever connects.
      unsubscribe('steering/task-abandoned');
    }, [subscribe, unsubscribe]);
    return null;
  }

  it('does not subscribe a channel the user abandoned while disconnected', () => {
    render(
      <WebSocketProvider>
        <Abandoner />
      </WebSocketProvider>,
    );

    act(() => fake.fireConnect());

    const subscribed = fake.emitted
      .filter((e) => e.event === 'subscribe')
      .map((e) => (e.payload as { channel: string }).channel);
    expect(subscribed).not.toContain('steering/task-abandoned');
  });

  it('and does not resubscribe it on every reconnect thereafter', () => {
    render(
      <WebSocketProvider>
        <Abandoner />
      </WebSocketProvider>,
    );

    act(() => fake.fireConnect());
    act(() => fake.fireConnect());
    act(() => fake.fireConnect());

    const count = fake.emitted
      .filter(
        (e) =>
          e.event === 'subscribe' &&
          (e.payload as { channel: string }).channel === 'steering/task-abandoned',
      )
      .length;
    expect(count).toBe(0);
  });
});
