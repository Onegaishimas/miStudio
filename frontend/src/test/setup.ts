import '@testing-library/jest-dom';
import { cleanup } from '@testing-library/react';
import { afterEach, vi } from 'vitest';

// Globally stub socket.io-client so no test opens a real network connection.
// WebSocketProvider calls io() inside a useEffect; without this, any component
// tree wrapped in <WebSocketProvider> would attempt a live socket.io handshake.
// The stub returns a harmless fake Socket with no-op event/emit methods.
vi.mock('socket.io-client', () => {
  const createFakeSocket = () => ({
    id: 'test-socket',
    connected: false,
    on: vi.fn(),
    off: vi.fn(),
    emit: vi.fn(),
    connect: vi.fn(),
    disconnect: vi.fn(),
    removeListener: vi.fn(),
    removeAllListeners: vi.fn(),
  });
  const io = vi.fn(() => createFakeSocket());
  return { io, default: io, Socket: class {} };
});

// Cleanup after each test
afterEach(() => {
  cleanup();
});
