import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface SavedEndpoint {
  url: string;
  label?: string;
  lastUsed: number; // timestamp
}

interface SavedConfigState {
  endpoints: SavedEndpoint[];
  addEndpoint: (url: string, label?: string) => void;
  removeEndpoint: (url: string) => void;
  touchEndpoint: (url: string) => void;
}

export const useSavedConfigStore = create<SavedConfigState>()(
  persist(
    (set) => ({
      endpoints: [],
      addEndpoint: (url: string, label?: string) =>
        set((state) => {
          const trimmed = url.trim();
          if (!trimmed) return state;
          const existing = state.endpoints.find((e) => e.url === trimmed);
          if (existing) {
            return {
              endpoints: state.endpoints.map((e) =>
                e.url === trimmed ? { ...e, label: label ?? e.label, lastUsed: Date.now() } : e
              ),
            };
          }
          return {
            endpoints: [{ url: trimmed, label, lastUsed: Date.now() }, ...state.endpoints],
          };
        }),
      removeEndpoint: (url: string) =>
        set((state) => ({
          endpoints: state.endpoints.filter((e) => e.url !== url),
        })),
      touchEndpoint: (url: string) =>
        set((state) => ({
          endpoints: state.endpoints.map((e) =>
            e.url === url ? { ...e, lastUsed: Date.now() } : e
          ),
        })),
    }),
    {
      name: 'mistudio-saved-config',
    }
  )
);
