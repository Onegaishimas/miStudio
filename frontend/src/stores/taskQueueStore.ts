/**
 * Task Queue Store
 *
 * Zustand store for managing task queue state.
 * Provides visibility and control over background operations.
 *
 * Loading/error state is tracked per list (active vs failed) because the two
 * Monitor sections poll on independent intervals — a shared flag makes the
 * pollers race and shows one section's errors in the other.
 */

import { create } from 'zustand';
import { TaskQueueEntry, RetryRequest } from '../types/taskQueue';
import * as taskQueueApi from '../api/taskQueue';

interface TaskQueueState {
  // State
  tasks: TaskQueueEntry[];
  failedTasks: TaskQueueEntry[];
  activeTasks: TaskQueueEntry[];
  loading: boolean; // fetchTasks / retry / delete operations
  activeLoading: boolean;
  failedLoading: boolean;
  error: string | null; // retry / delete / fetchTasks errors
  activeError: string | null;
  failedError: string | null;

  // Actions
  fetchTasks: (status?: string, entityType?: string) => Promise<void>;
  fetchFailedTasks: () => Promise<void>;
  fetchActiveTasks: () => Promise<void>;
  retryTask: (taskQueueId: string, request?: RetryRequest) => Promise<void>;
  deleteTask: (taskQueueId: string) => Promise<void>;
  clearError: () => void;
}

export const useTaskQueueStore = create<TaskQueueState>((set, get) => ({
  // Initial state
  tasks: [],
  failedTasks: [],
  activeTasks: [],
  loading: false,
  activeLoading: false,
  failedLoading: false,
  error: null,
  activeError: null,
  failedError: null,

  // Fetch all tasks with optional filtering
  fetchTasks: async (status?: string, entityType?: string) => {
    try {
      set({ loading: true, error: null });
      const response = await taskQueueApi.getTaskQueue(status, entityType);
      set({ tasks: response.data, loading: false });
    } catch (error) {
      console.error('[TaskQueueStore] Failed to fetch tasks:', error);
      set({
        error: error instanceof Error ? error.message : 'Failed to fetch tasks',
        loading: false,
      });
    }
  },

  // Fetch failed tasks
  fetchFailedTasks: async () => {
    try {
      set({ failedLoading: true, failedError: null });
      const response = await taskQueueApi.getFailedTasks();
      set({ failedTasks: response.data, failedLoading: false });
    } catch (error) {
      console.error('[TaskQueueStore] Failed to fetch failed tasks:', error);
      set({
        failedError: error instanceof Error ? error.message : 'Failed to fetch failed tasks',
        failedLoading: false,
      });
    }
  },

  // Fetch active tasks
  fetchActiveTasks: async () => {
    try {
      set({ activeLoading: true, activeError: null });
      const response = await taskQueueApi.getActiveTasks();
      set({ activeTasks: response.data, activeLoading: false });
    } catch (error) {
      console.error('[TaskQueueStore] Failed to fetch active tasks:', error);
      set({
        activeError: error instanceof Error ? error.message : 'Failed to fetch active tasks',
        activeLoading: false,
      });
    }
  },

  // Retry a failed task
  retryTask: async (taskQueueId: string, request?: RetryRequest) => {
    try {
      set({ loading: true, error: null });
      const response = await taskQueueApi.retryTask(taskQueueId, request);

      console.log('[TaskQueueStore] Task retry initiated:', response);

      // Remove from failed tasks list
      set((state) => ({
        failedTasks: state.failedTasks.filter((task) => task.id !== taskQueueId),
        loading: false,
      }));

      // Refresh active tasks to show the retry
      await get().fetchActiveTasks();
    } catch (error) {
      console.error('[TaskQueueStore] Failed to retry task:', error);
      set({
        error: error instanceof Error ? error.message : 'Failed to retry task',
        loading: false,
      });
      throw error;
    }
  },

  // Delete a task
  deleteTask: async (taskQueueId: string) => {
    try {
      set({ loading: true, error: null });
      await taskQueueApi.deleteTask(taskQueueId);

      // Remove from all task lists
      set((state) => ({
        tasks: state.tasks.filter((task) => task.id !== taskQueueId),
        failedTasks: state.failedTasks.filter((task) => task.id !== taskQueueId),
        activeTasks: state.activeTasks.filter((task) => task.id !== taskQueueId),
        loading: false,
      }));

      console.log('[TaskQueueStore] Task deleted:', taskQueueId);
    } catch (error) {
      console.error('[TaskQueueStore] Failed to delete task:', error);
      set({
        error: error instanceof Error ? error.message : 'Failed to delete task',
        loading: false,
      });
      throw error;
    }
  },

  // Clear errors
  clearError: () => {
    set({ error: null, activeError: null, failedError: null });
  },
}));
