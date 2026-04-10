/**
 * SettingsPanel - Application settings with tabbed interface.
 *
 * Tabs: Endpoints, API Keys, Labeling, Display
 * Data is persisted to the backend database via the settings API.
 * Sensitive values (API keys) are encrypted at rest.
 */

import { useState, useEffect } from 'react';
import { Plus, Trash2, Eye, EyeOff, Save, AlertCircle, CheckCircle2 } from 'lucide-react';
import { useSettingsStore } from '../../stores/settingsStore';

type SettingsTab = 'endpoints' | 'api_keys' | 'labeling' | 'display';

const TABS: { id: SettingsTab; label: string }[] = [
  { id: 'endpoints', label: 'Endpoints' },
  { id: 'api_keys', label: 'API Keys' },
  { id: 'labeling', label: 'Labeling' },
  { id: 'display', label: 'Display' },
];

export function SettingsPanel() {
  const [activeTab, setActiveTab] = useState<SettingsTab>('endpoints');
  const { fetchAll, isLoading } = useSettingsStore();

  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  return (
    <div className="px-6 py-8">
      {/* Tabs */}
      <div className="mb-6">
        <div className="border-b border-slate-800">
          <nav className="flex gap-1">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-6 py-3 font-medium transition-colors relative ${
                  activeTab === tab.id
                    ? 'text-emerald-400'
                    : 'text-slate-400 hover:text-slate-300'
                }`}
              >
                {tab.label}
                {activeTab === tab.id && (
                  <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-emerald-400" />
                )}
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Tab Content */}
      {isLoading && (
        <div className="text-slate-500 text-sm py-4">Loading settings...</div>
      )}
      {activeTab === 'endpoints' && <EndpointsTab />}
      {activeTab === 'api_keys' && <ApiKeysTab />}
      {activeTab === 'labeling' && <LabelingTab />}
      {activeTab === 'display' && <DisplayTab />}
    </div>
  );
}

// ─── Endpoints Tab ───────────────────────────────────────────────────────────

function EndpointsTab() {
  const { settings, getByCategory, upsert, remove } = useSettingsStore();
  const endpoints = getByCategory('endpoints');
  const [url, setUrl] = useState('');
  const [label, setLabel] = useState('');
  const [toast, setToast] = useState<string | null>(null);

  // Ollama URL setting
  const ollamaUrlSetting = settings.find((s) => s.key === 'ollama_url');
  const [ollamaUrl, setOllamaUrl] = useState(ollamaUrlSetting?.value ?? '');
  useEffect(() => {
    setOllamaUrl(ollamaUrlSetting?.value ?? '');
  }, [ollamaUrlSetting?.value]);

  const handleSaveOllamaUrl = async () => {
    await upsert({ key: 'ollama_url', value: ollamaUrl.trim(), is_sensitive: false, category: 'endpoints' });
    setToast('Ollama URL saved');
    setTimeout(() => setToast(null), 2000);
  };

  const handleAdd = async () => {
    const trimmed = url.trim();
    if (!trimmed) return;
    // Use a sanitized key: endpoint:<url>
    const key = `endpoint:${trimmed}`;
    await upsert({
      key,
      value: JSON.stringify({ url: trimmed, label: label.trim() || undefined, lastUsed: Date.now() }),
      is_sensitive: false,
      category: 'endpoints',
    });
    setUrl('');
    setLabel('');
    setToast('Endpoint saved');
    setTimeout(() => setToast(null), 2000);
  };

  const handleDelete = async (key: string) => {
    await remove(key);
  };

  const parseEndpoint = (value: string) => {
    try {
      return JSON.parse(value) as { url: string; label?: string; lastUsed?: number };
    } catch {
      return { url: value };
    }
  };

  return (
    <div className="max-w-2xl">
      <h2 className="text-lg font-semibold text-slate-200 mb-1">API Endpoints</h2>
      <p className="text-xs text-slate-500 mb-4">Saved OpenAI-compatible endpoint URLs for labeling jobs.</p>

      {/* Ollama / LLM Service URL */}
      <div className="bg-slate-900 border border-slate-700 rounded-lg p-4 mb-6">
        <label className="block text-sm font-medium text-slate-200 mb-1">Ollama / LLM Service URL</label>
        <p className="text-xs text-slate-500 mb-3">
          Base URL for the local LLM service used for labeling. Overrides the server environment variable.
        </p>
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={ollamaUrl}
            onChange={(e) => setOllamaUrl(e.target.value)}
            placeholder="http://k8s-millm.hitsai.local"
            className="flex-1 bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:border-emerald-500 focus:outline-none font-mono"
          />
          <button
            onClick={handleSaveOllamaUrl}
            disabled={!ollamaUrl.trim()}
            className="flex items-center gap-1.5 px-3 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 disabled:text-slate-500 text-white text-sm rounded transition-colors"
          >
            <Save className="w-4 h-4" /> Save
          </button>
          {ollamaUrlSetting && (
            <button
              onClick={() => { remove('ollama_url'); setOllamaUrl(''); }}
              className="p-2 text-slate-500 hover:text-red-400 transition-colors"
              title="Clear (revert to server default)"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}
        </div>
        {ollamaUrlSetting && (
          <p className="text-xs text-slate-600 mt-1">
            Set {new Date(ollamaUrlSetting.updated_at).toLocaleDateString()}
          </p>
        )}
      </div>

      {/* Add form */}
      <div className="bg-slate-900 border border-slate-700 rounded-lg p-4 mb-4">
        <div className="space-y-3">
          <div>
            <label className="block text-xs text-slate-400 mb-1">URL</label>
            <input
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="http://192.168.244.61:8001/v1"
              className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:border-emerald-500 focus:outline-none"
            />
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">Label (optional)</label>
            <input
              type="text"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder="miLLM GPU Server"
              className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:border-emerald-500 focus:outline-none"
            />
          </div>
          <button
            onClick={handleAdd}
            disabled={!url.trim()}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 disabled:text-slate-500 text-white text-sm rounded transition-colors"
          >
            <Plus className="w-4 h-4" /> Add Endpoint
          </button>
        </div>
      </div>

      {/* Toast */}
      {toast && (
        <div className="flex items-center gap-2 text-emerald-400 text-xs mb-3">
          <CheckCircle2 className="w-4 h-4" /> {toast}
        </div>
      )}

      {/* Saved endpoints list */}
      <h3 className="text-sm font-medium text-slate-300 mb-2">Saved Endpoints</h3>
      {endpoints.length === 0 ? (
        <p className="text-xs text-slate-500">No saved endpoints yet.</p>
      ) : (
        <div className="space-y-2">
          {endpoints.map((s) => {
            const ep = parseEndpoint(s.value);
            return (
              <div
                key={s.id}
                className="flex items-center justify-between bg-slate-900 border border-slate-800 rounded-lg px-4 py-3 group"
              >
                <div>
                  {ep.label && (
                    <div className="text-sm font-medium text-slate-200">{ep.label}</div>
                  )}
                  <div className="text-xs font-mono text-slate-400">{ep.url}</div>
                  {ep.lastUsed && (
                    <div className="text-xs text-slate-600 mt-0.5">
                      Last used: {new Date(ep.lastUsed).toLocaleDateString()}
                    </div>
                  )}
                </div>
                <button
                  onClick={() => handleDelete(s.key)}
                  className="p-1.5 text-slate-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all"
                  title="Delete endpoint"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ─── API Keys Tab ────────────────────────────────────────────────────────────

const API_KEY_PRESETS = [
  { key: 'openai_api_key', label: 'OpenAI API Key', placeholder: 'sk-proj-...' },
  { key: 'hf_token', label: 'HuggingFace Token', placeholder: 'hf_...' },
];

function ApiKeysTab() {
  const { settings, upsert, remove, fetchAll } = useSettingsStore();
  const apiKeys = settings.filter((s) => s.category === 'api_keys');
  const [editingKey, setEditingKey] = useState<string | null>(null);
  const [newValue, setNewValue] = useState('');
  const [showValue, setShowValue] = useState(false);
  const [toast, setToast] = useState<string | null>(null);

  const handleSave = async (key: string) => {
    if (!newValue.trim()) return;
    await upsert({
      key,
      value: newValue.trim(),
      is_sensitive: true,
      category: 'api_keys',
    });
    setEditingKey(null);
    setNewValue('');
    setShowValue(false);
    setToast(`${key} saved`);
    setTimeout(() => setToast(null), 2000);
    await fetchAll(); // Re-fetch to get masked value
  };

  const handleDelete = async (key: string) => {
    await remove(key);
    setToast(`${key} deleted`);
    setTimeout(() => setToast(null), 2000);
  };

  const getExisting = (key: string) => apiKeys.find((s) => s.key === key);

  return (
    <div className="max-w-2xl">
      <h2 className="text-lg font-semibold text-slate-200 mb-1">API Keys</h2>
      <p className="text-xs text-slate-500 mb-4">
        Keys are encrypted at rest (AES-256-GCM) and never displayed in full after saving.
      </p>

      {toast && (
        <div className="flex items-center gap-2 text-emerald-400 text-xs mb-3">
          <CheckCircle2 className="w-4 h-4" /> {toast}
        </div>
      )}

      <div className="space-y-3">
        {API_KEY_PRESETS.map((preset) => {
          const existing = getExisting(preset.key);
          const isEditing = editingKey === preset.key;

          return (
            <div
              key={preset.key}
              className="bg-slate-900 border border-slate-700 rounded-lg p-4"
            >
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm font-medium text-slate-200">{preset.label}</label>
                <div className="flex items-center gap-1">
                  {existing && !isEditing && (
                    <>
                      <button
                        onClick={() => {
                          setEditingKey(preset.key);
                          setNewValue('');
                          setShowValue(false);
                        }}
                        className="text-xs text-slate-400 hover:text-slate-200 px-2 py-1 rounded hover:bg-slate-800 transition-colors"
                      >
                        Edit
                      </button>
                      <button
                        onClick={() => handleDelete(preset.key)}
                        className="text-xs text-slate-400 hover:text-red-400 px-2 py-1 rounded hover:bg-slate-800 transition-colors"
                      >
                        Delete
                      </button>
                    </>
                  )}
                </div>
              </div>

              {isEditing ? (
                <div className="flex items-center gap-2">
                  <div className="relative flex-1">
                    <input
                      type={showValue ? 'text' : 'password'}
                      value={newValue}
                      onChange={(e) => setNewValue(e.target.value)}
                      placeholder={preset.placeholder}
                      className="w-full bg-slate-800 border border-slate-600 rounded px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:border-emerald-500 focus:outline-none pr-9"
                      autoFocus
                    />
                    <button
                      onClick={() => setShowValue(!showValue)}
                      className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
                    >
                      {showValue ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                  <button
                    onClick={() => handleSave(preset.key)}
                    disabled={!newValue.trim()}
                    className="px-3 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 disabled:text-slate-500 text-white text-sm rounded transition-colors"
                  >
                    <Save className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => { setEditingKey(null); setNewValue(''); setShowValue(false); }}
                    className="px-3 py-2 text-slate-400 hover:text-slate-200 text-sm"
                  >
                    Cancel
                  </button>
                </div>
              ) : existing ? (
                <div className="flex items-center gap-2">
                  <code className="text-sm font-mono text-slate-400 bg-slate-800 px-3 py-2 rounded flex-1">
                    {existing.value}
                  </code>
                </div>
              ) : (
                <div>
                  <div className="flex items-center gap-2">
                    <div className="relative flex-1">
                      <input
                        type={showValue && editingKey === null ? 'text' : 'password'}
                        value={editingKey === null && newValue ? newValue : ''}
                        onChange={(e) => { setEditingKey(null); setNewValue(e.target.value); }}
                        onFocus={() => setEditingKey(preset.key)}
                        placeholder={`Enter ${preset.label}...`}
                        className="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:border-emerald-500 focus:outline-none"
                      />
                    </div>
                  </div>
                  <p className="text-xs text-slate-600 mt-1">Not configured</p>
                </div>
              )}

              {existing && !isEditing && existing.updated_at && (
                <p className="text-xs text-slate-600 mt-1">
                  Set {new Date(existing.updated_at).toLocaleDateString()}
                </p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Labeling Tab ────────────────────────────────────────────────────────────

function LabelingTab() {
  const { settings, upsert } = useSettingsStore();
  const [toast, setToast] = useState<string | null>(null);

  const getValue = (key: string, defaultVal: string) => {
    const s = settings.find((s) => s.key === key);
    return s?.value ?? defaultVal;
  };

  const handleSave = async (key: string, value: string) => {
    await upsert({ key, value, is_sensitive: false, category: 'labeling' });
    setToast('Saved');
    setTimeout(() => setToast(null), 2000);
  };

  const [batchSize, setBatchSize] = useState(getValue('labeling_default_batch_size', '10'));
  const [maxExamples, setMaxExamples] = useState(getValue('labeling_default_max_examples', '25'));

  // Sync local state when settings load
  useEffect(() => {
    setBatchSize(getValue('labeling_default_batch_size', '10'));
    setMaxExamples(getValue('labeling_default_max_examples', '25'));
  }, [settings]);

  return (
    <div className="max-w-2xl">
      <h2 className="text-lg font-semibold text-slate-200 mb-1">Labeling Defaults</h2>
      <p className="text-xs text-slate-500 mb-4">
        These values pre-fill when starting a new labeling job.
      </p>

      {toast && (
        <div className="flex items-center gap-2 text-emerald-400 text-xs mb-3">
          <CheckCircle2 className="w-4 h-4" /> {toast}
        </div>
      )}

      <div className="bg-slate-900 border border-slate-700 rounded-lg p-4 space-y-4">
        <div>
          <label className="block text-xs text-slate-400 mb-1">Default Batch Size</label>
          <input
            type="number"
            value={batchSize}
            onChange={(e) => setBatchSize(e.target.value)}
            min={1}
            max={100}
            className="w-32 bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm text-slate-200 focus:border-emerald-500 focus:outline-none"
          />
        </div>
        <div>
          <label className="block text-xs text-slate-400 mb-1">Default Max Examples per Feature</label>
          <input
            type="number"
            value={maxExamples}
            onChange={(e) => setMaxExamples(e.target.value)}
            min={5}
            max={100}
            className="w-32 bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm text-slate-200 focus:border-emerald-500 focus:outline-none"
          />
        </div>
        <button
          onClick={async () => {
            await handleSave('labeling_default_batch_size', batchSize);
            await handleSave('labeling_default_max_examples', maxExamples);
          }}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white text-sm rounded transition-colors"
        >
          <Save className="w-4 h-4" /> Save Defaults
        </button>
      </div>
    </div>
  );
}

// ─── Display Tab ─────────────────────────────────────────────────────────────

function DisplayTab() {
  return (
    <div className="max-w-2xl">
      <h2 className="text-lg font-semibold text-slate-200 mb-1">Display Preferences</h2>
      <p className="text-xs text-slate-500 mb-4">
        Saved locally in your browser. These settings don't sync across devices.
      </p>

      <div className="bg-slate-900 border border-slate-700 rounded-lg p-4">
        <div className="flex items-center gap-3">
          <AlertCircle className="w-4 h-4 text-slate-500" />
          <p className="text-sm text-slate-400">
            Display preferences are managed via the sidebar collapse toggle and theme button in the header.
            Additional display settings will be added here in a future update.
          </p>
        </div>
      </div>
    </div>
  );
}
