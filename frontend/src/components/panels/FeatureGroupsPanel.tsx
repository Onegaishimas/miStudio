/**
 * Feature Groups panel (Feature 010).
 *
 * Browse cross-feature groups — features sharing a top activating token with
 * similar activation context — for a completed extraction. Same REST
 * endpoints the MCP server's `groups` tools use: one source of truth.
 */

import { useEffect, useMemo, useState } from 'react';
import { Boxes, Sliders } from 'lucide-react';
import { getSAE } from '../../api/saes';
import { useFeatureGroupsStore } from '../../stores/featureGroupsStore';
import { useFeaturesStore } from '../../stores/featuresStore';
import { useSteeringStore } from '../../stores/steeringStore';
import { useFeatureGroupsWebSocket } from '../../hooks/useFeatureGroupsWebSocket';
import { ComputeIndexBanner } from '../featureGroups/ComputeIndexBanner';
import { GroupList } from '../featureGroups/GroupList';
import { RelatedFeaturesDrawer } from '../featureGroups/RelatedFeaturesDrawer';
import { FeatureDetailModal } from '../features/FeatureDetailModal';

interface FeatureGroupsPanelProps {
  onNavigateToSteering?: () => void;
}

export function FeatureGroupsPanel({ onNavigateToSteering }: FeatureGroupsPanelProps = {}) {
  const { allExtractions, fetchAllExtractions } = useFeaturesStore();
  const {
    extractionId,
    setExtraction,
    status,
    selection,
    clearSelection,
    error,
  } = useFeatureGroupsStore();
  const { selectSAE, addFeature } = useSteeringStore();
  const [detailFeatureId, setDetailFeatureId] = useState<string | null>(null);
  const [handoffError, setHandoffError] = useState<string | null>(null);

  useFeatureGroupsWebSocket(extractionId);

  useEffect(() => {
    void fetchAllExtractions(['completed']);
  }, [fetchAllExtractions]);

  const completedExtractions = useMemo(
    () => allExtractions.filter((e) => e.status === 'completed'),
    [allExtractions]
  );

  // Auto-select the most recent completed extraction on first load
  useEffect(() => {
    if (!extractionId && completedExtractions.length > 0) {
      setExtraction(completedExtractions[0].id);
    }
  }, [extractionId, completedExtractions, setExtraction]);

  const currentExtraction = completedExtractions.find((e) => e.id === extractionId);

  const handleSteerSelected = async () => {
    if (!currentExtraction || selection.size === 0) return;
    setHandoffError(null);
    const saeId = currentExtraction.external_sae_id;
    if (!saeId) {
      setHandoffError('This extraction has no linked SAE — steering hand-off unavailable.');
      return;
    }
    try {
      const sae = await getSAE(saeId);
      selectSAE(sae); // clears prior selection
      const layer = currentExtraction.layer_index ?? sae.layer ?? 0;
      let added = 0;
      for (const [featureId, neuronIndex] of selection.entries()) {
        const ok = addFeature({
          feature_idx: neuronIndex,
          layer,
          strength: 10,
          label: null,
          feature_id: featureId,
        });
        if (!ok) break; // max features reached
        added += 1;
      }
      if (added > 0) {
        clearSelection();
        onNavigateToSteering?.();
      }
    } catch (e: any) {
      setHandoffError(e.detail || e.message || 'Failed to hand off to steering');
    }
  };

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Boxes className="w-5 h-5 text-emerald-400" />
          <h1 className="text-lg font-semibold text-slate-100">Feature Groups</h1>
        </div>
        {selection.size > 0 && (
          <button
            onClick={() => void handleSteerSelected()}
            className="flex items-center gap-2 px-3 py-1.5 text-sm bg-emerald-600 hover:bg-emerald-500 text-white rounded-md"
          >
            <Sliders className="w-4 h-4" />
            Steer selected ({selection.size})
          </button>
        )}
      </div>

      <p className="text-sm text-slate-500 mb-4">
        Groups of features that fire on the same top activating token with similar surrounding
        context — candidate concept clusters. Select members and hand them to Steering to
        validate a group's hypothesized meaning.
      </p>

      <div className="mb-4">
        <label className="text-xs text-slate-500 block mb-1">Extraction</label>
        <select
          value={extractionId ?? ''}
          onChange={(e) => setExtraction(e.target.value || null)}
          className="bg-slate-900 border border-slate-700 rounded-md px-3 py-2 text-sm text-slate-200 w-full max-w-xl"
        >
          <option value="">Select a completed extraction…</option>
          {completedExtractions.map((extraction) => (
            <option key={extraction.id} value={extraction.id}>
              {extraction.sae_name || extraction.id}
              {extraction.model_name ? ` — ${extraction.model_name}` : ''}
              {extraction.layer_index != null ? ` (L${extraction.layer_index})` : ''}
            </option>
          ))}
        </select>
      </div>

      {(handoffError || error) && (
        <p className="text-xs text-red-400 mb-3">{handoffError || error}</p>
      )}

      {extractionId && (
        <>
          <ComputeIndexBanner />
          {status?.status === 'completed' && (
            <GroupList onOpenFeature={setDetailFeatureId} />
          )}
        </>
      )}
      {!extractionId && completedExtractions.length === 0 && (
        <div className="text-sm text-slate-500 bg-slate-900 border border-slate-800 rounded-lg p-6 text-center">
          No completed extractions yet. Run a feature extraction from the SAEs or Extractions
          panel first.
        </div>
      )}

      <RelatedFeaturesDrawer onOpenFeature={setDetailFeatureId} />

      {detailFeatureId && (
        <FeatureDetailModal
          featureId={detailFeatureId}
          trainingId={currentExtraction?.training_id ?? null}
          onClose={() => setDetailFeatureId(null)}
        />
      )}
    </div>
  );
}
