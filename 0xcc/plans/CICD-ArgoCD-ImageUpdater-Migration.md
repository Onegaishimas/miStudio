# Plan: Adopt miLLM-style CI/CD for miStudio (selective rebuilds + ArgoCD Image Updater)

**Status:** IN EXECUTION (2026-07-15). User approved the recommended = matches-miLLM options. Manifests
authored by 3 parallel agents, cross-consistency-verified, and reconciled against the live cluster.

**Decisions locked (all "same as miLLM"):**
1. ArgoCD manifest source = **the private source repo `Onegaishimas/miStudio`** @ `gitops/pilot`, path `k8s/base`
   — this OVERRIDES the earlier §5 recommendation of the mirror: inspecting the live `millm` Application showed
   it watches the *source* repo, so miStudio follows the same proven pattern. The mirror only builds images.
2. Full Kustomize base (`k8s/base/`).
3. celery/mcp share the backend image — one backend build rolls all 4 refs (accepted).
4. `k8s_deploy` kept as documented break-glass.
5. ArgoCD prune OFF during burn-in.

**Blocker found & resolved during cutover:** the `mistudio` namespace had **no `mistudio-secrets` Secret** —
the live pods had been manually mutated to inline plaintext env values, while both the GitOps base and the
original single-file manifest reference `secretKeyRef: mistudio-secrets`. Created `mistudio-secrets` from the
live values (5 keys: database-url, database-url-sync, secret-key, mcp-auth-token, postgres-password). The
first ArgoCD sync now performs a **safe, value-preserving swap** of inline plaintext → secretKeyRef (identical
values), rolling backend/mcp/postgres pods once.

**Cluster facts confirmed live (2026-07-15):** ArgoCD + argocd-image-updater operational; `millm` Application
Synced/Healthy is the working blueprint; CRD-based `ImageUpdater` resource is REQUIRED (this cluster only
reconciles apps selected by one); shared `dockerhub-creds` + `regcred` exist in `argocd` ns; `mistudio-repo-creds`
+ `gitops/pilot` branch do NOT exist yet and are created during cutover.

## Goal

Replicate miLLM's GitOps pipeline for miStudio so that: push code → **only the changed image(s) rebuild** →
ArgoCD Image Updater pins the new digest in git → ArgoCD auto-syncs → pods roll. This retires the manual
`source scripts/k8s-helpers.sh; k8s_check; k8s_deploy` step.

## Current state (miStudio) — migrating FROM

- **No selectivity:** `.github/workflows/docker-images.yml` (runs in the public mirror `hitsainet/miStudio`)
  rebuilds **both** backend (~9 min CUDA) and frontend on **every** `main` push — no `paths:` filter.
- **`:latest` tags only** → no digest change → deploy needs manual node `docker pull` + `rollout restart`.
- **Manual deploy:** `scripts/k8s-helpers.sh :: k8s_deploy` SSHes to `192.168.244.61`, pulls `:latest`,
  `kubectl apply -f` a **host-local copy** `/home/sean/app/k8s-mistudio.hitsai.local/mistudio-deployment.yaml`,
  and `rollout restart`. **No ArgoCD, no Helm** — raw single-file manifest.
- **Two-repo split:** source `Onegaishimas/miStudio` (has `k8s/`, `scripts/`, CLAUDE.md) → `sync-to-clean.yml`
  strips private files, force-pushes to public `hitsainet/miStudio` (builds images, cuts releases).
  `k8s/` is NOT stripped, so the manifest exists in both repos.
- **5 Deployments** in ns `mistudio`: postgres, redis, `mistudio-backend` (3 containers: api/celery-worker/
  celery-beat), `mistudio-mcp` (reuses backend image), `mistudio-frontend`. **Backend image appears 4× across
  2 Deployments** — all must update together on a backend rebuild. Backend `strategy: Recreate` (single GPU).
- **2 CI images:** `hitsai/mistudio-backend`, `hitsai/mistudio-frontend` (Docker Hub). No separate mcp image.

## Target state (from miLLM) — migrating TO

miLLM's proven pattern (files at `/home/x-sean/app/miLLM/`):
- **Selective build** (`.github/workflows/docker-images.yml`): a **`detect` job** does `git diff
  before→SHA`, emits a JSON array of changed images; a **`build` job matrix** builds only those. Path→image:
  backend ← `millm/|Dockerfile|docker-entrypoint.sh|pyproject.toml|alembic`; admin-ui ← `admin-ui/`. Empty
  diff → `images=[]` → nothing builds. Fallback: no usable base → build everything (never silently skip).
- **Tags:** always `:latest` **plus** `:<sha8>` (or `:vX.Y.Z` on tags). Image Updater tracks `:latest` **by digest**.
- **Kustomize, not Helm:** `k8s/base/` with an `images:` block; Image Updater writes digest pins to a sibling
  `.argocd-source-<app>.yaml` parameter-override file on the deploy branch.
- **Deploy branch:** `sync-main-to-gitops-pilot.yml` merges `main → gitops/pilot` (preserving the pin file);
  ArgoCD `Application` watches `gitops/pilot` path `k8s/base`, `syncPolicy.automated{selfHeal:true}` (prune
  off during burn-in), `ServerSideApply=true`, and `ignoreDifferences` on the restart annotation.
- **Image Updater** (CRD variant): `ImageUpdater/millm-images` + Application annotations —
  `image-list: backend=…:latest, admin-ui=…:latest`, `update-strategy: digest`,
  `kustomize.image-name` per alias, `write-back-method: git`, `git-branch: gitops/pilot`.

## Migration design for miStudio (concrete)

1. **Rewrite `hitsainet/miStudio`'s `docker-images.yml`** to the detect+matrix pattern. Path→image:
   - `mistudio-backend` ← diff touches `backend/**`. (Backend image also powers mcp + celery — one build covers all 4 uses.)
   - `mistudio-frontend` ← diff touches `frontend/**`.
   - Since the mirror is **force-pushed** by sync-to-clean (like miLLM), keep the orphan-SHA re-fetch + the
     "no base → build all" fallback verbatim from miLLM's detect job.
   - Keep tag scheme `:latest` + `:<sha8>` (Image Updater needs the moving `:latest` to poll + the digest to pin).
2. **Introduce Kustomize** `k8s/base/` (convert the single `mistudio-deployment.yaml`): split into resource
   files + `kustomization.yaml` with an `images:` block for `hitsai/mistudio-backend` and
   `hitsai/mistudio-frontend`, both `newTag: latest`. The backend image's 4 references all resolve through
   the one `images[]` entry — Kustomize rewrites every occurrence, solving the "update 4 places" problem.
3. **Deploy branch:** add `gitops/pilot` + a `sync-main-to-gitops-pilot.yml` (adapted from miLLM) preserving
   `k8s/base/.argocd-source-mistudio.yaml`.
4. **ArgoCD manifests** `k8s/argocd/mistudio-app.yaml`: AppProject `mistudio`, Application watching the
   chosen repo @ `gitops/pilot` path `k8s/base`, automated+selfHeal (prune off initially), + Image Updater
   annotations/CR tracking `hitsai/mistudio-backend:latest` and `hitsai/mistudio-frontend:latest` by digest,
   git write-back to `gitops/pilot`.
5. **Cluster one-time:** confirm ArgoCD + argocd-image-updater installed on `mcs-lnxgpu01`; create the
   `argocd` repo-credential Secret (PAT w/ `contents:write` for the manifest repo), the `regcred` Docker Hub
   pull secret, and the `gitops/pilot` branch. Keep `mistudio-secrets` as-is.
6. **Cutover + rollback runbook:** first sync with prune off; verify Image Updater writes a pin on the next
   backend push; retire (or demote to break-glass) `k8s_deploy`. Rollback = revert the pin commit or scale
   the Application's auto-sync off and `k8s_deploy` manually.

## §5 — Open decisions for the user (blockers before execution)

1. **Which repo is the ArgoCD manifest source** — the public mirror `hitsainet/miStudio` (where images build,
   already public, natural for a burn-in) or the private `Onegaishimas/miStudio`? Image Updater's git
   write-back needs `contents:write` on that repo. **Recommendation:** the mirror `hitsainet/miStudio`
   (matches miLLM's separation of concerns; the k8s manifest already syncs there).
2. **Kustomize migration vs kustomize-image-only** — full Kustomize base (recommended, matches miLLM, clean)
   vs minimal (keep the raw manifest, add just a kustomization wrapper with an `images:` transform).
3. **Selectivity granularity for celery/mcp** — they share the backend image, so backend rebuild = they all
   roll. Confirm that's acceptable (it's the same as today).
4. **Retire `k8s_deploy` fully, or keep as break-glass?** Recommendation: keep it, documented as emergency-only.
5. **Prune** — leave ArgoCD prune off during burn-in (recommended), enable later.

## Suggested execution (once approved) — parallelizable

- **Agent A:** author the new `docker-images.yml` (detect+matrix) + `sync-main-to-gitops-pilot.yml`.
- **Agent B:** convert `k8s/mistudio-deployment.yaml` → `k8s/base/` Kustomize (resources + kustomization + images).
- **Agent C:** write `k8s/argocd/mistudio-app.yaml` (AppProject + Application + ImageUpdater CR + annotations).
- **Then (serial, needs cluster):** install/verify ArgoCD + Image Updater, create secrets + `gitops/pilot`,
  first sync with prune off, verify an end-to-end selective build → pin → auto-sync, write the runbook.

## Verification

- Push a frontend-only change → confirm **only** `mistudio-frontend` builds (backend job skipped).
- Confirm Image Updater commits a digest pin to `.argocd-source-mistudio.yaml` on `gitops/pilot`.
- Confirm ArgoCD auto-syncs and only the frontend Deployment rolls; app healthy at `k8s-mistudio.hitsai.local`.
- Push a backend change → backend (+ mcp + celery) roll together; frontend untouched.

## Key reference files (miLLM blueprint)

`/home/x-sean/app/miLLM/.github/workflows/docker-images.yml` · `.../sync-main-to-gitops-pilot.yml` ·
`/home/x-sean/app/miLLM/k8s/argocd/millm-app.yaml` · `/home/x-sean/app/miLLM/k8s/base/{kustomization,backend,frontend}.yaml` ·
(pin file `k8s/base/.argocd-source-millm.yaml` exists only on `origin/gitops/pilot`).
