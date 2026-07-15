# miStudio CI/CD Runbook (GitOps: selective builds + ArgoCD Image Updater)

**Live since:** 2026-07-15. Mirrors miLLM's pipeline. Replaces the manual `k8s_deploy` (now break-glass only).

## The flow (what happens on a normal push)

```
push to Onegaishimas/miStudio main
  → sync-to-clean.yml            mirrors main → hitsainet/miStudio (strips 0xcc/, scripts/, CLAUDE.md, …)
  → sync-main-to-gitops-pilot.yml merges main → gitops/pilot (source repo), preserving the pin file
  → docker-images.yml (mirror)   detect+matrix: builds ONLY changed images (backend←backend/, frontend←frontend/)
                                 tags :latest + :<sha8>, pushes to Docker Hub with provenance/sbom/attestation
  → ArgoCD Image Updater         sees the new :latest digest, commits a pin to
                                 gitops/pilot:k8s/base/.argocd-source-mistudio.yaml
  → ArgoCD Application `mistudio` syncs k8s/base from gitops/pilot → pods roll to the pinned digest
```

Nothing to run by hand. Push code, wait ~3 min (frontend) / ~9 min (backend) for the build, then a minute or two for the pin + sync.

## Key facts

| Thing | Value |
|---|---|
| ArgoCD Application | `mistudio` (ns `argocd`), project `mistudio` |
| Deploy branch | `gitops/pilot` (source repo `Onegaishimas/miStudio`), path `k8s/base` |
| Images | `hitsai/mistudio-backend`, `hitsai/mistudio-frontend` (Docker Hub) |
| Backend image reuse | 4 refs (backend api/celery-worker/celery-beat + mcp) collapse through one kustomize `images:` entry |
| Pin file | `k8s/base/.argocd-source-mistudio.yaml` (Image-Updater-owned; never hand-edit on main) |
| Sync policy | `automated.selfHeal: true`, `ServerSideApply=true`. **Prune OFF** (burn-in). |
| ImageUpdater CR | `mistudio-images` — REQUIRED; this cluster only reconciles apps a CR selects |
| Secrets (argocd ns) | `mistudio-repo-creds` (git write-back PAT), shared `dockerhub-creds`, `regcred` |
| Secrets (mistudio ns) | `mistudio-secrets` (database-url, database-url-sync, secret-key, mcp-auth-token, postgres-password) |

## Common operations

**Check status:**
```bash
source scripts/k8s-helpers.sh
k8s "kubectl get application mistudio -n argocd -o custom-columns=SYNC:.status.sync.status,HEALTH:.status.health.status,REV:.status.sync.revision"
```

**Force a sync (rarely needed — auto-sync handles it):**
```bash
k8s "kubectl annotate application mistudio -n argocd argocd.argoproj.io/refresh=hard --overwrite"
```

**See the current digest pin:**
```bash
git show origin/gitops/pilot:k8s/base/.argocd-source-mistudio.yaml
```

**Manual full image rebuild** (e.g. base image CVE): `workflow_dispatch` the docker-images.yml in `hitsainet/miStudio` — a dispatch has no diff base → builds everything.

## Rollback

- **Bad image:** revert the pin — `git revert` the Image Updater commit on `gitops/pilot` (or reset the pin file to the previous digest) and push; ArgoCD syncs back.
- **Bad manifest:** revert the offending commit on main; the sync workflow carries it to gitops/pilot.
- **Pipeline wedged / emergency:** break-glass = `source scripts/k8s-helpers.sh; k8s_deploy` still works (applies `k8s/mistudio-deployment.yaml` + rollout). Note: with `selfHeal: true`, ArgoCD will fight manual drift — scale the Application's auto-sync off first: `kubectl patch application mistudio -n argocd --type merge -p '{"spec":{"syncPolicy":{"automated":null}}}'`, fix, then re-enable.

## Gotchas learned during cutover (2026-07-15)

1. **`value:` → `valueFrom:` transition needs `Replace=true` once.** The pre-existing pods had secret-backed env inlined as plaintext `value:`; the GitOps base uses `valueFrom: secretKeyRef`. SSA can't clear the old `value:`, so the first sync needed a one-time `Replace=true` sync-option (added, synced, then removed). If you ever re-adopt imperatively-managed resources with conflicting literal fields, do the same.
2. **`mistudio-secrets` must exist before first sync** — it didn't (pods ran inlined plaintext). It was seeded from the live pod values. If you rotate any of those 5 keys, update both the Secret and the running services.
3. **Enabling prune:** once burn-in is done, add `prune: true` under `syncPolicy.automated` in `k8s/argocd/mistudio-app.yaml`, commit, and re-apply the app.
4. **`.local` ingress DNS:** backend pods can't resolve `k8s-mistudio.hitsai.local`; the Deployment carries a `hostAliases` entry (preserved from the original manifest) — keep it.

## Retiring the old single-file manifest

`k8s/mistudio-deployment.yaml` is now superseded by `k8s/base/`. It's kept as the `k8s_deploy` break-glass target. When retiring, update `scripts/k8s-helpers.sh :: k8s_deploy` to point at `kubectl apply -k k8s/base` (or delete it entirely once GitOps is trusted).
