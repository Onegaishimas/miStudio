"""
What miStudio publishes must be loadable by whoever downloads it.

Spec §2.2 says a consumer reads `payload["J"]` and that "absence of `J` raises
with the offending key list". This project wrote the BARE `{layer: matrix}` form,
so every artifact it has ever produced would have failed to load for anyone who
downloaded it — invisible locally, because our own reader accepted the bare form
and nothing else ever read one. Publishing them would have shipped files nobody
could open.

That is why `normalise_payload` (which taught the reader both shapes) had to land
before this: changing what we EMIT is only safe once everything already on disk
still reads.

MUTATION CONTROLS (each must turn this file red):
  * emit the bare map again              -> "a consumer can LOAD what we emit"
  * omit d_model from the wrapper        -> "carries the fields a consumer reads"
  * derive source_layers from anything
    other than the keys                  -> "source_layers EQUALS the key set"
  * ship validation.json                 -> "our LOCAL VERDICT does not travel"
  * drop the deferred wording from the
    model card                           -> "the card says what was NOT checked"
  * publish a staged artifact            -> "an UNPUBLISHED artifact is refused"
  * return no revision                   -> "the commit sha is recorded"
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.services.jlens_acquire_service import (
    PUBLISHED_FILES,
    model_card,
    publish_artifact,
    published_path,
)
from src.services.jlens_artifact_service import (
    JLensArtifactService,
    normalise_payload,
)


class TestWhatWeEmitIsConformant:
    def test_a_consumer_can_LOAD_what_we_emit(self, tmp_path):
        """The reference consumer reads `payload["J"]` and raises without it.

        This is the whole reason the emitted format changed: a bare map is
        readable here and nowhere else, so every lens this project published
        would have been a file nobody could open.
        """
        service = JLensArtifactService(tmp_path)
        ref = service.write_staged(
            "org/model",
            {l: torch.zeros(8, 8) for l in range(3)},
            "model: org/model\n",
            n_prompts=634,
        )
        raw = torch.load(ref.lens_path, map_location="cpu", weights_only=True)
        assert "J" in raw, (
            "the emitted checkpoint has no 'J'; a conformant consumer raises "
            f"with the offending key list {sorted(raw)}"
        )
        assert sorted(raw["J"]) == [0, 1, 2]

    def test_it_carries_the_fields_a_consumer_reads(self, tmp_path):
        """`d_model` is the one other field read WITHOUT a fallback (§2.2)."""
        service = JLensArtifactService(tmp_path)
        ref = service.write_staged(
            "org/model",
            {l: torch.zeros(8, 8) for l in range(3)},
            "model: org/model\n",
            n_prompts=634,
        )
        raw = torch.load(ref.lens_path, map_location="cpu", weights_only=True)
        assert raw["d_model"] == 8
        assert raw["n_prompts"] == 634

    def test_source_layers_EQUALS_the_key_set(self, tmp_path):
        """A1 requires equality, and `normalise_payload` refuses a file where
        the two disagree — so emitting a stale list would make our own reader
        reject our own artifact."""
        service = JLensArtifactService(tmp_path)
        ref = service.write_staged(
            "org/model",
            {l: torch.zeros(4, 4) for l in (0, 3, 7)},
            "model: org/model\n",
        )
        raw = torch.load(ref.lens_path, map_location="cpu", weights_only=True)
        assert raw["source_layers"] == [0, 3, 7]
        assert sorted(normalise_payload(raw)) == [0, 3, 7]

    def test_OUR_OWN_READER_still_accepts_it(self, tmp_path):
        """The change is safe only because both shapes read. If this regressed,
        every artifact in the registry would become unreadable at once."""
        service = JLensArtifactService(tmp_path)
        ref = service.write_staged(
            "org/model", {0: torch.zeros(4, 4)}, "model: org/model\n"
        )
        assert sorted(service._load_payload(ref)) == [0]  # noqa: SLF001

    def test_mismatched_widths_are_refused(self, tmp_path):
        """A consumer reads one `d_model` without a fallback; there is no
        honest value to write for a lens whose matrices disagree."""
        service = JLensArtifactService(tmp_path)
        with pytest.raises(ValueError, match="d_model"):
            service.write_staged(
                "org/model",
                {0: torch.zeros(4, 4), 1: torch.zeros(8, 8)},
                "model: org/model\n",
            )


class TestTheLayoutMatchesTheSpec:
    def test_the_published_path_is_the_conformant_one(self):
        """`<model>/jlens/<dataset>/`, so a consumer that already resolves
        published lenses finds ours without being told anything new."""
        assert published_path("google/gemma-2-2b-it") == "gemma-2-2b-it/jlens/mistudio"
        assert (
            published_path("Qwen/Qwen3-8B", "wikitext") == "qwen3-8b/jlens/wikitext"
        )

    def test_our_LOCAL_VERDICT_does_not_travel(self):
        """`validation.json` records two classes as DEFERRED because they need a
        live external consumer and have never run anywhere. Shipping it invites
        a reader to take this installation's verdict for the lens's own."""
        assert "validation.json" not in PUBLISHED_FILES
        assert "acquisition.json" not in PUBLISHED_FILES
        assert PUBLISHED_FILES == ("config.yaml",)


class TestTheModelCardIsHonest:
    def test_the_card_says_what_was_NOT_checked(self):
        """A green suite here does not mean interoperability proven, and a
        reader who assumes it does is reading something never measured."""
        card = model_card(
            "org/m",
            {"n_prompts": 634, "converged": True},
            {
                "results": [
                    {
                        "check": "cross_implementation",
                        "status": "deferred",
                        "detail": "needs a live consumer",
                    }
                ]
            },
        )
        assert "deferred" in card
        assert "not a pass" in card

    def test_the_card_forbids_porting_bands(self):
        """BR-002, restated for whoever downloads this. The published boundaries
        were measured on one model, and porting them is the error this project
        makes impossible by construction locally — a README is the only place
        that constraint can travel."""
        card = model_card("org/m", {}, None)
        assert "Band boundaries" in card
        assert "must not be inferred" in card

    def test_the_card_states_the_checkpoint_shape(self):
        """So a consumer knows what to expect without opening it."""
        card = model_card("org/m", {}, None)
        assert '"J"' in card and "weights_only=True" in card


class TestPublishing:
    @staticmethod
    def _artifact(tmp_path):
        directory = tmp_path / "m"
        directory.mkdir()
        torch.save({"J": {0: torch.zeros(2, 2)}, "d_model": 2}, directory / "m_jacobian_lens.pt")
        (directory / "config.yaml").write_text("model: org/m\nn_prompts: 500\n")
        (directory / "validation.json").write_text('{"passed": true}')
        (directory / "acquisition.json").write_text('{"source": {}}')
        return directory

    def test_it_uploads_ONLY_the_conformant_files(self, tmp_path):
        directory = self._artifact(tmp_path)
        captured = {}

        def fake_upload(folder_path, repo_id, path_in_repo, commit_message):
            captured["files"] = sorted(p.name for p in __import__("pathlib").Path(folder_path).iterdir())
            captured["path"] = path_in_repo
            return types.SimpleNamespace(oid="deadbeef")

        api = MagicMock()
        api.upload_folder = fake_upload
        with patch("huggingface_hub.HfApi", return_value=api):
            out = publish_artifact(directory, "org/m", "you/lenses", "tok")

        assert captured["files"] == ["README.md", "config.yaml", "m_jacobian_lens.pt"], (
            captured["files"]
        )
        assert "validation.json" not in captured["files"]
        assert "acquisition.json" not in captured["files"]
        assert captured["path"] == "m/jlens/mistudio"
        assert out["revision"] == "deadbeef"

    def test_the_COMMIT_SHA_is_recorded(self, tmp_path):
        """The uploader this follows returns `commit_hash: None` with a comment
        saying it "would need to get [it] from the API response". For an
        artifact whose whole purpose is portable evidence, "published at X"
        without a revision names a moving target."""
        directory = self._artifact(tmp_path)
        api = MagicMock()
        api.upload_folder = lambda **kw: types.SimpleNamespace(oid="abc123")
        with patch("huggingface_hub.HfApi", return_value=api):
            out = publish_artifact(directory, "org/m", "you/lenses", "tok")
        assert out["revision"] == "abc123"
        assert "abc123" in out["url"]

    def test_a_directory_with_TWO_lenses_is_refused(self, tmp_path):
        from src.services.jlens_acquire_service import AcquisitionRefused

        directory = self._artifact(tmp_path)
        torch.save({"J": {0: torch.zeros(2, 2)}}, directory / "other_jacobian_lens.pt")
        api = MagicMock()
        with patch("huggingface_hub.HfApi", return_value=api):
            with pytest.raises(AcquisitionRefused, match="exactly one"):
                publish_artifact(directory, "org/m", "you/lenses", "tok")
