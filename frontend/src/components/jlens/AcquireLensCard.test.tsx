/**
 * Downloading a published lens, and publishing ours.
 *
 * The two prerequisites this card exists to make visible BEFORE a fetch:
 * a lens is unusable without its model's weights (validating one means reading
 * out through it), and a file with no `config.yaml` beside it cannot have its
 * weight identity checked — the pairing then rests on the operator's assertion,
 * and the artifact records that it does.
 *
 * MUTATION CONTROLS (each must turn this file red):
 *   * send the branch instead of the resolved sha -> "pins the RESOLVED commit"
 *   * drop the weights-missing warning            -> "warns BEFORE the fetch"
 *   * render every candidate as identity-checkable -> "marks a config-less file"
 *   * enable publish with no artifact             -> "publish is refused"
 *   * drop the what-does-not-travel note          -> "says the local verdict"
 *   * preview on the acquire button               -> "looks BEFORE it fetches"
 */

import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';

vi.mock('../../api/jlens', () => ({
  jlensApi: { previewRepo: vi.fn(), acquire: vi.fn(), publish: vi.fn() },
}));

import { jlensApi } from '../../api/jlens';
import { AcquireLensCard } from './AcquireLensCard';

const PREVIEW = {
  repo_id: 'org/lenses',
  // NOT `main`. The card must send this back, or the acquisition names a
  // moving target.
  revision: 'abc123def4567890',
  candidates: [
    {
      path: 'gemma/jlens/wikitext/gemma_jacobian_lens.pt',
      size_bytes: 265_429_252,
      has_config: true,
      has_convergence: true,
      fits_envelope: true,
      envelope_detail: 'within a full fit',
    },
    {
      path: 'loose_lens.pt',
      size_bytes: 1024,
      has_config: false,
      has_convergence: false,
      fits_envelope: true,
      envelope_detail: null,
    },
  ],
};

function mount(over: Partial<React.ComponentProps<typeof AcquireLensCard>> = {}) {
  return render(
    <AcquireLensCard
      modelId="m_1"
      modelRepoId="google/gemma-2-2b-it"
      weightsPresent
      hasArtifact
      {...over}
    />,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  (jlensApi.previewRepo as ReturnType<typeof vi.fn>).mockResolvedValue(PREVIEW);
  (jlensApi.acquire as ReturnType<typeof vi.fn>).mockResolvedValue({
    task_id: 'acq-12345678',
  });
  (jlensApi.publish as ReturnType<typeof vi.fn>).mockResolvedValue({
    task_id: 'pub-12345678',
  });
});

const open = () => fireEvent.click(screen.getByRole('button', { name: /Browse/ }));

describe('acquiring', () => {
  it('looks BEFORE it fetches', async () => {
    mount();
    open();
    fireEvent.click(screen.getByRole('button', { name: /Look inside/ }));
    await waitFor(() =>
      expect(jlensApi.previewRepo as ReturnType<typeof vi.fn>).toHaveBeenCalled(),
    );
    // AND NOTHING WAS DOWNLOADED. A mistyped path must cost a request, not a
    // multi-gigabyte fetch and a slot on the single-GPU queue.
    expect(jlensApi.acquire as ReturnType<typeof vi.fn>).not.toHaveBeenCalled();
  });

  it('pins the RESOLVED commit, not the branch', async () => {
    mount();
    open();
    fireEvent.click(screen.getByRole('button', { name: /Look inside/ }));
    await waitFor(() => screen.getByText(/2 candidates/));
    fireEvent.click(screen.getAllByRole('radio')[0]);
    fireEvent.click(screen.getByTestId('jlens-acquire-run'));

    await waitFor(() =>
      expect(jlensApi.acquire as ReturnType<typeof vi.fn>).toHaveBeenCalled(),
    );
    const sent = (jlensApi.acquire as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(sent.revision).toBe('abc123def4567890');
    expect(sent.path_in_repo).toBe('gemma/jlens/wikitext/gemma_jacobian_lens.pt');
    expect(sent.repo_id).toBe('org/lenses');
  });

  it('warns BEFORE the fetch when the weights are missing', () => {
    mount({ weightsPresent: false });
    open();
    const warning = screen.getByTestId('jlens-acquire-weights-missing');
    expect(warning).toHaveTextContent(/not downloaded/);
    expect(warning).toHaveTextContent(/google\/gemma-2-2b-it/);
  });

  it('does NOT warn when the weights are present', () => {
    mount();
    open();
    expect(screen.queryByTestId('jlens-acquire-weights-missing')).toBeNull();
  });

  it('marks a config-less file as identity-UNVERIFIED', async () => {
    mount();
    open();
    fireEvent.click(screen.getByRole('button', { name: /Look inside/ }));
    await waitFor(() => screen.getByText(/2 candidates/));

    const rows = screen.getAllByRole('listitem');
    // The one WITH a config reads differently from the one without — a single
    // shared label would tell the operator nothing.
    expect(within(rows[0]).getByText('identity checkable')).toBeInTheDocument();
    expect(within(rows[1]).getByText('unverified')).toBeInTheDocument();
  });

  it('explains what UNVERIFIED costs, once one is chosen', async () => {
    mount();
    open();
    fireEvent.click(screen.getByRole('button', { name: /Look inside/ }));
    await waitFor(() => screen.getByText(/2 candidates/));
    fireEvent.click(screen.getAllByRole('radio')[1]);
    expect(screen.getByText(/rests on your assertion/)).toBeInTheDocument();
  });

  it('cannot acquire until a file is chosen', async () => {
    mount();
    open();
    fireEvent.click(screen.getByRole('button', { name: /Look inside/ }));
    await waitFor(() => screen.getByText(/2 candidates/));
    expect(screen.getByTestId('jlens-acquire-run')).toBeDisabled();
    fireEvent.click(screen.getAllByRole('radio')[0]);
    expect(screen.getByTestId('jlens-acquire-run')).toBeEnabled();
  });
});

describe('publishing', () => {
  const toPublish = () =>
    fireEvent.click(screen.getByRole('button', { name: /^Publish$/ }));

  it('is REFUSED when there is no published artifact', () => {
    mount({ hasArtifact: false });
    open();
    toPublish();
    // A TARGET REPO IS SUPPLIED FIRST. Without it the button is disabled for a
    // DIFFERENT reason, and the assertion below passes whether or not the
    // artifact check exists — a fixture agreeing by construction, which is
    // exactly how the mutation survived the first version of this test.
    fireEvent.change(screen.getByTestId('jlens-acquire-repo'), {
      target: { value: 'you/lenses' },
    });
    expect(screen.getByTestId('jlens-publish-no-artifact')).toHaveTextContent(
      /staged artifact is not published/,
    );
    expect(screen.getByTestId('jlens-publish-run')).toBeDisabled();
  });

  it('IS enabled once both the repo and an artifact are present', () => {
    // The positive control. Without it, "disabled" could be permanent.
    mount({ hasArtifact: true });
    open();
    toPublish();
    fireEvent.change(screen.getByTestId('jlens-acquire-repo'), {
      target: { value: 'you/lenses' },
    });
    expect(screen.getByTestId('jlens-publish-run')).toBeEnabled();
  });

  it('says the local verdict does NOT travel', () => {
    mount();
    open();
    toPublish();
    // A reader of the published repo must not take this installation's verdict
    // for the lens's own — two of its checks have never been run anywhere.
    //
    // ASSERTED ON THE NEGATION ITSELF. Matching only "local validation verdict
    // does" survives a rewrite that says it DOES travel, which is the exact
    // sentence being guarded against.
    const note = screen.getByTestId('jlens-publish-note');
    expect(note).toHaveTextContent(/does\s*not\s*travel/i);
    expect(note).toHaveTextContent(/never been run/);
    expect(note).not.toHaveTextContent(/verdict does travel/i);
  });

  it('sends the corpus segment it was given', async () => {
    mount();
    open();
    toPublish();
    fireEvent.change(screen.getByTestId('jlens-acquire-repo'), {
      target: { value: 'you/lenses' },
    });
    fireEvent.change(screen.getByTestId('jlens-publish-dataset'), {
      target: { value: 'wikitext' },
    });
    fireEvent.click(screen.getByTestId('jlens-publish-run'));
    await waitFor(() =>
      expect(jlensApi.publish as ReturnType<typeof vi.fn>).toHaveBeenCalled(),
    );
    const sent = (jlensApi.publish as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(sent.target_repo).toBe('you/lenses');
    expect(sent.dataset).toBe('wikitext');
    expect(sent.model_id).toBe('m_1');
  });

  it('cannot publish without a target repo', () => {
    mount();
    open();
    toPublish();
    expect(screen.getByTestId('jlens-publish-run')).toBeDisabled();
  });
});

describe('the token field', () => {
  it('is masked by default and can be revealed', () => {
    mount();
    open();
    const field = screen.getByTestId('jlens-acquire-token');
    expect(field).toHaveAttribute('type', 'password');
    fireEvent.click(screen.getByRole('button', { name: /Show token/ }));
    expect(screen.getByTestId('jlens-acquire-token')).toHaveAttribute(
      'type',
      'text',
    );
  });

  it('is not sent when left empty, so the configured one is used', async () => {
    mount();
    open();
    fireEvent.click(screen.getByRole('button', { name: /Look inside/ }));
    await waitFor(() =>
      expect(jlensApi.previewRepo as ReturnType<typeof vi.fn>).toHaveBeenCalled(),
    );
    const sent = (jlensApi.previewRepo as ReturnType<typeof vi.fn>).mock
      .calls[0][0];
    expect(sent.access_token).toBeUndefined();
  });
});
