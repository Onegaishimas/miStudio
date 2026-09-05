# miStudio + miLLM Combined Product Roadmap

## Platform Direction

miStudio and miLLM should evolve as two complementary components of a broader mechanistic AI platform.

**miStudio** is the research and experimentation environment. It is where researchers prepare experimental data, inspect model internals, discover and interpret representations, fit mechanistic interpretability methods, manipulate internal state, compare behavioral effects, investigate interactions among representations, develop causal hypotheses, and accumulate evidence about how a model actually works.

**miLLM** is the instrumented inference environment. It provides the runtime counterpart to miStudio: acquiring the models and interpretability artifacts required for inference, exposing model internals while inference is occurring, applying mechanistic interventions, reporting mechanistic telemetry, and eventually evaluating or controlling model behavior using validated knowledge of internal state.

Both products are intentionally **rich interactive applications**.

Their GUIs are not thin administrative layers over APIs, and they should not become secondary as automation capabilities expand. Mechanistic research is highly exploratory and often visual. Researchers need to see feature activations, compare generations, inspect circuits, manipulate steering strengths, examine representation geometry, explore image regions, navigate temporal state, and investigate unexpected results interactively.

The same is true of miLLM. Instrumented inference benefits from direct visibility into loaded models and SAEs, active steering configurations, runtime behavior, internal representations, interventions, and future mechanistic policies.

The platform should therefore follow a consistent principle:

> **Major capabilities should receive rich native GUI experiences, deterministic programmatic APIs, and appropriate MCP exposure for agents and research harnesses.**

These are complementary interaction modes rather than competing interfaces.

The combined lifecycle is:

```text
Acquire / Construct Data
        ↓
Observe Model Internals
        ↓
Discover Representations
        ↓
Interpret and Characterize Them
        ↓
Experimentally Manipulate Them
        ↓
Discover Interactions and Mechanisms
        ↓
Validate Causal Effects
        ↓
Package Validated Knowledge
        ↓
Deploy to Inference
        ↓
Observe and Control at Runtime
        ↓
Return Runtime Evidence to Research
```

The roadmap should not be organized around any single interpretability method.

Sparse autoencoders are already a mature and important component of miStudio. Jacobian Lens and J-space provide another representation framework. Linear probes, directions, subspaces, population codes, attribution methods, activation patching, and future techniques should all fit into the same broader research model.

The long-term objective is not simply to support more interpretability algorithms.

It is to create a coherent environment in which:

- different methods can discover representations,
- different interventions can manipulate them,
- multiple forms of evidence can be compared,
- human researchers can explore results visually,
- automated workflows and agents can execute the same experiments programmatically,
- validated findings can be operationalized through miLLM.

## 1. Current Platform Foundation

The roadmap begins from a substantial existing platform.

Several capabilities that might otherwise appear as future roadmap items already exist and should instead be treated as foundations.

### 1.1 Mature SAE Research and Experimental Steering

miStudio already provides extensive SAE-based steering designed for experimentation rather than simple feature demonstration.

A researcher can hold a prompt constant while systematically changing the steering strength of an SAE feature and producing a sequence of responses for direct comparison.

```text
Same Prompt

Feature 394 @ -2.0  → Response
Feature 394 @ -1.0  → Response
Feature 394 @  0.0  → Baseline
Feature 394 @ +1.0  → Response
Feature 394 @ +2.0  → Response
Feature 394 @ +3.0  → Response
```

This supports inspection of directionality, thresholds, progressive behavioral change, nonlinear response, saturation, instability, unintended effects, and differences between positive and negative steering.

miStudio also supports **sequential experiments across multiple features**, applying comparable strength sequences independently to candidate representations.

```text
Feature A
 -2  -1   0  +1  +2

Feature B
 -2  -1   0  +1  +2

Feature C
 -2  -1   0  +1  +2
```

miStudio additionally supports **simultaneous multi-feature steering**:

```text
Prompt
  ↓
Feature A   +1.4
Feature B   -0.7
Feature C   +2.0
  ↓
Model
  ↓
Response
```

This matters because representations may reinforce one another, inhibit one another, jointly encode a concept, participate in a broader mechanism, or become behaviorally meaningful only when manipulated together.

The current SAE steering capability therefore establishes four experimental patterns that should inform the broader roadmap:

- **Single-Feature Strength Sweeps** - hold stimulus and representation constant while changing intervention magnitude.
- **Sequential Feature Comparison** - repeat comparable experiments independently across several candidate representations.
- **Simultaneous Multi-Feature Intervention** - manipulate several representations within one inference.
- **Baseline-Preserving Comparison** - retain unsteered outputs for direct comparison.

It also establishes a fifth design principle: **rich visual analysis**. Results should be presented so that behavioral changes remain obvious and easy to compare.

These patterns should later extend to J-space, subspaces, patching, multimodal representations, circuits, and other intervention methods.

### 1.2 Existing Hugging Face Integration

Hugging Face is already a shared artifact-acquisition ecosystem for both products.

#### miStudio

miStudio currently supports:

- dataset download,
- dataset upload,
- SAE download,
- SAE upload,
- model download.

Its relationship with Hugging Face is therefore both acquisitive and publication-oriented.

```text
                 Hugging Face
                ↕           ↕
           Datasets        SAEs
                ↕           ↕
                  miStudio

Hugging Face
      ↓
    Models
      ↓
   miStudio
```

#### miLLM

miLLM supports the same download integrations:

- dataset download,
- SAE download,
- model download.

It does not require upload functionality because publishing artifacts is not a runtime-server responsibility.

```text
Hugging Face
     │
     ├── Models ────────┐
     ├── SAEs ──────────┼──→ miLLM
     └── Datasets ──────┘
```

The architectural distinction is deliberate:

> **Artifact acquisition is shared. Artifact publication primarily belongs to miStudio.**

Future Hugging Face work should focus on exact revision pinning, checksums, provenance, dependency resolution, workflow integration, publication metadata, and reproducibility manifests rather than basic transfer capabilities that already exist.

### 1.3 Existing Neuronpedia Integration

miStudio already publishes:

- SAEs,
- feature activation labels,

to Neuronpedia.

```text
miStudio
   │
   ├── SAE
   └── Feature Activation Labels
            ↓
       Neuronpedia
```

Future work can deepen interoperability through richer metadata, experiment provenance, evidence levels, links back to miStudio experiments, updated interpretations, circuit relationships, and additional artifact types as supported.

The broader principle is:

> **miStudio should connect the research ecosystem rather than attempt to replace it.**

### 1.4 Existing MCP Integration

miStudio and miLLM already share an MCP server, and MCP controls are extended as capabilities are added to either product.

MCP should therefore be treated as an existing platform interface, not as a future architectural layer.

The correct development pattern is:

```text
New Product Capability
        │
        ├── Rich GUI experience
        ├── REST/API support
        └── Appropriate MCP controls
```

MCP is valuable for research agents, coding agents, automation harnesses, reproducible agent-driven experiments, and programmatic orchestration. It complements the GUIs and APIs; it does not replace them.

### 1.5 Jacobian Lens and J-space: Current Development State

Jacobian Lens and J-space represent the next major expansion beyond miStudio's mature SAE subsystem.

**Existing / Active Expansion**

- Jacobian Lens fitting exists.
- J-Lens capability is being actively expanded.
- J-space analysis and characterization are being developed.

**Currently Under Development**

- **J-space swapping is actively being implemented but is not yet a completed capability.**

**Subsequent Roadmap**

- systematic swap experiments,
- J-space intervention sweeps,
- generalized J-space steering,
- combination experiments,
- quantitative causal analysis,
- future runtime intervention through miLLM.

```text
CURRENT

Extensive SAE discovery and analysis
        ↓
Mature SAE steering
        ↓
Single-feature strength sweeps
        ↓
Sequential feature comparison
        ↓
Simultaneous multi-feature steering

CURRENT / ACTIVE DEVELOPMENT

Jacobian Lens fitting
        ↓
J-space analysis
        ↓
J-space characterization

CURRENTLY UNDER DEVELOPMENT

J-space swapping
        ↓
Controlled swap experiments

ROADMAP

General representation interventions
        ↓
Subspaces / patching / heterogeneous mechanisms
        ↓
Validated Intervention Artifacts
        ↓
miLLM runtime execution
```

## 2. Product Experience Architecture

Both products should remain rich interactive applications as the platform grows.

The GUI, API, and MCP interfaces should expose the same underlying capabilities in forms appropriate to their consumers.

```text
                         Human User
                            │
                   ┌────────┴────────┐
                   ▼                 ▼
             miStudio GUI        miLLM GUI
             Research UX         Runtime UX
                   │                 │
                   ▼                 ▼
             miStudio API         miLLM API
                   ▲                 ▲
                   │                 │
                   └────────┬────────┘
                            │
                      Shared MCP Server
                            ▲
                            │
              Agent / Harness / MCP Client
```

**GUI:** exploration, visualization, comparison, hypothesis development, interactive experiment configuration, drill-down analysis.

**API:** scripting, deterministic automation, application integration, custom clients, reproducible pipelines.

**MCP:** semantic agent interaction, research harnesses, AI-assisted experimentation, automated multi-step workflows.

The roadmap should explicitly consider all three surfaces whenever major new functionality is added.

## 3. Unified Representation Framework

**Primary owner: miStudio**  
**Runtime consumer: miLLM**

The most important architectural evolution is to make **Representation** independent of the technique used to discover it.

SAEs provide one view. J-space provides another. A probe may expose decodable information. A subspace may encode a distributed concept. A population code may require many dimensions simultaneously.

The platform therefore needs to distinguish:

> **What representation exists?**

from:

> **How was it discovered?**

A common representation architecture should support:

```text
Model Internal State
        │
        ├── Native Activation Space
        ├── SAE Feature Space
        ├── Jacobian Lens / J-space
        ├── Probe Directions
        ├── Steering Directions
        ├── Learned Subspaces
        ├── Population Codes
        └── Future Representation Systems
```

Representation Objects could include neurons, SAE features, feature groups, vectors, directions, coordinates, bases, clusters, subspaces, distributed states, spatial representations, and temporal representations.

This enables different methods to coexist without forcing every discovery into an SAE-style feature abstraction.

**GUI implication:** provide visualizations appropriate to each representation type. A feature may need activation-context views; a subspace may require geometry plots; visual representations need spatial overlays; temporal representations need time navigation.

**API implication:** Representation objects should have consistent machine-readable schemas.

**MCP implication:** Agents should be able to reason semantically about representation objects regardless of how they were discovered.

## 4. Generalized Intervention Experiment Framework

**Primary owner: miStudio**

The mature SAE steering subsystem provides the blueprint.

Today:

```text
Prompt
   ×
SAE Feature
   ×
Strength
```

Future:

```text
Stimulus
   ×
Representation System
   ×
Representation Object(s)
   ×
Intervention Type
   ×
Magnitude
   ×
Context
```

The Intervention Experiment abstraction should preserve controlled stimuli, explicit baselines, comparable intervention series, systematic strength changes, multiple candidate representations, multi-representation combinations, and direct output comparison.

It should eventually support SAE steering, J-space swapping, J-space steering, activation patching, ablation, clamping, subspace projection, representation replacement, circuit disruption, and circuit enhancement.

**GUI:** remain optimized for comparative experimentation, with baseline and intervention conditions directly comparable.

**API:** intervention experiments should be reproducibly defined and executed.

**MCP:** agents should be able to construct, launch, inspect, compare, and iterate experiments.

## 5. Dataset Studio

**Primary owner: miStudio**

miStudio already supports Hugging Face dataset exchange. Dataset Studio should extend this into a complete experimental-data environment.

Supported sources should include Hugging Face, uploaded files, local files, directories, HTTP, object storage, Git, generated data, and derived miStudio datasets.

Initial formats should include TXT, CSV, JSON, JSONL, Parquet, and Arrow. Later expansion should include images, image/text, audio, audio/text, video, and multimodal records.

### Schema Mapping

Researchers should be able to map arbitrary dataset fields into model input while retaining additional fields as experimental metadata.

Example input schema:

```text
instruction
response
domain
difficulty
quality_score
```

Example model template:

```text
{{instruction}}

{{response}}
```

Remaining fields can support filtering, stratification, comparison, and subsequent research questions.

### Transformation and Composition

Support filtering, deterministic sampling, random sampling, stratified sampling, weighted sampling, concatenation, balancing, oversampling, undersampling, templating, normalization, deduplication, splitting, and modality-specific preprocessing.

Example controlled corpus:

```text
Research Corpus A

General text        50%
Source code         20%
Scientific text     15%
Synthetic examples  10%
Rare examples        5%
```

Dataset composition becomes an experimental variable.

### Lineage

Every derived dataset should retain parent datasets, Hugging Face revision where applicable, transformations, filters, random seeds, weights, schema mapping, preprocessing, record counts, and checksums.

**GUI:** schema inspection, record previews, transformation previews, sample-distribution views, lineage visualization, composition controls.

**API/MCP:** the same operations should be reproducible programmatically.

## 6. Workflow Studio

**Primary owner: miStudio**

Workflow Studio should make end-to-end mechanistic research executable.

Example SAE workflow:

```text
Dataset
   ↓
Tokenization
   ↓
Activation Extraction
   ↓
SAE Training
   ↓
Feature Extraction
   ↓
Activation Sampling
   ↓
Feature Interpretation
   ↓
Steering Experiment
   ↓
Validation
   ↓
Publication
```

Example J-space workflow:

```text
Dataset
   ↓
Activation Capture
   ↓
Jacobian Collection
   ↓
J-Lens Fit
   ↓
J-space Characterization
   ↓
Swap Experiment
   ↓
Validation
```

Each node should define inputs, outputs, configuration, dependencies, resource requirements, retries, caching, checkpoints, and failure policy.

Existing external integrations should be usable as workflow operations:

```text
Hugging Face Dataset
        ↓
Download
        ↓
Transform
        ↓
Train SAE
        ↓
Extract Features
        ↓
Generate Labels
        │
        ├── Publish SAE → Hugging Face
        └── Publish SAE + Labels → Neuronpedia
```

Workflow Studio should remain intentionally focused on mechanistic research rather than becoming a generic workflow product.

**Rich GUI role:** visualize the workflow graph, inspect node configuration and artifacts, open specialist UIs, watch status, view errors, retry stages, branch workflows, compare runs, and reuse artifacts.

**API/MCP:** workflows should also be launchable and inspectable programmatically. Humans and agents operate on the same underlying workflow objects.

## 7. Scheduler and Resource Orchestration

**Primary owner: miStudio**

Workflow Studio determines **what** should execute. The scheduler determines **where and when**.

The scheduler should understand CPU workloads, GPU workloads, GPU memory, system memory, storage, worker capability, job priority, concurrency, dependencies, and checkpoints.

```text
Dataset preprocessing      → CPU Worker
Activation extraction      → GPU 0
SAE training               → GPU 1
J-Lens fitting             → GPU 2
Steering campaign          → GPU 0
Interpretation jobs        → Worker Pool
HF publication             → IO Worker
Neuronpedia publication    → IO Worker
```

Later support should include heterogeneous GPUs, remote workers, Kubernetes, resource reservations, and affinity rules.

**GUI:** queued jobs, dependencies, worker availability, resource utilization, blocked stages, and resource requirements.

**API/MCP:** submit, pause, resume, cancel, reprioritize, and inspect jobs.

## 8. Research Campaigns

**Primary owner: miStudio**

The existing steering-strength sweep is already a specialized research campaign. Research Campaigns should generalize that pattern across many experimental variables.

```text
Models:
  Model A
  Model B

Datasets:
  General
  Code
  Mixed

Layers:
  7
  14
  18

Representation:
  SAE
  J-space

Intervention:
  Baseline
  Low
  Medium
  High
```

miStudio should expand the matrix, schedule required work, retain provenance, aggregate results, and support comparison.

Campaign dimensions could include model, checkpoint, dataset, dataset composition, layer, SAE architecture, feature, feature group, steering strength, J-space representation, prompt class, random seed, and intervention type.

**Rich GUI role:** experiment matrices, filtering, sorting, comparative tables, behavioral metrics, internal-state metrics, response comparisons, interaction plots, outlier identification, and drill-down.

**MCP role:** agents can create campaigns, inspect aggregated results, identify promising experiments, and launch follow-up campaigns.

Automation reduces the cost of executing experiments; visualization reduces the cost of understanding them.

## 9. Feature Interaction and Combination Analysis

**Primary owner: miStudio**

Because simultaneous multi-feature steering already exists, representation interaction should become a dedicated research capability.

Representations may reinforce each other, suppress each other, interact nonlinearly, create threshold effects, matter only jointly, or participate in larger mechanisms.

The roadmap should include pairwise interaction sweeps, selected higher-order combinations, factorial designs, synergy detection, antagonism detection, interaction scoring, candidate feature-group discovery, and combinatorial-search reduction.

**GUI:** rich matrix and interaction visualizations.

**API/MCP:** agents can search large feature-combination spaces programmatically and surface the most interesting interactions for human review.

## 10. Representation Discovery Beyond SAE Features

**Primary owner: miStudio**

### Sparse Autoencoders

Existing capabilities include SAE training, Hugging Face import/export, feature extraction, activation analysis, interpretation, Neuronpedia publishing, steering, strength sweeps, sequential feature comparisons, and simultaneous multi-feature intervention.

Future expansion should focus on automated steering campaigns, feature interactions, quantitative outcome measures, side-effect analysis, and evidence tracking.

### Jacobian Lens / J-space

**Existing / Active Expansion**

- J-Lens fitting,
- fitting characterization,
- J-space analysis,
- representation exploration.

**Currently Under Development**

- J-space swapping.

**Subsequent Expansion**

- controlled swap campaigns,
- systematic candidate swaps,
- J-space steering,
- J-space combination experiments,
- downstream measurements,
- comparison against SAE representations.

Once swapping is implemented, it should inherit the experimental discipline established by SAE steering:

```text
Controlled Stimulus
        ↓
Baseline
        ↓
Swap Condition A
        ↓
Swap Condition B
        ↓
Comparable Outputs
        ↓
Internal + Behavioral Analysis
```

### Probes, Directions, and Subspaces

Additional methods should include linear probes, nonlinear probes, classifier directions, steering directions, concept subspaces, basis discovery, distributed representations, and population codes.

Each should participate in the same GUI/API/MCP interaction model.

## 11. Representation Geometry

**Primary owner: miStudio**

Some representations may be sparse and localized. Others may be genuinely distributed.

Capabilities should include PCA, SVD, ICA, CCA, intrinsic dimensionality, subspace analysis, representational similarity, cross-layer comparison, cross-model comparison, manifold analysis, and population-code analysis.

A concept may turn out to be one SAE feature, several interacting features, a dense direction, a multidimensional subspace, or a population representation.

A recurring multi-feature configuration might eventually become a first-class **Composite Representation** with its own interpretation, geometry, steering profile, validation evidence, and deployment configuration.

**GUI:** dimensionality plots, projection views, layer comparisons, cluster views, subspace overlap, model comparisons.

**MCP:** compare representations across layers, estimate intrinsic dimensionality, identify related subspaces, summarize geometric changes.

## 12. Representation Emergence Analysis

**Primary owner: miStudio**

Researchers should not always need to choose the interesting layer in advance.

```text
Controlled Dataset
      ↓
Capture Multiple Layers
      ↓
SAE / Probe / J-space Analysis
      ↓
Measure Accessibility
      ↓
Identify Transition Region
      ↓
Characterize Geometry
      ↓
Run Intervention Experiments
```

The same approach should operate across training checkpoints, model sizes, architectures, fine-tuning stages, dataset variants, and modality combinations.

**GUI:** emergence curves and layer-by-layer transitions.

**API/MCP:** automate broad sweeps and identify interesting transition regions for human exploration.

## 13. Mechanism and Circuit Discovery

**Primary owner: miStudio**

Discovering a representation does not explain what the model does with it. Mechanism Discovery should therefore broaden beyond homogeneous feature circuits.

Mechanisms may contain SAE features, feature groups, J-space states, directions, subspaces, native activations, and heterogeneous combinations.

```text
SAE Feature Group A
        ↓
J-space Representation B
        ↓
Subspace C
        ↓
Output-Related Representation D
```

Capabilities should include activation correlation, attribution, gradient attribution, mediation, cross-layer dependencies, candidate circuit discovery, mechanism ranking, persistence testing, context comparison, and cross-method evidence.

Existing multi-feature steering already provides useful experimental tests: suppressing one feature to test another, testing whether two representations matter only jointly, and observing whether manipulating an upstream representation alters a predicted downstream representation.

**GUI:** mechanism graphs should be interactive and explorable, with links to evidence, feature views, and intervention experiments.

**MCP:** agents should be able to query mechanism graphs and launch targeted tests.

## 14. Counterfactual and Causal Experimentation

**Primary owner: miStudio**

The roadmap is not to add SAE steering; that already exists extensively. The roadmap is to broaden the experimental intervention vocabulary.

**Existing**

- positive SAE steering,
- negative SAE steering,
- strength sweeps,
- response comparison,
- sequential feature experiments,
- simultaneous multi-feature steering.

**Currently Being Added**

- J-space swapping.

**Future**

- generalized J-space steering,
- activation patching,
- ablation,
- clamping,
- subspace projection,
- subspace removal,
- representation replacement,
- circuit interventions,
- state transplantation.

```text
Controlled Stimulus
       ↓
Baseline
       ↓
Intervention Series
       ↓
Comparable Outputs
       ↓
Internal-State Measurement
       ↓
Behavioral Measurement
       ↓
Hypothesis Comparison
       ↓
Replication
```

A major enhancement should be quantitative measurement alongside visual comparison. Potential metrics include semantic movement, task performance, target behavior, response similarity, downstream feature changes, J-space changes, mechanism-state changes, and side effects.

**GUI:** rich comparison remains central.

**MCP/API:** the same experiments should be reproducible and scalable programmatically.

## 15. Evidence and Validation

**Primary owner: miStudio**

Interpretability findings should carry explicit evidence levels.

```text
Observed
   ↓
Correlated
   ↓
Decoded
   ↓
Localized
   ↓
Attributed
   ↓
Intervened
   ↓
Dose / Response Established
   ↓
Causally Supported
   ↓
Context Validated
   ↓
Replicated
```

Dose/response is especially relevant because miStudio already performs systematic strength sweeps. Multi-feature experiments can establish interaction, dependence, synergy, antagonism, and conditional effects.

Evidence should persist with the representation or mechanism rather than only exist in individual experiment screens.

**GUI:** researchers should be able to inspect the evidence trail visually.

**MCP/API:** agents should be able to reason over evidence levels and choose appropriate follow-up experiments.

## 16. Multi-Representation Evidence Fusion

**Primary owner: miStudio**

Different methods should be able to corroborate one another.

```text
Hypothesis: Concept X

SAE representation        strong
J-space representation    strong
Linear probe              strong
Attribution               moderate
SAE steering              causal support
J-space swap              causal support
Cross-prompt replication  confirmed
Cross-dataset replication confirmed
```

miStudio should aggregate agreement, disagreement, robustness, intervention results, and replication.

## 17. Multimodal Dataset and Model Support

**Research owner: miStudio**  
**Runtime owner: miLLM**

The platform should progressively become modality-agnostic:

```text
Language
Vision
Video
Audio
Multimodal Models
World Models
Scientific Foundation Models
```

Multimodal support should reuse the same Representation, Intervention, Evidence, Workflow, GUI, API, and MCP abstractions.

### 17.1 Vision and VLMs

miStudio should support image datasets, image/text datasets, image preprocessing, vision encoder instrumentation, image-patch activations, SAE training over visual representations, spatial localization, representation geometry, VLM adaptor analysis, cross-modal tracking, and interventions.

```text
Image
  ↓
Vision Encoder
  ↓
Projection / Adapter
  ↓
Language Model
  ↓
Response
```

**GUI:** image display, activation overlays, patch highlighting, layer navigation, feature comparisons, intervention comparison.

**MCP:** agents can automate analysis without replacing visual exploration.

### 17.2 Video and Temporal Models

Video introduces persistent state. Relevant representations may include motion, velocity, trajectory, object permanence, interaction, state transitions, and future state.

miStudio should support video datasets, temporal preprocessing, spatial-temporal activation capture, motion analysis, latent-state tracking, temporal geometry, and counterfactual intervention.

**GUI:** playback, timeline navigation, activation trajectories, spatial-temporal overlays, frame comparison, intervention comparison.

**MCP:** agents can identify interesting frames, states, or transitions for human inspection.

### 17.3 Audio

Audio introduces representations across time and frequency. Relevant concepts may include phonemes, words, pitch, speaker identity, prosody, acoustic events, and semantic content.

miStudio should support WAV/FLAC, Hugging Face audio datasets, custom datasets, resampling, waveform analysis, spectrograms, audio encoder instrumentation, temporal localization, frequency localization, representation discovery, and intervention.

**GUI:** waveform views, spectrograms, activation overlays, transcript alignment, time navigation, intervention comparison.

**miLLM:** eventually provide instrumented audio-language inference with an equally rich runtime interface.

## 18. Generalized Activation Context

**Primary implementation: miStudio**  
**Shared runtime abstraction: miLLM**

The platform cannot permanently assume:

```text
activation ↔ token
```

Instead:

```text
Activation Context
   ├── token span
   ├── image patch
   ├── image region
   ├── audio interval
   ├── time/frequency region
   ├── video frame range
   ├── spatial-temporal region
   ├── latent state
   └── multimodal alignment
```

This context should be shared by activation observations, representations, labels, interventions, mechanisms, and evidence.

This generalization should happen relatively early because multimodal GUI experiences depend on it as much as the underlying analysis does.

## 19. Internal-State and World-Model Research

**Primary owner: miStudio**

As temporal and multimodal support matures, dynamic internal variables can become explicit research targets: position, velocity, acceleration, object identity, object permanence, trajectories, spatial relationships, temporal relationships, causality, uncertainty, and expected future state.

```text
Controlled Observation
        ↓
Candidate Internal Variable
        ↓
Representation Discovery
        ↓
Localization
        ↓
Geometry
        ↓
Interaction Discovery
        ↓
Steering / Swap / Patch
        ↓
Counterfactual Prediction
        ↓
Measure Expected Change
```

The strength-sweep experimental philosophy is especially important for continuous internal variables.

## 20. White-Box Evaluation

**Evaluation development: miStudio**  
**Runtime execution: miLLM**

Traditional evaluation:

```text
Input
  ↓
Output
  ↓
Score
```

Future mechanistic evaluation:

```text
Input
  ↓
Internal Representations
  ↓
Internal Mechanism
  ↓
Output
  ↓
Evaluation
```

miStudio should validate rules such as expected representation present, prohibited representation present, expected feature group active, required mechanism missing, known failure mechanism active, and internal state inconsistent with output.

**miStudio GUI:** inspect evaluation evidence, triggering internal states, associated mechanisms, false positives, false negatives, contextual behavior.

**miLLM GUI:** active evaluations, triggered conditions, interventions, runtime effects, policy status.

**MCP/API:** agents and automated systems can query and act on evaluation results.

## 21. miLLM as the Instrumented Runtime Environment

**Primary owner: miLLM**

miLLM should remain a rich interactive inference product.

Its GUI should expand with capabilities around Hugging Face model acquisition, SAE acquisition, model loading, SAE attachment, prompt/response inference, steering, multi-feature steering, future J-space intervention, intervention artifacts, telemetry, white-box evaluations, and mechanistic policies.

**miStudio** is optimized for discovery, rich experimentation, comparison, visualization, research campaigns, mechanism discovery, and validation.

**miLLM** is optimized for instrumented inference, runtime observation, artifact loading, validated intervention execution, telemetry, and runtime control.

Both remain full interactive products.

## 22. Intervention Artifacts

**Authoring and validation: miStudio**  
**Resolution and execution: miLLM**

Validated interventions should become portable artifacts.

```text
Intervention Artifact

Model:
  Hugging Face model/revision

Representation System:
  SAE

SAE:
  Hugging Face SAE/revision

Representations:
  Feature 394   +1.4
  Feature 821   -0.6
  Feature 2388  +0.9

Applicable Context:
  ...

Validation Campaign:
  experiment-1843

Evidence:
  dose-response confirmed
  interaction confirmed
  cross-prompt replication confirmed

Expected Effect:
  ...

Known Side Effects:
  ...
```

miLLM can resolve dependencies directly through Hugging Face.

```text
Intervention Artifact
        ↓
Resolve Model
        ↓
Resolve SAE
        ↓
Load Intervention
        ↓
Instrumented Inference
```

**GUI:** miStudio provides artifact-authoring and validation views; miLLM provides artifact-loading, status, configuration, and runtime-effect views.

**MCP:** agents can select, deploy, enable, disable, and inspect validated interventions.

## 23. Conditional Mechanistic Control

**Policy authoring: miStudio**  
**Policy enforcement: miLLM**

Static steering is useful. Conditional intervention is more powerful.

```text
WHEN:
  internal state X is detected

AND:
  context Y applies

THEN:
  Feature A +1.2
  Feature B -0.7
  Feature C +0.4

UNTIL:
  condition Q
```

Policies may reference SAE features, feature groups, J-space states, subspaces, circuits, multimodal representations, and combinations of signals.

miStudio should provide rich policy-design and simulation experiences. miLLM should provide rich runtime policy visibility and control. MCP should allow agents and harnesses to manage validated policies programmatically.

## 24. Runtime Evidence Feedback

The research/runtime relationship should be bidirectional.

miLLM may reveal new contexts, unexpected internal states, intervention side effects, model drift, mechanism failures, and new candidate interactions.

```text
       miStudio
 Research / Validation
          │
          │ validated artifact
          ▼
        miLLM
 Runtime / Inference
          │
          │ telemetry
          ▼
       miStudio
 Refine / Replicate / Retest
```

The miStudio GUI should make runtime evidence explorable. MCP should allow automated follow-up experiments to be created from runtime findings.

## 25. Reproducible Research Packages

**Primary owner: miStudio**

Research packages should contain or reference:

```text
Dataset manifests
Hugging Face dataset references
Dataset transformations

Model manifests
Hugging Face model references

Workflow definitions
Activation configurations

SAE artifacts
Hugging Face SAE references
Feature definitions
Feature labels
Neuronpedia references

Steering experiments
Strength sweeps
Multi-feature configurations
Feature-interaction results

J-Lens artifacts
J-space definitions
Swap experiments

Probes
Directions
Subspaces
Representation geometry

Mechanisms
Circuits
Interventions
Validation evidence

Metrics
Provenance
README
```

Another installation should be able to:

```text
Import
  ↓
Resolve Dependencies
  ↓
Inspect
  ↓
Reproduce
  ↓
Extend
```

The GUI should provide rich browsing of package contents and lineage rather than treating the package as an opaque archive.

## 26. Ownership and Interaction Model

The core ownership distinction remains:

> **miStudio owns research, experimentation, visualization, validation, publication, and mechanistic artifact authoring.**

> **miLLM owns artifact acquisition for inference, rich instrumented runtime interaction, mechanistic observation, validated intervention execution, and policy enforcement.**

> **The shared MCP server exposes appropriate capabilities from both products to agents and research harnesses without displacing their native GUIs.**

| Capability | miStudio | miLLM | MCP |
|---|---|---|---|
| Hugging Face dataset download | Exists | Exists | Expose as useful |
| Hugging Face dataset upload | Exists | Not required | miStudio control |
| Hugging Face SAE download | Exists | Exists | Expose |
| Hugging Face SAE upload | Exists | Not required | miStudio control |
| Hugging Face model download | Exists | Exists | Expose |
| Neuronpedia SAE upload | Exists | Not required | miStudio control |
| Neuronpedia label upload | Exists | Not required | miStudio control |
| SAE training | Exists / Primary | | Expose |
| SAE interpretation | Exists / Rich GUI | | Expose |
| SAE steering | Exists / Extensive GUI | Runtime | Expose |
| Strength sweeps | Exists / Extensive GUI | | Expose |
| Sequential feature experiments | Exists | | Expose |
| Simultaneous multi-feature steering | Exists | Runtime | Expose |
| J-Lens fitting | Exists / Active Expansion | | Extend MCP |
| J-space analysis | Active Development | Future runtime | Extend MCP |
| J-space swapping | Under Development | Future runtime | Add as implemented |
| Dataset Studio | Roadmap / Rich GUI | Consumer | Expose |
| Workflow Studio | Roadmap / Rich GUI | Supporting | Expose |
| Research Campaigns | Roadmap / Rich GUI | | Expose |
| Representation Geometry | Roadmap / Rich GUI | | Expose analysis |
| Emergence Analysis | Roadmap / Rich GUI | | Expose |
| Mechanism Discovery | Primary / Rich GUI | Observe | Expose |
| Generalized Interventions | Primary / Rich GUI | Execute validated | Expose |
| Runtime telemetry | Analyze | Primary / Rich GUI | Query/stream |
| White-box evaluation | Design / Validate | Execute / Display | Expose |
| Conditional policy | Author / Simulate | Enforce / Display | Manage |
| VLM research | Primary / Rich GUI | Runtime / Rich GUI | Expose |
| Video research | Primary / Rich GUI | Runtime | Expose |
| Audio research | Primary / Rich GUI | Runtime | Expose |

## 27. Proposed Delivery Progression

### Release 1 - Dataset Engineering and Workflow Foundation

**miStudio:** extend existing dataset support with custom ingestion, schema mapping, transformation, composition, lineage; add visual Workflow Studio, templates, scheduling, retry/resume, Hugging Face workflow nodes, and Neuronpedia publication nodes.

**MCP/API:** expose workflow and dataset operations as they are added.

**Outcome:** complete experiments become reproducible research objects that can be operated interactively or programmatically.

### Release 2 - Unified Representation and Intervention Framework

Introduce Representation System, Representation Object, Representation Group, Intervention, and Intervention Experiment. Preserve specialized SAE GUI experiences while moving them onto generalized underlying abstractions. Continue integrating J-Lens, J-space, probes, directions, and subspaces.

**Outcome:** new interpretability methods can share experimentation, visualization, API, and MCP infrastructure.

### Release 3 - Research Campaigns and Interaction Analysis

Add automated feature sweeps, steering-strength campaigns, multi-feature matrices, interaction studies, model comparisons, layer comparisons, dataset comparisons, and behavioral metrics.

**GUI:** rich campaign dashboards and drill-down analysis.

**MCP:** agents can launch campaigns and identify experiments worth human inspection.

**Outcome:** miStudio becomes capable of systematic research programs rather than only individual experiments.

### Release 4 - Representation Geometry and Emergence

Add PCA/SVD/CCA, intrinsic dimensionality, subspace analysis, population codes, layer comparison, emergence analysis, and checkpoint analysis.

**GUI:** purpose-built geometry and emergence visualizations.

**Outcome:** the workbench expands beyond sparse features into arbitrary representation structures.

### Release 5 - J-space Swapping and Generalized Causal Experimentation

Complete the active J-space swapping work and integrate it into the shared Intervention Experiment framework. Then add swap campaigns, J-space intervention comparison, activation patching, ablation, subspace manipulation, representation replacement, quantitative causal metrics, and evidence tracking.

**GUI:** rich baseline/swap comparison and internal-effect views.

**MCP:** expose swap and causal experiment operations once implemented.

**Outcome:** the experimental methodology already mature in SAE steering expands into representation-agnostic causal experimentation.

### Release 6 - Research-to-Runtime Integration

**miStudio:** add Intervention Artifacts, validated feature groups, runtime compatibility metadata, policy authoring, and visual artifact-validation workflows.

**miLLM:** add artifact loading, dependency resolution, runtime SAE intervention expansion, J-space runtime support, structured telemetry, and rich runtime artifact status.

**MCP:** expose deployment and runtime controls.

**Outcome:** validated experiments can move cleanly into instrumented inference without diminishing either product's GUI experience.

### Release 7 - Vision and VLM Support

**miStudio:** image datasets, vision activation capture, spatial context, visual SAE research, image-region overlays, VLM adaptor analysis, cross-modal tracking, intervention comparison.

**miLLM:** instrumented VLM inference, rich visual runtime state, compatible interventions.

**Outcome:** the mechanistic research environment becomes visually multimodal.

### Release 8 - Video and Temporal State

**miStudio:** video datasets, temporal preprocessing, spatial-temporal context, motion analysis, latent-state tracking, temporal geometry, visual timeline exploration, counterfactual interventions.

**miLLM:** extend runtime support for compatible temporal/world models.

**Outcome:** miStudio becomes capable of studying representations evolving through time.

### Release 9 - Audio

**miStudio:** audio datasets, waveform and spectral preprocessing, audio-model instrumentation, time/frequency context, waveform/spectrogram visualization, speech and acoustic representation analysis, interventions.

**miLLM:** instrumented audio-language inference with rich runtime visualization.

**Outcome:** the platform becomes truly modality-agnostic.

### Release 10 - White-Box Evaluation and Conditional Mechanistic Control

**miStudio:** internal-state evaluation authoring, mechanism assertions, failure-signature research, policy simulation, validation.

**miLLM:** runtime white-box evaluation, conditional interventions, mechanistic policy enforcement, rich policy status and telemetry.

**MCP:** expose validated policy and evaluation controls.

**Outcome:** mechanistic research becomes operational evaluation and runtime control.

### Release 11 - Integrated Multimodal and World-Model Research

**miStudio:** cross-modal representations, modality transitions, multimodal mechanisms, temporal causal structures, learned state variables, world-model workflows, scientific foundation-model analysis.

**miLLM:** compatible multimodal observation and intervention.

**Outcome:** the platform can investigate learned systems independent of whether their primary modality is text, vision, audio, video, or dynamic latent state.

## 28. Long-Term Platform Architecture

The mature architecture is best understood as **two rich products with multiple complementary access paths**.

```text
                         HUMAN EXPERIENCE

            ┌─────────────────────────────────┐
            │                                 │
            ▼                                 ▼
   ┌──────────────────┐              ┌──────────────────┐
   │     miStudio     │              │      miLLM       │
   │                  │              │                  │
   │   Rich Research  │              │  Rich Inference  │
   │       GUI        │              │       GUI        │
   └────────┬─────────┘              └────────┬─────────┘
            │                                 │
            │        APIs / SERVICES          │
            │                                 │
            └──────────────┬──────────────────┘
                           │
                    Shared MCP Server
                           ▲
                           │
                  Agents / Harnesses
```

External ecosystems remain connected directly:

```text
                    Hugging Face
                 /       |       \
            Models      SAEs    Datasets
             ↓  ↓        ↓  ↓      ↓  ↓
         miStudio       miLLM


                  miStudio
                /          \
               ▼            ▼
        Hugging Face    Neuronpedia
        Dataset + SAE   SAE + Labels
         Publishing      Publishing
```

This avoids a false hierarchy. MCP does not sit above the products. APIs do not replace GUIs. GUIs are not merely views onto automation. Each interface serves a different mode of research and operation.

## 29. Strategic End State

The combined platform ultimately supports seven related functions.

### Discover

Identify internal representations through SAEs, J-space, probes, directions, subspaces, population representations, and future methods.

### Experiment

Manipulate representations individually and in combination. This is already a significant miStudio strength through its rich SAE steering environment.

### Understand

Determine what representations encode, where they occur, when they emerge, what geometry they occupy, how they respond to intervention magnitude, and how they interact with other representations.

### Relate

Discover dependencies and mechanisms connecting representations across layers, contexts, models, and modalities.

### Validate

Use dose/response steering, multi-feature intervention, swapping, ablation, patching, representation replacement, and replication to establish which mechanistic hypotheses have meaningful causal support.

### Deploy

Package validated representations, feature groups, interventions, mechanisms, and policies into portable artifacts. Where appropriate, large external dependencies should remain referenced through Hugging Face.

### Observe and Control

Use miLLM to acquire required models and SAEs, run instrumented inference, expose internal state interactively, apply validated interventions, enforce mechanistic policies, provide rich runtime telemetry, and return evidence to miStudio.

```text
Discover Representation
        ↓
Inspect It Visually
        ↓
Interpret It
        ↓
Manipulate It
        ↓
Sweep Intervention Strength
        ↓
Combine It With Other Representations
        ↓
Measure Interaction Effects
        ↓
Identify Candidate Mechanism
        ↓
Test Counterfactuals
        ↓
Validate and Replicate
        ↓
Publish / Reference Supporting Artifacts
        ↓
Package Mechanistic Knowledge
        ↓
Deploy Through miLLM
        ↓
Observe Runtime State
        ↓
Return Evidence to miStudio
```

Humans and agents should both be able to participate throughout this lifecycle.

A human researcher may discover an unexpected interaction visually and then launch a large automated campaign. An agent may execute a broad search through MCP and surface the most interesting results for visual investigation. A runtime anomaly observed in miLLM may become the starting point for a new miStudio experiment.

The interaction modes reinforce each other.

**miStudio is a rich interactive laboratory for discovering, visualizing, manipulating, comparing, relating, and validating model internals.**

**miLLM is a rich interactive instrumented inference environment for observing and operationalizing mechanistic capabilities during model execution.**

**Their APIs provide deterministic programmatic access, while the shared MCP server allows agents and research harnesses to participate in the same workflows without diminishing the role of either product's GUI.**

The platform should strengthen all of these modes together:

> **automation expands what can be done; rich interfaces expand what can be understood.**
