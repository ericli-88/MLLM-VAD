# M³VAD Architecture and Research Decisions

## Baseline audit

The preserved baseline has seven stages:

1. `image_captioner.py` generates BLIP-2 captions for sampled frames.
2. `create_index.py` embeds candidate captions with ImageBind text features and stores them in per-video FAISS inner-product indexes.
3. `image_text_caption_cleaner.py` samples a centered visual clip and retrieves the most image-aligned caption for each sampled frame.
4. `llm_anomaly_scorer.py` first summarizes the cleaned caption sequence with Llama 2, then prompts Llama 2 for one anomaly score per sampled center frame.
5. `create_summary_index.py` embeds temporal summaries with ImageBind text features.
6. `video_text_score_refiner.py` uses ImageBind video features to retrieve similar summaries and their raw anomaly scores.
7. `eval.py` applies a similarity-weighted score combination, repeats each sampled score over the configured frame interval, and computes frame-level ROC-AUC/PR-AUC.

The baseline temporal summary is text-only: Llama 2 summarizes frame captions. It does not directly process video. Its FAISS indexes are static per-video indexes, not chronological Dynamic Memory.

## M³VAD data flow

M³VAD keeps BLIP-2 but adds direct video understanding. For a target frame `t`, `build_centered_window` converts five seconds on each side to frame indices using the verified source FPS. At 30 FPS this is exactly `[t-150, t+150]`, inclusive, with boundary clipping. `uniform_indices` samples across the entire bounded interval and includes both endpoints. The default 16 frames reduce OneVision token and memory cost while preserving temporal coverage.

The inference order is strict:

1. Obtain Caption and 10-second Summary.
2. Compute current visual, Caption, and Summary embeddings.
3. Call the Base Scorer without Dynamic Memory.
4. Retrieve only Trusted entries whose frame index is lower than `t`.
5. Compute program-owned Normality Support.
6. Call the Memory-Augmented Scorer when references exist.
7. Save the result.
8. Only then evaluate the current frame for Memory admission.

This prevents the current target from serving as its own reference. Ground-truth labels are available only in the later evaluation process.

## Representation spaces

- Visual representation: a normalized weighted fusion of ImageBind's target-image and local-video embeddings. The default target/clip weights are 0.4/0.6.
- Caption representation: normalized Sentence Transformer embedding of the BLIP-2 caption.
- Summary representation: normalized Sentence Transformer embedding of the OneVision summary.
- Reliability alignment: ImageBind text embeddings are compared with the ImageBind visual representation because these modalities share ImageBind's aligned space.

Caption and Summary vectors are compared only with vectors produced by the same text encoder. They are never directly compared with ImageBind visual vectors.

The initial retrieval weights are visual `0.45`, Caption `0.20`, and Summary `0.35`. This emphasizes direct visual/video evidence while giving temporal action semantics more influence than a single-frame caption. These are research defaults, not fitted parameters. They must be ablated or tuned on a separate validation set, never on final UCF-Crime test labels.

## Retrieval

Multimodal retrieval has two stages:

1. ImageBind visual similarity creates a scene-aware candidate pool through FAISS.
2. Visual, Caption, and Summary cosine similarities re-rank candidates using normalized configurable weights.

Greedy temporal diversity filtering prevents several nearly overlapping windows from occupying all Top-K positions. Returning fewer than `K` is valid when Memory is empty, small, or lacks temporally diverse entries. The text-only ablation bypasses visual candidate selection and uses Caption/Summary similarity only.

## Normality Support

Normality Support is a deterministic bounded support score, not a calibrated probability. Cosine similarities are transformed from `[-1,1]` to `[0,1]` only to create bounded evidence-strength inputs. The current formula is:

```text
support = 0.45 * mean combined-similarity support
        + 0.20 * cross-modality agreement
        + 0.20 * memory coverage
        + 0.15 * query/reference reliability
```

All components and retrieved examples are logged. Probability calibration would require an independent labeled validation set and a separately evaluated calibration method.

## Memory contamination controls

Legacy mode directly trusts every candidate with final score `<= 0.2` for faithful ablation.

Robust mode requires:

- final anomaly score `<= 0.2`;
- Base Score `<= 0.2`;
- adequate ImageBind visual/Caption and visual/Summary alignment;
- stable recent target scores;
- no explicit Caption/Summary evidence conflict;
- adequate aggregate reliability;
- no near-duplicate trusted normal pattern.

Failed low-score candidates remain Pending and cannot be retrieved. Stable deterministic IDs make retries idempotent. Entries can be removed explicitly, and capacity management prunes low-reliability Pending entries before removing redundant Trusted entries.

## Persistence and isolation

Every video has an independent Memory directory. Metadata, bound Caption/Summary text, scores, reliability metrics, and embeddings are saved together by Memory ID. FAISS indexes have a saved ordered ID mapping. Loads validate video identity, metadata/embedding ID equality, embedding dimensions, FAISS vector counts, and FAISS ID ordering.

## Resource considerations

LLaVA-OneVision's Hugging Face implementation pools each video frame to a fixed visual token sequence, but cost still grows with the number of sampled frames. The 7B checkpoint is roughly 16 GB of model files before runtime activations. The 0.5B checkpoint is suitable for pipeline development; 4-bit loading and Flash Attention are optional deployment choices and must be validated on the target CUDA stack.

ImageBind is reused to minimize architectural drift from the baseline. A stronger temporal encoder can be substituted behind `VisualEncoder`, but it should be compared through the existing ablation interface and must not be trained on final test labels.
