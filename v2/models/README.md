# Inference model — main_round2

歌声データベース: **波音リツ（歌唱・提供: カノン）**. [MODEL_TERMS.md](../../MODEL_TERMS.md) applies to these weights; the code MIT license does not license them.

Source training-checkpoint SHA-256: `c7456443968249a263ccc18a9553f5f921b41502a96b72f0b0dc75a103123dde`.
Distribution inference-package SHA-256: `661c77f6d9fd93e9546dc1b291d3bb720b40cb8d7cfd70be1e9f370f7b836fab` (also in SHA256SUMS.txt).

Only model tensors, inference configuration and source SHA are retained. No optimizer, history, training order, RNG, manifest or source snapshots. Tensor values are unchanged. Architecture is specialized to the source model; arbitrary other checkpoint configurations are unsupported. Stage 3 acceptance criteria remain unmet. No additional voice weights are included. Singing/speech training tools are documented in the v2 guide.

Inactive self-oscillating-source tension tensors are omitted; every retained inference tensor is unchanged.

