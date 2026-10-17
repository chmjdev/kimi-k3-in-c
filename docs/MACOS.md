# Apple Silicon preparation

The local repository is `~/Projects/Binteca/AILab/models/kimik3dev`, with origin
`git@github.com:chmjdev/kimi-k3-in-c.git`. Apple Silicon support is on
`feat/apple-silicon-metal-neon`.

Install Apple's command line tools and Homebrew, then run from the repository:

```sh
./scripts/prepare-mac.sh
```

The script installs Homebrew's `libomp` if needed, builds `bin/k3` with native ARM
tuning, NEON and Metal support, then runs the existing tests with synthetic
fixtures. It does not download weights or run the real Kimi checkpoint.

## Verified on 2026-10-17

Local hardware probes reported Apple M5 Pro, `Mac17,8`, 48 GiB unified memory
and 18 logical CPUs. Xcode's Clang and Homebrew `libomp` were present.
The repository volume had approximately 293 GiB available when checked.

`make -j6 all` and `make -j6 test` passed. The synthetic full-model oracle
matched its reference: 32/32 teacher-forcing positions and 20/20 generated
positions, with state reuse and incremental decode passing.

Metal tests detected Apple M5 Pro and passed CPU/GPU parity for BF16, FP32 and
MXFP4 matrix operations. Maximum absolute differences were `6.705523e-08`,
`1.862645e-08` and `2.145767e-06`, respectively.

Tokenizer parity did not run because the real checkpoint's `tiktoken.model`
was absent. Real checkpoint generation and throughput remain unverified.

## Full checkpoint storage

The project's full setup requires about 1.56 TB for the checkpoint and 109 GB
for the packed trunk, roughly 1.7 TB total. The checked internal volume cannot
hold that setup. A model storage location with sufficient free space must be
selected before downloading or packing weights. No external location has been
selected or verified.

Inference and benchmarks remain on lab01 under the estate policy; the local
preparation above validates the build and synthetic correctness tests.
