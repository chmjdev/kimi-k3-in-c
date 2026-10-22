# Lab01 experiment — 26 September 2026

Source baseline:1c7803907bff24eb9916259417c1be1e5afdf831. Built with the native Linux toolchain on dual Xeon Silver4214 (24physical/48logical CPUs),187GiB RAM. Engine/fixtures live at `/var/tmp/binteca-kimi-experiment/source`. This is an isolated executable experiment, not a deployed chat service. Existing lab services are preserved.

`make -j8` and `OMP_NUM_THREADS=4 make test` passed the supplied reference oracle:32/32teacherforcing,20/20generatedspan,20/20greedy,20/20incremental; state-reuse logits bit-identical. Tokenizer parity subsequently passed45/45cases using the official tokenizer/config at publisher revision f831ab66814297da540d832a5235f8e904f29d06; no full weight shards were downloaded. These fixtures represent a small13layer model, not full Kimi checkpoint inference.

## Measured CPU tuning

Three runs per thread setting, five repetitions inside each kernel benchmark, `OMP_PROC_BIND=spread OMP_PLACES=cores`,native optimized build. Median kernel milliseconds:

| Threads | BF16 matrix | MXFP4 matrix |
|---:|---:|---:|
|4|7.83|1.45|
|12|4.00|0.86|
|16|3.91|0.75|
|24|3.63|0.40|
|48|3.62|0.35|

Use the exact medians in data/lab01-kernel-threads.json if rounding differs.24threads is a conservative starting point;48offers little BF16 benefit and a smaller MXFP4 gain while using SMT. Compared with4threads,24threads significantly improves these kernel times. This is NOT measured full-model tokens/second; storage streaming may dominate. Output hashes are preserved in the results. No floating-point reassociation or kernel arithmetic changes were made.

`scripts/run-lab01.sh MODEL TRUNK [options]` supplies24threads with spread/core binding,incremental decoding,24GBtrunk/1GBexpertcache. The trunk-first memory allocation is a provisional starting point based on upstream cache findings, not validated full-model tuning. Additional CLI options may override budgets. It avoids the roughly128GBserverpreset while other loaded services currently leave about85GiBavailable. Check fresh memory and disk before a real run.

## Storage gate

Host free space measured1,617,245,503,488bytes. Checkpoint1,560,936,091,448bytes plus roughly109GBtrunk exceeds this before operational headroom. User selected an additional localSSD/NVMevolume. No unused volume is currently attached: sda/sdb belong to existingRAID1root/swap. Do not repurpose those disks.

Attach and mount a separate volume; an empty2TBvolume may meet the budget, while larger capacity gives more room. Target must have at least1,719,936,091,448freebytes under the new default preflight (checkpoint+109GBtrunk+50GBreserve). No bulk weights download started.

The downloader now checks checkpoint+trunk+reserve using Python integer disk accounting (avoids awk scientific notation and shell overflow). `K3_TRUNK_BYTES=0` is allowed only when the trunk is planned on a separately sized filesystem; `K3_RESERVE_BYTES` overrides the default50GBreserve. Three automated budget boundary tests live in tests/test_download_budget.py.

After the volume is available, resolve/pin the publisher revision, download and checksum all shards, pack the trunk there, then run short correctness and latency comparisons with repeated trials. Only then choose memory budgets/threads and add a supervised native service through JCDS if an appropriate serving layer is implemented. Never report the checkpoint as running based only on fixture tests.
