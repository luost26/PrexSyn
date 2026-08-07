# Paper and v1 differences

!!! important "Need the exact paper features?"
    Use the [`dev-v0` branch](https://github.com/luost26/prexsyn/tree/dev-v0). Its environment, APIs, data, and checkpoints differ from v1.

The [PrexSyn paper](https://arxiv.org/abs/2512.00384) describes the original research system. The current v1 code keeps the projection architecture and C++ data engine but does not expose every paper feature.

| Area | Paper | Current v1 |
| --- | --- | --- |
| Chemical-space projection | ECFP4-conditioned generation and ranking | Supported; benchmark script included |
| Descriptor conditioning | ECFP4, FCFP4, BRICS fragments, and physicochemical descriptors | ECFP4 and FCFP4 registered |
| Composite property queries | AND, OR, and NOT composition | Deprecated |
| Property-conditioned generation | Included | Deprecated |
| Molecular optimization | Query-space optimization and paper benchmarks | Standard GuacaMol tasks supported with the ECFP4 genetic sampler; composite-query tasks remain deprecated |
| Data engine | Multithreaded C++ generation and detokenization | Supported by PrexSyn Engine 1.1 |

The maintained v1 reproduction paths cover [chemical-space projection](chemical-space-projection.md) and the [standard molecular-optimization benchmark](molecular-optimization.md).
