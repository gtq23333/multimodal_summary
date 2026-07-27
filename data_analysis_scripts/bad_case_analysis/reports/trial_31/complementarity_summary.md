# Complementarity Summary

## Focus pairs @ K=5

- **Proposed vs Qwen3-VL-Rerank-ImgCap+Link**: miss_jaccard=0.4464, rescue Proposed->Qwen3-VL-Rerank-ImgCap+Link=18, rescue Qwen3-VL-Rerank-ImgCap+Link->Proposed=13, kappa=0.3348
- **Proposed vs Proposed-v2**: miss_jaccard=0.66, rescue Proposed->Proposed-v2=10, rescue Proposed-v2->Proposed=7, kappa=0.6367
- **Proposed vs Layout-Order**: miss_jaccard=0.5536, rescue Proposed->Layout-Order=12, rescue Layout-Order->Proposed=13, kappa=0.47
- **Proposed-v2 vs Qwen3-VL-Rerank-ImgCap+Link**: miss_jaccard=0.5, rescue Proposed-v2->Qwen3-VL-Rerank-ImgCap+Link=14, rescue Qwen3-VL-Rerank-ImgCap+Link->Proposed-v2=12, kappa=0.4348

## Union IR

- **Proposed**: IR@3=0.400, IR@4=0.495, IR@5=0.547, IR@6=0.600, IR@7=0.621
- **Qwen3-VL-Rerank-ImgCap+Link**: IR@3=0.432, IR@4=0.547, IR@5=0.600, IR@6=0.642, IR@7=0.695
- **Proposed+QwenLink**: IR@3=0.600, IR@4=0.695, IR@5=0.737, IR@6=0.789, IR@7=0.810

## Shared hard cases @ K=5 (miss>=3 primary methods)

- Count: 42
  - 2016_G_A194.pdf-b86a... | 图12二分法流程示意图 | miss=5
  - 2016_G_A433.pdf-ace5... | 图10 考虑水流力的情况下受力示意图 | miss=5
  - 2017_G_A156.pdf-526d... | 图12附件3介质最终重建图像 | miss=5
  - 2017_G_B104.pdf-435f... | 图10 附件三任务点分布情况 | miss=5
  - 2017_G_B154.pdf-6b29... | 图14各个任务点处的行动力消耗 | miss=5
  - 2017_G_B447.pdf-bba1... | 图33不合理打包（左），合理打包（右） | miss=5
  - 2023_G_A092.pdf-5db8... | 图9求解入射遮挡区域的流程图 | miss=5
  - 2023_G_A127.pdf-c7af... | 图7:锥形光线离散化示意图 | miss=5
