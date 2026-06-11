### CORRECT% by Model × Budget (pooled over all tasks and contexts)

| Model / Budget | FullContext | KiaOmni-G ✦ | KiaOmni-σ8 ✦ | BlockSal | Ada-SnapKV | H2O | RealSnapKV |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen2.5-7B** B=98 | *47.5%* | **25.6%** | 24.7% | 24.7% | 20.3% | 19.2% | 11.7% |
| **Qwen2.5-7B** B=128 | *47.8%* | 27.5% | **28.9%** | 26.9% | 23.1% | 20.6% | 13.1% |
| **Qwen2.5-7B** B=256 | *46.7%* | 34.2% | **37.5%** | 36.9% | 30.3% | 25.0% | 22.2% |
| **Qwen2.5-7B** B=512 | *47.5%* | **42.5%** | 41.4% | 39.7% | 36.1% | 31.7% | 30.3% |

| Model / Budget | FullContext | KiaOmni-G ✦ | KiaOmni-σ8 ✦ | BlockSal | Ada-SnapKV | H2O | RealSnapKV |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Mistral-7B** B=98 | *45.8%* | 21.7% | 20.8% | **22.8%** | 20.8% | 21.1% | 18.3% |
| **Mistral-7B** B=128 | *45.6%* | **25.0%** | 22.8% | 22.2% | 23.1% | 20.6% | 15.6% |
| **Mistral-7B** B=256 | *45.8%* | **31.9%** | 29.7% | **31.9%** | 26.4% | 21.9% | 17.8% |
| **Mistral-7B** B=512 | *45.8%* | **37.2%** | 34.7% | 32.8% | 25.8% | 25.0% | 21.1% |

| Model / Budget | FullContext | KiaOmni-G ✦ | KiaOmni-σ8 ✦ | BlockSal | Ada-SnapKV | H2O | RealSnapKV |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Falcon3-7B** B=98 | *41.2%* | 16.1% | **17.4%** | 14.2% | 14.8% | 14.5% | 11.3% |
| **Falcon3-7B** B=128 | *41.6%* | **20.6%** | 19.0% | 15.2% | 17.1% | 20.0% | 14.8% |
| **Falcon3-7B** B=256 | *41.0%* | **25.8%** | 24.8% | 24.8% | 24.5% | 19.7% | 15.5% |
| **Falcon3-7B** B=512 | *40.6%* | **33.9%** | 33.5% | 31.6% | 27.4% | 26.1% | 17.7% |

| Model / Budget | FullContext | KiaOmni-G ✦ | KiaOmni-σ8 ✦ | BlockSal | Ada-SnapKV | H2O | RealSnapKV |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BioMistral-7B** B=98 | *57.3%* | 38.8% | 37.6% | 32.9% | **41.2%** | 35.3% | 19.6% |
| **BioMistral-7B** B=128 | *57.3%* | **47.1%** | 46.7% | 42.7% | 46.7% | 43.9% | 25.9% |
| **BioMistral-7B** B=256 | *57.3%* | 50.2% | 52.5% | 53.7% | **54.1%** | 50.6% | 41.2% |
| **BioMistral-7B** B=512 | *57.3%* | **56.5%** | 55.3% | **56.5%** | **56.5%** | 55.7% | 52.9% |

### CORRECT% by Model × Context Length (pooled over all tasks and budgets)

| Model / Context | FullContext | KiaOmni-G ✦ | KiaOmni-σ8 ✦ | BlockSal | Ada-SnapKV | H2O | RealSnapKV |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen2.5-7B** 4K | *42.7%* | 29.8% | **31.0%** | 28.7% | 26.5% | 23.5% | 19.0% |
| **Qwen2.5-7B** 8K | *49.2%* | 32.9% | 32.9% | **33.1%** | 27.5% | 24.6% | 19.8% |
| **Qwen2.5-7B** 16K | *50.2%* | 34.6% | **35.4%** | 34.4% | 28.3% | 24.2% | 19.2% |

| Model / Context | FullContext | KiaOmni-G ✦ | KiaOmni-σ8 ✦ | BlockSal | Ada-SnapKV | H2O | RealSnapKV |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Mistral-7B** 4K | *41.9%* | **27.9%** | 26.2% | 25.0% | 23.8% | 21.9% | 17.5% |
| **Mistral-7B** 8K | *44.8%* | **28.5%** | 26.0% | 27.5% | 24.0% | 22.1% | 18.3% |
| **Mistral-7B** 16K | *50.6%* | **30.4%** | 28.7% | 29.8% | 24.4% | 22.5% | 18.8% |

| Model / Context | FullContext | KiaOmni-G ✦ | KiaOmni-σ8 ✦ | BlockSal | Ada-SnapKV | H2O | RealSnapKV |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Falcon3-7B** 4K | *37.5%* | **21.9%** | 21.5% | 18.1% | 18.3% | 16.7% | 13.1% |
| **Falcon3-7B** 8K | *39.4%* | **22.5%** | 22.0% | 21.3% | 19.5% | 19.5% | 14.4% |
| **Falcon3-7B** 16K | *48.8%* | **29.6%** | 29.3% | 26.5% | 26.9% | 25.9% | 17.9% |

| Model / Context | FullContext | KiaOmni-G ✦ | KiaOmni-σ8 ✦ | BlockSal | Ada-SnapKV | H2O | RealSnapKV |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **BioMistral-7B** 4K | *49.5%* | 39.3% | 38.8% | 37.9% | **40.2%** | 37.6% | 28.1% |
| **BioMistral-7B** 8K | *52.2%* | 45.6% | 45.8% | 44.7% | **47.2%** | 43.9% | 33.1% |
| **BioMistral-7B** 16K | *78.3%* | 67.5% | 67.5% | 64.2% | **69.6%** | 65.4% | 49.6% |

### Cross-Model Mean — % of FullContext CORRECT% per Budget

| Budget | KiaOmni-G ✦ | KiaOmni-σ8 ✦ | BlockSal | Ada-SnapKV | H2O | RealSnapKV |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| B=98 | **52.0%** | 51.4% | 48.4% | 49.0% | 45.8% | 31.6% |
| B=128 | **61.1%** | 59.4% | 54.1% | 55.4% | 53.2% | 35.6% |
| B=256 | 73.4% | 74.4% | **75.8%** | 69.2% | 59.5% | 49.0% |
| B=512 | **88.2%** | 85.5% | 82.9% | 74.6% | 70.7% | 61.5% |

*Source: 35981 judged predictions across 4 models × 4 budgets × 3 contexts. ✦ = KiaOmni variant · **Bold** = best eviction policy per row · *Italic* = FullContext oracle.*