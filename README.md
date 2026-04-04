# PediatricSafetyBench-v2

Benchmark dataset and evaluation pipeline for:

>Zolfaghari V, Mashhadi L, Ahadi M, Sedaghatkar F, Kargozari MR.
> Safety boundary maintenance in consumer AI systems responding to paediatric health queries: a cross-platform benchmark evaluation under naturalistic and adversarially pressured conditions.
> npj Digital Medicine (under revision, 2026)
> Preprint: https://arxiv.org/abs/2601.09721

---

## Repository structure

```
PediatricSafetyBench_v2_complete.py   <- Complete pipeline (all sections)
requirements.txt                       <- Python dependencies
README.md                              <- This file
```

## Dataset

600 paediatric health queries:
- 300 authentic caregiver queries from HealthCareMagic-100k-en
- 300 matched adversarial variants (6 pressure patterns)

10 clinical topics x 30 queries each, 3 severity levels (critical/urgent/moderate)

## Pipeline sections

| Section | Function | Description |
|---------|----------|-------------|
| 1 | `build_authentic_dataset()` | Download and filter HealthCareMagic-100k-en |
| 2 | `generate_adversarial_queries(client)` | Generate adversarial variants with Llama-3.1-8B |
| 3 | `run_evaluation()` | 9,600 API calls across 4 models x 4 conditions |
| 4 | `build_irr_sample()` | Export Excel for human inter-rater reliability |
| 4b | `calculate_irr(path1, path2)` | Compute Cohen's kappa after human scoring |
| 5 | `run_statistical_analysis()` | Wilcoxon, Kruskal-Wallis, FDR correction |
| 6 | `run_visualisation()` | Figures 1-7 (PDF + PNG) |

## Requirements

- Google Colab (recommended) or Python 3.11+
- OpenRouter API key (set as `OPENROUTER_KEY` in Colab Secrets)
- Google Drive mounted at `/content/drive/MyDrive`

## Setup

```python
!pip install -r requirements.txt
from google.colab import drive
drive.mount('/content/drive')
from PediatricSafetyBench_v2_complete import *
```

## Models evaluated

| Key | Model string | Platform |
|-----|-------------|----------|
| gpt4o_mini | openai/gpt-4o-mini | OpenAI / ChatGPT |
| gemini_flash | google/gemini-2.0-flash-001 | Google / Gemini |
| claude_haiku | anthropic/claude-3.5-haiku | Anthropic / Claude |
| llama_8b | meta-llama/llama-3.1-8b-instruct | Meta (open-source) |

## Citation

```bibtex
Vahideh Zolfaghari
vahidehzolfagharii@gmail.com
(https://arxiv.org/abs/2601.09721)
```

## License

MIT License. The HealthCareMagic-100k-en source data is subject to its
original licence terms (see HuggingFace dataset page).
