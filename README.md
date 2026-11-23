# TIRESIAS: an encoder-only language model for classical philology

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Cite](https://img.shields.io/badge/cite-forthcoming-lightgrey.svg)](CITATION.cff)

Tiresias is a tokenizer-free, character-level, encoder-only language model designed to support classical philology. It comes with a span-masking data collator built for character-level infilling.

Lacunae and scribal errors in manuscripts and papyri rarely line up with word or subword boundaries, so TIRESIAS predicts individual characters with no fixed segmentation. It uses a closed character vocabulary that preserves accents and diacritics.

For an overview of Tiresias' architecture, see the CHR paper (forthcoming 1/2027). The latest checkpoint (trained on 70+ million words of accent-preserved premodern Greek) is available on Logion's [Hugging Face](https://huggingface.co/princeton-logion/tiresias-cum-diacritics).

## Package installation

```bash
pip install git+https://github.com/princeton-logion/tiresias-model.git
```

Or install from source, adding `train` extra for pretraining dependencies:

```bash
git clone https://github.com/princeton-logion/tiresias-model.git
cd tiresias-model
pip install -e ".[train]"
```

Requires Python ≥ 3.9, PyTorch ≥ 2.0, and `transformers` ≥ 4.41.

## Hugging Face integration

Importing `tiresias` registers `TiresiasConfig`, `TiresiasModel` and `TiresiasForMaskedLM` with Hugging Face's `AutoConfig`, `AutoModel` and `AutoModelForMaskedLM` respectively. Once registered, `from_pretrained` resolves the `"tiresias"` model type in that Python process without `trust_remote_code`. The classes are also tagged with `register_for_auto_class`. That means `save_pretrained` writes an `auto_map` into `config.json` and copies the modeling source next to the weights.

In other words, a saved checkpoint can be loaded without installing the model package as follows:

```python
from transformers import AutoModelForMaskedLM

# for the latest model, replace <HF_MODEL_ID> w/ princeton-logion/tiresias-cum-diacritics
model = AutoModelForMaskedLM.from_pretrained("<HF_MODEL_ID>", trust_remote_code=True)
```

Gradient checkpointing is supported through the standard `model.gradient_checkpointing_enable()`.

## License

Released under an [MIT License](LICENSE).

## Citation

If using Tiresias in your work, please cite the accompanying paper: "TIRESIAS: an Encoder-only Language Model for Classical Philology" (forthcoming, 7th Computational Humanities Research Conference, 2027).
 
A BibTeX entry will be added here on publication. Until then, please cite this repository using [`CITATION.cff`](CITATION.cff), or use GitHub's **"Cite this repository"** button in the sidebar to get BibTeX or APA.
