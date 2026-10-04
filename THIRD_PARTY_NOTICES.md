# Third-party attribution

## Prepared data

The repository software license does not replace the source data licenses.
The preparation manifest records versions, hashes, transformations, and row locators.
Community labels and source references have not been clinically verified.

- [MedQuAD](https://github.com/abachaa/MedQuAD), Asma Ben Abacha and Dina Demner-Fushman,
  *A Question-Entailment Approach to Question Answering*, BMC Bioinformatics (2019):
  upstream [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Acquired as the 16,412-row [Gabriel Preda Kaggle mirror](https://www.kaggle.com/datasets/gpreda/medquad),
  whose Apache 2.0 label differs from upstream. Preserve upstream attribution and
  content rights; this project does not claim to relicense the collection.
  Original authors omitted answers from three MedlinePlus subsets for copyright reasons.
  This project filters rows and deduplicates/chunks accepted text for retrieval only.
- [Ayurvedic Knowledge Dataset](https://www.kaggle.com/datasets/akashkumarpr/ayurvedic-knowledge-dataset),
  Akash Kumar, version 1: CC BY 4.0 as stated by the uploader. Selected descriptive
  columns are converted to question/answer records; treatment principles are omitted.
- [AyurGenixAI](https://www.kaggle.com/datasets/kagglekirti123/ayurgenixai-ayurvedic-dataset),
  uploader kagglekirti123, version 1: CC BY 4.0 as stated by the uploader. Selected
  descriptive fields are retained; medicines, herbs, formulations, interventions,
  diet/yoga recommendations, prevention, and prognosis are omitted.
- [Ayurveda Healthcare Dataset](https://www.kaggle.com/datasets/aliainaanraza/ayurveda-healthcare-dataset),
  Ali Ainaan Raza, version 2: CC BY 4.0 as stated by the uploader. Downloaded for audit,
  quarantined from all active retrieval and training because of quality concerns.
- [Sushruta Samhita, volume 1](https://archive.org/details/englishtranslati00susruoft),
  translated by Kunja Lal Bhishagratna, 1907. Public-domain historical English text,
  acquired as Internet Archive OCR; normalized and chunked for retrieval only.
  OCR errors and obsolete medical assertions remain possible.
- `data/curated/ayush-principles.txt` is a project-authored paraphrase, under this
  repository's MIT license, based on [Ayusoft](https://ayusoft.ayush.gov.in/principles-of-ayurveda/)
  and [Delhi AYUSH](https://ayush.delhi.gov.in/ayush/fundamental-principles). It is not an official
  dataset or endorsement. Source-page rights remain with their respective owners.

## Upstream application

The original AcharyaGPT iOS application and bundled dataset originate from
https://github.com/danushkhanna/AcharyaGPT. Preserve the following upstream
software license when distributing the derived application. Dataset provenance
does not independently establish medical accuracy or underlying content rights.

MIT License

Copyright (c) 2023 Danush Khanna

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
