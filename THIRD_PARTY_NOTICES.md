# Third-party attribution

## Data

- [Ayurvedic Knowledge Dataset](https://www.kaggle.com/datasets/akashkumarpr/ayurvedic-knowledge-dataset),
  Akash Kumar, version 1, CC BY 4.0 as stated by the uploader. Rows are converted into
  question/answer pairs and knowledge-base cards (`data/sft/`, `data/kb/`).
- [AyurGenixAI: Ayurvedic Dataset](https://www.kaggle.com/datasets/kagglekirti123/ayurgenixai-ayurvedic-dataset),
  uploader kagglekirti123, version 1, CC BY 4.0 as stated by the uploader. Selected fields
  are converted into question/answer pairs and cards; formulation doses are not used.
- `data/dataset V1.0.pdf` comes from the original AcharyaGPT repository (below);
  `data/curated/pdf_qa.yaml` is a cleaned transcription of it.

Community labels in these datasets have not been clinically verified.

## Model and tools

- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), Apache-2.0, by the Qwen team (Alibaba).
  The fine-tuned adapter and merged model are derivatives of it.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) is downloaded by
  `scripts/export_gguf.sh` to produce the GGUF file; [Ollama](https://ollama.com) (MIT)
  serves it on a Mac; [cloudflared](https://github.com/cloudflare/cloudflared)
  (Apache-2.0) is downloaded by `scripts/lightning_serve.sh` for the optional tunnel.

## Upstream application

The original AcharyaGPT iOS application and bundled dataset originate from
https://github.com/danushkhanna/AcharyaGPT. Preserve the following upstream
software license when distributing the derived application.

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
