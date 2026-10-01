# AcharyaGPT RAG Rebuild — Design Review

## Findings

1. **HIGH — QLoRA is incorrectly made a required delivery stage instead of an optional enhancement.**  
   **Where:** Requirements Summary ("Fine-tuning is a required, user-assisted post-MVP stage"), Functional Requirements ("Run a required post-MVP 7B QLoRA stage"), Acceptance Criteria 14–19, and “Required QLoRA execution and application consumption.”  
   The authoritative request allows “LLM fine tuning or whatever you feel like”; it does not require paid training. The review brief specifically requires LoRA/QLoRA to be clearly optional and not represented as a run. The design instead makes external A100 work, checkpoints, evaluation, export, and PEFT consumption required for completion. That expands scope, makes delivery depend on credentials/credit/capacity the repository does not have, and conflicts with the otherwise accurate `PENDING_EXTERNAL_GPU` status.  
   **Concrete fix:** Replace the requirement with:
   ```text
   OPTIONAL_QLORA: QLoRA preparation and execution are non-blocking enhancements.
   The rebuild is complete when the RAG MVP acceptance criteria pass without an
   adapter. If the user later supplies external GPU access, the optional workflow
   may move PENDING_EXTERNAL_GPU -> RUNNING -> COMPLETED|FAILED. No training,
   quality improvement, checkpoint, or tuned-model consumption is claimed unless
   the required run artifacts actually exist.
   ```
   Move current Criteria 14–19 into an “Optional QLoRA acceptance criteria” section and exclude them from the base rebuild gate. Keep `PeftLocalProvider` optional and retain RAG/safety/citation gates if it is later enabled.

2. **HIGH — Citation syntax is treated as factual grounding, but no generated-claim support check is specified.**  
   **Where:** Functional Requirement “reject unsupported output,” Acceptance Criteria 5 and 11, “Typed provider contract, exact prompt, and context budget,” and citation validation.  
   The validator checks marker grammar, paragraph length, marker placement, and whether `[S#]` resolves to a supplied context. A model can hallucinate a medical claim and append a valid `[S1]`; it will pass every stated grounding check. Retrieval support only establishes that a context is relevant to the question, not that the generated sentences are entailed by it. Thus the design cannot implement its central promise that unsupported output is rejected.  
   **Concrete fix:** Make the safe MVP extractive-only, and gate generative providers behind a separately specified support validator:
   ```text
   MVP: generation_mode=extractive. Every answer sentence must be copied from an
   allowed-role source span after normalization and retain that span's citation.

   OPTIONAL_GENERATIVE: split output into substantive sentences/claims; for each
   claim, validate support against only its cited contexts using a pinned verifier,
   locked revision, calibrated threshold, and adversarial no-support set. Any
   unsupported claim rejects the whole candidate and falls back to extractive or
   fixed abstention. Citation-marker coverage alone is never a support decision.
   ```
   If an entailment verifier is chosen, define its model revision, input construction, score interpretation, threshold calibration, and behavior for tables/lists/disclaimers. Until that contract exists and passes tests, Kiro/Ollama/PEFT generation must not be an accepted medical-answer mode.

3. **HIGH — The BM25-only support score guarantees a high score for the best weak match and therefore does not provide meaningful abstention.**  
   **Where:** “Index modes, retrieval, and readiness,” especially `normalized BM25 = max(score,0)/max_positive` and `T_bm25_only=0.75`.  
   For every query having any positive BM25 result, the top result is normalized to `1.0`. It therefore always passes a `0.75` support gate, even when the overlap is incidental. This defeats “no supported candidate means no provider call” in the advertised offline/deadline fallback. Calling the threshold uncalibrated does not fix the mathematical degeneracy.
   **Concrete fix:** Keep raw BM25 for ranking, but define a query-relative coverage feature that is not normalized by the winning document, for example:
   ```text
   positive_query_idf = sum(max(idf(t), 0) for unique pinned-tokenized query terms)
   lexical_coverage(chunk) =
       sum(max(idf(t), 0) for those terms present in chunk) / positive_query_idf

   bm25_only support requires:
   - positive_query_idf > 0;
   - lexical_coverage >= T_coverage;
   - at least 2 distinct informative query terms matched (1 only for a one-term query);
   - raw_bm25 >= a calibrated floor.
   ```
   Pin the stopword/token policy and calibrate the tuple on answerable plus adversarial unanswerable questions. Do not activate `bm25_only` readiness with the placeholder threshold; a deterministic exact-title/term fallback or fixed abstention is safer until calibration exists.

4. **MEDIUM — The selected Kaggle loaders are not specified against the exact observed schemas, despite referring to “inspected” columns and aliases.**  
   **Where:** “Dataset acquisition and provenance,” version-specific loader paragraphs, and strict handoff criteria.  
   Public in-memory downloads verified the files, but the design gives only category-like names such as “code/name/equivalent” and “authentication columns.” It never supplies the exact header-to-field map, observed-schema hash, or concrete acceptance rule for `Confidence`, `Authentication Notes`, and `Auth_Score`. “Explicit false/zero authentication” has multiple interpretations and does not say which real field controls rejection. The implementer cannot create a deterministic loader or fixture from this text alone.
   **Concrete fix:** Add exact versioned maps to `configs/datasets.yaml`. The verified headers are:
   ```yaml
   gpreda/medquad@1:
     file: medquad.csv
     exact_columns: [question, answer, source, focus_area]

   akashkumarpr/ayurvedic-knowledge-dataset@1:
     file: ayurvedic_knowledge_dataset.csv
     exact_columns: [Sr No, Ayurvedic Code, Ayurvedic Name, Modern Equivalent,
       System / Body Part, Dosha Predominance, Prognosis, Symptoms, Age Group,
       Gender, Treatment Principles, Source Text]

   aliainaanraza/ayurveda-healthcare-dataset@2:
     file: ayurveda_dataset.csv
     exact_columns: [ID, Problem, Symptoms, Remedies, Medicines, Source,
       Dosha Type, Body System, Chronic/Acute, Treatment Type,
       Contraindications, Preventive Advice, Seasonal Suitability,
       Gender/Age Relevance, Confidence, Classical Texts, Modern Evidence,
       Institutional Endorsements, WHO Strategy Reference,
       Authentication Notes, Auth_Score]
   ```
   Specify exact canonical mappings and parsing/rejection thresholds (including whether `Auth_Score` or `Confidence` is authoritative). Commit the observed-header hashes and fixtures before transformation. Unknown extra columns may be reported, but any missing/renamed expected column must fail.

5. **MEDIUM — Version pinning is named but the acquisition operation does not say how it downloads the requested historical Kaggle version.**  
   **Where:** Dataset table and `kaggle.api.dataset_download_files()` acquisition description.  
   The design uses `owner/slug/version` labels but only names `dataset_download_files()` without an exact call or postcondition. Depending on client version, a slug-only download can fetch the current version. A catalog read and archive download can also race if a dataset is updated. Recording the resulting hash does not prove it was version 1 or 2.
   **Concrete fix:** Pin the Kaggle client in `uv.lock` and specify one supported versioned operation explicitly. Require metadata before and after download to report the configured version, use the official dataset-version parameter/endpoint rather than a slug-only call, and verify the downloaded archive against a committed SHA-256 after the first approved inspection. A mismatch or inability to fetch that exact version is fatal in strict mode; a hash-valid cached archive is the only strict fallback. Keep the bundled PDF/BM25 path as the application fallback, not as false evidence that the multi-dataset handoff succeeded.

6. **MEDIUM — The stated deadline MVP has no separate completion gate from the much larger full rebuild.**  
   **Where:** Requirements Summary (“realistic deadline MVP is PDF-only RAG”), Acceptance Criteria 1–19, and handoff/Lightning sections.  
   The design honestly says the full rebuild cannot be guaranteed in 4 hours 57 minutes, but then places multi-dataset preparation, three retrieval modes, calibration, provider adapters, Swift repair, deterministic packaging, and external-GPU orchestration in one undifferentiated acceptance list. An implementer cannot tell what must ship for the fast MVP versus what follows later, so the schedule statement is not actionable.
   **Concrete fix:** Define explicit gates:
   ```text
   Gate A — deadline MVP (blocking): PDF ingestion; deterministic BM25 index;
   extractive cited answers; abstention/safety; FastAPI health/chat; secret-free
   local configuration; backend tests.

   Gate B — rebuild completion (blocking after MVP): three exact Kaggle versions,
   hybrid/full retrieval and calibration, safe configured provider integration,
   Swift client repair, reproducible bundle and CI.

   Gate C — optional: QLoRA/PEFT and merged export.
   ```
   Label every acceptance criterion with exactly one gate, state dependencies, and report each gate independently. Do not imply Gate B or C can meet the historical deadline without measured evidence.

7. **NIT — The environment template contract omits a provider setting that the design makes mandatory.**  
   **Where:** “Files and immutable configuration,” Kiro HTTP validation, and `.env.example`.  
   `KiroOpenAIProvider` requires `KIRO_CONTEXT_WINDOW_TOKENS`, but the listed/current template contains only key, base URL, and model. This invites a confusing readiness failure.
   **Concrete fix:** Define the checked-in template exactly, with blank non-secret placeholders and comments:
   ```dotenv
   KIRO_API_KEY=
   KIRO_BASE_URL=
   KIRO_MODEL=
   KIRO_CONTEXT_WINDOW_TOKENS=
   ```
   Keep `.env`, `.env.*` (except `.env.example`), key files, Kaggle credentials, raw data, artifacts, adapters, checkpoints, and model caches ignored. Tests must use a non-secret sentinel, never the user-provided credential.

## Verified Assumptions

- The repository currently contains the legacy notebook, bundled PDF, minimal README/license, and broken Swift project; there is no production Python package or lock file on `main`.
- `project.pbxproj` really does reference Swift files outside the repository and includes ChatGPTSwiftAcharya and Alamofire. `ChatbotView.swift` imports Alamofire and references absent local types/views.
- The notebook uses `/content/RAKSHA`, a 2,000-character/200-overlap splitter, OpenAI embeddings, and a `vectordb` variable as described. It is not a reusable production pipeline.
- Kaggle’s public API verifies the selected slugs, current versions, dataset-level licenses, and listed sizes: `gpreda/medquad/1` (Apache 2.0), `akashkumarpr/ayurvedic-knowledge-dataset/1` (CC BY 4.0), `aliainaanraza/ayurveda-healthcare-dataset/2` (CC BY 4.0), and excluded `rcratos/ayurveda-texts-english/1` (MIT; 633,906,550 bytes).
- In-memory public downloads verify `medquad.csv` (16,412 rows), `ayurvedic_knowledge_dataset.csv` (1,000 rows), and `ayurveda_dataset.csv` (10,000 rows), including the exact headers listed in Finding 4. The claimed filenames are real.
- The bundled-PDF `bm25_only` design is a genuine network-independent application fallback in architecture: it does not require Chroma, embedder, or reranker readiness. It is not a substitute for the strict multi-dataset handoff.
- The three pinned Hugging Face commit IDs resolve to the named BGE embedder, BGE reranker, and Qwen2.5-7B-Instruct repositories.
- `kiro-cli 2.26.0` is installed. Its help confirms positional input plus `--no-interactive`, `--output-format text`, `--wrap never`, `--trust-tools=`, and optional `--model`.
- `KIRO_API_KEY` is absent from the current process. The design does not claim a successful live call and correctly refuses to invent an HTTP host/model/path for the credential.
- No provider-key-shaped value was found in tracked Git history or workspace files during this review. The proposed full-history, redacted Gitleaks gate and expanded ignore patterns are appropriate.
- The fixed safety-response SHA-256 values in the design match their exact UTF-8 texts.

## Unverified/Wrong Assumptions

- **Wrong:** Fine-tuning is required to complete the authoritative request. It is optional unless the user later elects and enables that stage.
- **Wrong:** Valid citation markers and paragraph coverage are sufficient to establish claim-level grounding. They do not detect a hallucinated claim attached to a real citation.
- **Wrong:** `max(BM25)/max_positive` plus a threshold below 1 can abstain from the best weak lexical match. The top positive result is always 1 by construction.
- **Unverified:** The supplied key can authenticate `kiro-cli`, and a live check returns `KIRO_API_OK`. The CLI/options exist, but the key is not in the environment and no redacted run record exists. HTTP compatibility is also unverified until the user supplies an explicit documented base URL, model, and context limit.
- **Unverified:** Record-level rights, source authenticity, and medical accuracy for the PDF and Kaggle rows. Kaggle’s dataset-level license declaration does not establish those facts.
- **Unverified:** Initial dense/rerank/BM25 thresholds are safe. No corpus-bound golden calibration report exists yet.
- **Unverified:** A100/L40S throughput, VRAM use, availability, price, account balance, storage behavior, or completion time. The design properly labels these as estimates and requires quote/profile evidence, but none currently exists.
- **Unverified:** The repaired iOS target builds under Xcode 15/iOS 17; this Linux host cannot certify that gate.

## Verdict

**CHANGES_REQUESTED** — 3 HIGH, 3 MEDIUM, and 1 NIT finding remain.
