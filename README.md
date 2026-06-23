# RenAIssance: Self-Supervised OCR for 17th-Century Spanish Printed Sources

A four-stage OCR pipeline that pre-trains a Vision Transformer on unlabeled historical Spanish documents before fine-tuning TrOCR with confidence-gated Tesseract silver labels.

## What it does

This pipeline digitizes early modern Spanish printed sources from roughly 1620 to 1750. The main idea is to adapt the visual encoder to 17th-century typography using self-supervised masked autoencoder (MAE) pre-training before any text supervision is introduced. This reduces the domain shift that causes standard OCR models to misread long-s glyphs, ligatures, and period abbreviations.

The work was carried out as a GSoC 2026 contribution to the RenAIssance project at the HumanAI Foundation / CERN.

## How it works

1. **Layout analysis** â€” Extracts the main text block from scanned PDFs using adaptive Gaussian binarization, connected-component filtering, and Otsu projection profiling.
2. **Tesseract silver reference** â€” Generates a deterministic OCR reference with `tesseract --oem 3 --psm 6 -l spa` on the cropped text block.
3. **MAE self-supervised pre-training** â€” Trains a small Vision Transformer encoder on ~300â€“500 unlabeled line images with 75% of patches masked, so the model learns historical letterform structure from pixels alone.
4. **Confidence-gated fine-tuning** â€” Transfers the MAE patch embeddings into TrOCR and fine-tunes only on lines where Tesseract's mean word confidence is at least 70%.
5. **Quality-gated LLM correction** â€” Applies a constrained Llama 3.3-70B pass only when the TrOCR output passes Spanish character-ratio, digit-ratio, punctuation-density, and length-ratio checks.
6. **Evaluation** â€” Reports character error rate (CER), word error rate (WER), and character confusion matrices against the Tesseract silver standard.

## Tech stack

- Python, PyTorch, Transformers (TrOCR)
- OpenCV, pdf2image, Pillow, pytesseract
- jiwer, scikit-learn, seaborn, matplotlib
- Groq API for LLM correction
- Google Colab with a T4 GPU runtime

## Results / Metrics

The pipeline improves on the untuned TrOCR base model on all 5 test sources. CER is measured against the Tesseract silver standard. 'FT' = MAE-pre-trained + fine-tuned TrOCR (no LLM correction).

| Source | Base CER | MAE+FT | Full Pipeline | Winner |
|--------|:--------:|:------:|:-------------:|:------:|
| Buendia â€” *Instruccion* (1740) | 72.0% | 62.1% | **61.6%** | Pipeline |
| Guardiola â€” *Tratado nobleza* (1591) | 40.4% | 41.3% | **40.1%** | Pipeline |
| PORCONES.228.38 (1646) | 112.7% | **70.8%** | 71.0% | FT |
| PORCONES.23.5 (1628) | 89.3% | 78.4% | **78.2%** | Pipeline |
| PORCONES.748.6 (1650) | 77.4% | **75.9%** | 75.9% | FT |

The largest improvement is on PORCONES.228.38, where MAE+FT drops CER from 112.7% to 70.8% â€” an absolute reduction of about 42 percentage points (~37% relative). The full pipeline result for this source is 71.0%. This source was insertion-heavy at baseline; MAE pre-training on the actual historical glyphs appears to address the glyph-confusion problem behind the high insertion rate.

## How to run

Open `GSOC_2026_02_Execution_Pipeline_v2.ipynb` in Google Colab with a T4 GPU runtime:

1. Mount your Google Drive and place source PDFs in a single folder.
2. Set `BASE_DIR` in the setup cell to that folder.
3. Run all cells in order. The notebook auto-discovers PDFs and estimates ~50 minutes total runtime on T4.
4. Add a Groq API key to Colab Secrets as `GROQ_API_KEY` for the LLM correction stage.

System dependencies (auto-installed in Colab):

```
poppler-utils
 tesseract-ocr
 tesseract-ocr-spa
```

Python dependencies are listed in `requirements.txt`.

## License

No license file is currently present in the repository.

---

## Suggested GitHub metadata

**Suggested descriptions (<=160 chars):**

1. Self-supervised MAE and confidence-gated OCR pipeline for 17th-century Spanish archives.
2. Historical Spanish OCR with masked autoencoder pre-training and TrOCR fine-tuning.

**Suggested topic tags:**

`ocr`, `historical-documents`, `self-supervised-learning`, `computer-vision`, `trocr`
