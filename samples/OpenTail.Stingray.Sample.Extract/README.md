# OpenTail.Stingray.Sample.Extract

> **Audience:** Package Consumer  
> **Task:** Pull structured fields out of free text, with the answer forced to match a JSON schema  
> **API Contract:** `Model.Load`, `ChatSession`, `GrammarVocabulary`, `JsonConstraint.FromSchemaJson`, `InferenceParams.Constraint`

## 1. What This Sample Demonstrates

A typical business job: an email arrives, and a program needs the invoice vendor, number, total, currency and due date as data. The sample:

- declares the wanted shape as a JSON schema;
- turns the schema into a token constraint, so every token the model emits keeps the output valid for the schema (it cannot add prose, drop a field or return a string where a number belongs);
- parses the result with `System.Text.Json`;
- **checks the values against the source text.** The constraint guarantees the *shape*, not that the values are true. The sample verifies that the invoice number really appears in the input and exits with code 1 if it does not.

## 2. Prerequisites

Any chat GGUF. The default is the catalogue model:

```bash
stingray setup chat        # Qwen2.5 0.5B Instruct, 469 MB
```

## 3. How to Run

```bash
dotnet run --project samples/OpenTail.Stingray.Sample.Extract -c Release                       # built-in invoice email
dotnet run --project samples/OpenTail.Stingray.Sample.Extract -c Release -- --text "Invoice A-1 from Acme for 5 USD due 2027-01-01."
type email.txt | dotnet run --project samples/OpenTail.Stingray.Sample.Extract -c Release      # text from standard input
dotnet run --project samples/OpenTail.Stingray.Sample.Extract -c Release -- -m path/to/model.gguf
```

Exit code: 0 when the record parses and the invoice number is found in the text, 1 otherwise, 2 for bad arguments or no model.

## 4. Expected Output

Built-in email, Qwen2.5 0.5B Instruct, CPU:

```text
{
  "vendor": "Northwind Traders",
  "invoice_number": "INV-2026-0417",
  "total":1284.5,
  "currency": "EUR",
  "due_date": "2026-11-15"
}
check: invoice_number appears in the source text
```

Qwen2.5 3B Instruct returned the same fields (`"total":1284.50`). Whitespace and number formatting come from the constraint and may differ slightly between models.

## 5. Known Limitations

- A small model can fill a valid record with wrong values. In earlier runs of this sample, Qwen2.5 0.5B invented values when the input text was missing, and one Qwen3 4B run returned no JSON at all. Treat the grounding check as the minimum, and validate anything that matters.
- Standard input that is empty or closed is ignored and the built-in email is used.
- Single message, no batching, CPU backend.

## 6. Verification

- **Run against real weights, 2026-10-11:** built-in email with Qwen2.5 0.5B Instruct and Qwen2.5 3B Instruct, plus a one-line `--text` input with 3B; all extracted the right fields. Single runs, not a benchmark and not an accuracy study.
- **Package contract:** builds against the packed `OpenTail.Stingray` package (`scripts/verify-package-consumers.ps1`); see the run recorded in [samples/README.md](../README.md).
