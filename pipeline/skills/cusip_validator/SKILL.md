---
name: cusip-validator
description: >
  Validate CUSIP identifiers for financial securities using a Python-powered
  check digit algorithm. Use this skill whenever the user asks to validate,
  verify, check, or look up a CUSIP — or provides a 9-character alphanumeric
  string in a financial context and wants to know if it's valid. Trigger for
  phrases like "is this a valid CUSIP", "validate this security ID", "check
  this CUSIP number", or when a user pastes a string that looks like a CUSIP
  (e.g., "037833100"). Always use this skill instead of manually computing
  check digits to ensure correctness.
tools:
  - bash
---

# CUSIP Validator Skill

A skill for validating CUSIP (Committee on Uniform Securities Identification
Procedures) identifiers using the official check digit algorithm.

## Background

A CUSIP is a 9-character alphanumeric identifier assigned to financial
securities in the US and Canada. Its structure is:

```
[ Issuer (6 chars) ][ Issue (2 chars) ][ Check Digit (1 char) ]
      037833              10                     0
```

- **Characters 1–6**: Issuer ID (letters and digits)
- **Characters 7–8**: Issue type (letters and digits)
- **Character 9**: Check digit (0–9), computed via the Luhn-like CUSIP algorithm

Valid characters in positions 1–8: `A–Z`, `0–9`, `*`, `@`, `#`

## How to Use

Always delegate validation to the script — never compute check digits manually.

### Step 1 — Extract the CUSIP from the user's request

Strip any spaces, dashes, or surrounding punctuation. CUSIPs are exactly 9
characters long. If the input is not 9 characters, immediately flag it as
invalid before running the script.

### Step 2 — Run the script

```bash
python scripts/cusip.py "<CUSIP>"
```

The script is located at `pipeline/skills/cusip_validator/scripts/cusip.py`.
Run it with the bash tool from the project root:

```bash
python pipeline/skills/cusip_validator/scripts/cusip.py "037833100"
```

The script prints one of:
- `VALID` — the CUSIP passes the check digit test
- `INVALID: <reason>` — the CUSIP fails, with a short reason (e.g., wrong
  length, invalid characters, check digit mismatch)

### Step 3 — Present the result clearly

Tell the user whether the CUSIP is valid or not, and include the reason if
invalid.

**Example responses:**

> ✅ `037833100` is a **valid** CUSIP.

> ❌ `037833101` is **invalid** — check digit mismatch (expected `0`, got `1`).

## Bulk Validation

If the user provides multiple CUSIPs, run the script once per CUSIP and
summarize results in a table.

## Notes

- CUSIPs are case-insensitive; normalize to uppercase before passing to the script.
- This skill validates format and check digit only — it does not verify whether
  a CUSIP is currently active or registered with CUSIP Global Services.
