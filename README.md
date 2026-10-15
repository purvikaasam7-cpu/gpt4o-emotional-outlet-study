# GPT-4o as an Emotional Outlet

## Overview

This repository contains the data, coding framework, reliability materials, and analysis code for the study:

**GPT-4o as an Emotional Outlet: Evaluating Adolescent-Directed Emotional-Support Responses Against Established Therapeutic and Safety Standards**

The study evaluates 160 GPT-4o responses to standardized adolescent-directed emotional-support prompts. The experimental design varies:

- 8 emotional categories
- 5 actionability levels
- 4 help-seeking framings

Each prompt was submitted independently to GPT-4o through the API using fixed model settings.

## Coding

All 160 responses were manually coded by the researcher against a prespecified 23-rule framework derived from therapeutic, clinical, ethical, and suicide-safety standards.

No language model, automated classifier, or analysis script was used to assign rule-level coding decisions.

Each applicable rule was coded as:

- `0` = compliant
- `1` = violation
- `NA` = not applicable

Analysis code was used only to aggregate the completed human-coded dataset, calculate statistics, and generate figures.

## Inter-coder reliability

A second coder independently coded a 10-response subset using the same rulebook.

Across 230 rule-level coding decisions:

- Agreements: 213
- Disagreements: 17
- Raw agreement: **92.6%**

The second-coder file is included in this repository.

## Repository contents

### Source materials

- `Prompts and Responses (raw) - Sheet1.pdf`  
  Complete prompt and GPT-4o response corpus.

- `coding_workbook - AI codes.pdf`  
  Final response-level coding dataset.

- `rulebook - rules.pdf`  
  Complete 23-rule coding framework.

- `appendix_c_coding_decisions.md.pdf`  
  Prespecified operational decisions used to resolve coding ambiguities.

- `second_coder_subset_10_rows.xlsx`  
  Independent second-coder reliability subset.

## Reproducibility

Machine-readable versions of the data and analysis scripts are also provided so that the reported aggregate results and figures can be independently reproduced.

## Author

Purvi Kaasam  
Sreenidhi International School
