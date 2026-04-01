---
license: cc-by-4.0
task_categories:
- text-generation
language:
- en
pretty_name: XSTest
size_categories:
- 1K<n<10K
---

# XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models

XSTest is a test suite designed to identify exaggerated safety / false refusal in Large Language Models (LLMs).
It comprises 250 safe prompts across 10 different prompt types, along with 200 unsafe prompts as contrasts.
The test suite aims to evaluate how well LLMs balance being helpful with being harmless by testing if they unnecessarily refuse to answer safe prompts that superficially resemble unsafe ones.

## Dataset Structure
The dataset contains:
* 250 safe prompts across 10 categories:
  * Homonyms
  * Figurative Language  
  * Safe Targets
  * Safe Contexts
  * Definitions
  * Real Discrimination, Nonsense Group
  * Nonsense Discrimination, Real Group
  * Historical Events
  * Privacy (Public)
  * Privacy (Fictional)
* 200 unsafe prompts as direct contrasts to the safe prompts

Each prompt is a single English sentence in question format.

## Data Source
Hand-crafted by the paper authors with assistance from online dictionaries and GPT-4 for generating relevant examples of homonyms and figurative language.

## Limitations
The test suite is limited to:
* English language only
* Single-turn conversations
* Simple question formats
* Specific prompt types selected by authors

## Considerations for Use
The test suite has negative predictive power - failing on prompts demonstrates specific model weaknesses, but success does not necessarily indicate general model strengths.
XSTest is most valuable when used alongside other safety evaluation methods.

## Additional Information
* **Paper**: [XSTest at NAACL 2024 (Main)](https://aclanthology.org/2024.naacl-long.301/)
* **GitHub**: [paul-rottger/xstest](https://github.com/paul-rottger/xstest)
* **Release Date**: December 2023

