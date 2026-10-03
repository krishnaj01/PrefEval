"""Verbatim copies of the upstream PrefEval prompt-construction code.

Source: https://github.com/amazon-science/PrefEval at commit 5079505
(the last upstream commit before local modifications in this fork).

Only import lines were edited (marked "[phase1 edit]"):
  * common_utils.py: openai / google-genai imports made optional
  * baselines_handling_classification.py: `utils.` imports made package-relative
No prompt text, prompt formatting, retrieval logic or answer parsing was changed,
so prompts sent to the local model are byte-identical to the paper's code path.
"""
