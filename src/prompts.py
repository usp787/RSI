"""Fixed plain-text prompts shared by generation and SFT.

The study uses Qwen2.5 base checkpoints, so prompts do not rely on an
instruction-model chat template. Evaluation prompt text is identical across M0
and every iterated checkpoint.
"""

from __future__ import annotations


INSTRUCTION = (
    "Solve the problem carefully and show a coherent derivation. "
    "End with exactly one final answer in the form \\boxed{answer}."
)

FEW_SHOT = r"""
Problem: A notebook costs $4 and a pen costs $2. What is the total cost of 3 notebooks and 2 pens?
Solution: Three notebooks cost 3 * 4 = 12 dollars. Two pens cost 2 * 2 = 4 dollars. The total is 12 + 4 = 16 dollars. Therefore, \boxed{16}.

Problem: Solve for x: 3x + 5 = 20.
Solution: Subtracting 5 gives 3x = 15. Dividing by 3 gives x = 5. Therefore, \boxed{5}.
""".strip()


def render_prompt(problem: str, *, few_shot: bool) -> str:
    sections = [INSTRUCTION]
    if few_shot:
        sections.append(FEW_SHOT)
    sections.append(f"Problem: {problem.strip()}\nSolution:")
    return "\n\n".join(sections)
