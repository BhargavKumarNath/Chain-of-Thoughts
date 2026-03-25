import re
import sympy
import logging
from typing import Dict, Any, Tuple
from app.schemas.reasoning import TrustScore

logger = logging.getLogger(__name__)

class VerificationEngine:
    """
    Verifies LLM reasoning steps using external deterministic solvers (SymPy)
    and calculates the final TrustScore.
    """
    
    def __init__(self):
        # Matches basic equations like "2 + 2 = 4" or "x = 5"
        self.equation_pattern = re.compile(r'([^=]+)=([^=]+)')

    def _verify_math(self, content: str) -> bool:
        """
        Extracts equations from text and verifies them using SymPy.
        Returns False ONLY if a mathematically invalid equation is found.
        Returns True if math is valid, or if no math is present.
        """
        if '=' not in content:
            return True
            
        # Clean string to find potential math
        clean_content = re.sub(r'[^\w\s\+\-\*\/\=\.\(\)]', '', content)
        
        for match in self.equation_pattern.finditer(clean_content):
            left_side, right_side = match.groups()
            try:
                # Parse left and right sides
                expr_left = sympy.sympify(left_side.strip(), evaluate=True)
                expr_right = sympy.sympify(right_side.strip(), evaluate=True)
                
                # Check equality (allowing for floating point tolerances)
                if expr_left.is_Number and expr_right.is_Number:
                    if not sympy.isclose(expr_left, expr_right, rel_tol=1e-5):
                        return False
            except Exception:
                # If it's not actually math (e.g., "Strategy = DIRECT"), ignore
                continue
                
        return True

    def verify_and_score(self, llm_output: Dict[str, Any], risk_estimate: float) -> Tuple[TrustScore, list]:
        """
        Runs verification signals over the steps and computes aggregate trust.
        Returns (TrustScore, flagged_step_indices)
        """
        steps = llm_output.get("reasoning_steps", [])
        flagged_indices =[]
        logical_consistency = 1.0
        
        for step in steps:
            content = step.get("content", "")
            
            # Signal 1: Math Verification
            if not self._verify_math(content):
                step["flagged"] = True
                flagged_indices.append(step.get("step_index"))
                logical_consistency -= 0.3 # Heavy penalty for bad math
                
        logical_consistency = max(0.0, logical_consistency)
        
        # Calculate aggregate trust based on signals and pre-calculated risk
        entropy = risk_estimate * 0.5 # Heuristic mapping for now
        evidence_alignment = 1.0 - (len(flagged_indices) * 0.1)
        contradiction_rate = len(flagged_indices) / max(len(steps), 1)
        
        aggregate = (logical_consistency * 0.4) + (evidence_alignment * 0.3) + ((1.0 - risk_estimate) * 0.3)
        aggregate = max(0.0, min(1.0, aggregate))
        
        trust = TrustScore(
            logical_consistency=logical_consistency,
            evidence_alignment=evidence_alignment,
            entropy=entropy,
            contradiction_rate=contradiction_rate,
            aggregate_score=aggregate
        )
        
        return trust, flagged_indices