"""
verifier.py — True verification pipeline with trust calibration and grounded hallucination risk.

Replaces the old regex-only equation checker with:
- Deterministic mathematical verification via MathSolver
- Logical consistency checks across reasoning steps
- Post-verification trust scoring (weighted: verification 50%, consistency 25%, self-consistency 25%)
- Grounded hallucination risk (near zero for verified deterministic domains)
- Verification method tracking and configurable tolerance thresholds
- Safe fallbacks: solver failures never crash the pipeline
"""

import logging
import re
from typing import Any, Dict, List, Tuple, Optional

from app.core.math_solver import MathSolver, SolverResult, ToleranceConfig
from app.schemas.reasoning import (
    TrustScore,
    VerificationDetail,
    VerificationStatusEnum,
)

logger = logging.getLogger(__name__)


# Trust score weight configuration
WEIGHT_VERIFICATION = 0.50
WEIGHT_REASONING_CONSISTENCY = 0.25
WEIGHT_SELF_CONSISTENCY = 0.25

# Thresholds
TRUST_VERIFIED_FLOOR = 0.85   # Minimum trust when deterministic verification passes
TRUST_FAILED_CEILING = 0.30   # Maximum trust when deterministic verification fails


class VerificationEngine:
    """
    Runs a multi-signal verification pipeline on LLM outputs and computes
    calibrated trust scores and grounded hallucination risk.
    """

    def __init__(self, tolerances: Optional[ToleranceConfig] = None):
        self.solver = MathSolver(tolerances=tolerances)
        self.tolerances = tolerances or ToleranceConfig()

        # Contradiction detection keywords
        self._negation_pairs = [
            ("increase", "decrease"), ("increases", "decreases"),
            ("greater", "less"), ("more", "fewer"),
            ("positive", "negative"), ("true", "false"),
            ("always", "never"), ("must", "cannot"),
            ("add", "subtract"), ("gain", "loss"),
        ]

    # 1. Mathematical verification
    def _verify_math_in_steps(
        self, steps: List[Dict[str, Any]], final_answer: str
    ) -> Tuple[List[VerificationDetail], List[int]]:
        """
        Verify all arithmetic and mathematical claims in reasoning steps
        and the final answer. Returns (verification_details, flagged_step_indices).
        """
        details: List[VerificationDetail] = []
        flagged_indices: List[int] = []

        for step in steps:
            content = str(step.get("content", ""))
            step_index = int(step.get("step_index", 0))

            # Arithmetic checks within each step
            arith_results = self.solver.extract_and_verify_arithmetic(content)
            for result in arith_results:
                detail = VerificationDetail(
                    method=result.method,
                    passed=result.success,
                    expected=str(result.computed_answer) if result.computed_answer is not None else None,
                    actual=None,
                    confidence=result.confidence,
                    message=f"Step {step_index}: {result.details or result.error or ''}",
                )
                details.append(detail)

                if not result.success and result.confidence > 0.5:
                    if step_index not in flagged_indices:
                        flagged_indices.append(step_index)

        # Verify the final answer if it contains arithmetic
        final_arith = self.solver.extract_and_verify_arithmetic(final_answer)
        for result in final_arith:
            details.append(VerificationDetail(
                method=result.method,
                passed=result.success,
                confidence=result.confidence,
                message=f"Final answer: {result.details or result.error or ''}",
            ))

        # Probability checks across all step text
        all_text = " ".join(str(s.get("content", "")) for s in steps) + " " + final_answer
        prob_result = self.solver.verify_probability(all_text)
        if prob_result.method != "none" and prob_result.confidence > 0:
            details.append(VerificationDetail(
                method=prob_result.method,
                passed=prob_result.success,
                confidence=prob_result.confidence,
                message=prob_result.details or prob_result.error or "",
            ))

        return details, flagged_indices

    # 2. Logical consistency checks
    def _check_logical_consistency(
        self, steps: List[Dict[str, Any]]
    ) -> Tuple[float, List[VerificationDetail]]:
        """
        Check reasoning steps for internal contradictions and logical flow.
        Returns (consistency_score 0-1, verification_details).
        """
        details: List[VerificationDetail] = []
        if len(steps) < 2:
            return 1.0, details

        contradiction_count = 0
        total_pairs = 0

        # Pairwise contradiction detection between consecutive steps
        for i in range(len(steps) - 1):
            content_a = str(steps[i].get("content", "")).lower()
            content_b = str(steps[i + 1].get("content", "")).lower()
            total_pairs += 1

            for pos, neg in self._negation_pairs:
                if (pos in content_a and neg in content_b) or (neg in content_a and pos in content_b):
                    # Check if they're talking about the same subject (simple heuristic)
                    words_a = set(content_a.split())
                    words_b = set(content_b.split())
                    overlap = words_a & words_b - {"the", "a", "an", "is", "are", "to", "and", "or", "in", "of"}
                    if len(overlap) >= 3:
                        contradiction_count += 1
                        step_i = int(steps[i].get("step_index", i + 1))
                        step_j = int(steps[i + 1].get("step_index", i + 2))
                        details.append(VerificationDetail(
                            method="logical_consistency",
                            passed=False,
                            confidence=0.6,
                            message=f"Potential contradiction between step {step_i} and step {step_j}",
                        ))
                        break

            # Circular Step Detection (Jaccard Similarity)
            words_a = set(content_a.split())
            words_b = set(content_b.split())
            union = words_a | words_b
            if union:
                jaccard = len(words_a & words_b) / len(union)
                if jaccard > 0.8:
                    contradiction_count += 0.5  # Penalize circular reasoning
                    details.append(VerificationDetail(
                        method="circular_step",
                        passed=False,
                        confidence=0.8,
                        message=f"High circularity (sim={jaccard:.2f}) between adjacent steps",
                    ))

        # Assumption alignment: check if conclusions reference prior assumptions
        assumption_alignment = self._check_assumption_alignment(steps, soft=False)
        details.extend(assumption_alignment[1])

        if total_pairs == 0:
            consistency = 1.0
        else:
            consistency = max(0.0, 1.0 - (contradiction_count / total_pairs) * 0.5)

        # Factor in assumption alignment
        consistency = (consistency * 0.7) + (assumption_alignment[0] * 0.3)

        return round(consistency, 4), details

    def _check_assumption_alignment(
        self, steps: List[Dict[str, Any]], soft: bool = False
    ) -> Tuple[float, List[VerificationDetail]]:
        """
        Check that assumptions stated in early steps are honored in conclusions.
        Returns (alignment_score 0-1, details).
        When soft=True (non-deterministic), use lenient scoring — paraphrased/abstracted matches count.
        """
        details: List[VerificationDetail] = []
        if len(steps) < 2:
            return 1.0, details

        # Gather all stated assumptions
        all_assumptions = []
        for step in steps:
            assumptions = step.get("assumptions", [])
            if isinstance(assumptions, str):
                assumptions = [assumptions]
            if isinstance(assumptions, list):
                all_assumptions.extend([str(a).lower().strip() for a in assumptions if str(a).strip()])

        if not all_assumptions:
            return 1.0, details  # No assumptions to check

        # Check if the final step's content references or is consistent with assumptions
        final_content = str(steps[-1].get("content", "")).lower()
        # For soft mode, also check all step content (paraphrased refs count)
        if soft:
            all_content = " ".join(str(s.get("content", "")).lower() for s in steps)
            referenced = sum(1 for a in all_assumptions if any(word in all_content for word in a.split() if len(word) > 3))
        else:
            referenced = sum(1 for a in all_assumptions if any(word in final_content for word in a.split() if len(word) > 3))
        alignment = min(1.0, referenced / max(len(all_assumptions), 1))

        # Soft mode: floor at 0.5 (paraphrasing is acceptable)
        if soft:
            alignment = max(0.5, alignment)

        if alignment < 0.3 and len(all_assumptions) > 1 and not soft:
            details.append(VerificationDetail(
                method="assumption_alignment",
                passed=False,
                confidence=0.4,
                message=f"Low assumption alignment: {referenced}/{len(all_assumptions)} assumptions reflected in conclusion",
            ))

        return alignment, details

    # 3. Aggregation: compute verification status, trust, hallucination risk
    def _compute_verification_status(
        self, math_details: List[VerificationDetail], consistency_score: float
    ) -> Tuple[VerificationStatusEnum, float]:
        """
        Determine overall verification status and confidence.
        Returns (status, confidence 0-1).
        """
        if not math_details:
            # No deterministic checks were possible — PARTIAL based on consistency alone
            return VerificationStatusEnum.PARTIAL, consistency_score * 0.5

        deterministic_checks = [d for d in math_details if d.confidence >= 0.8]
        if not deterministic_checks:
            return VerificationStatusEnum.PARTIAL, consistency_score * 0.5

        all_passed = all(d.passed for d in deterministic_checks)
        any_failed = any(not d.passed for d in deterministic_checks)

        if all_passed:
            avg_conf = sum(d.confidence for d in deterministic_checks) / len(deterministic_checks)
            return VerificationStatusEnum.PASSED, min(1.0, avg_conf)
        elif any_failed and not all(not d.passed for d in deterministic_checks):
            # Some passed, some failed
            pass_ratio = sum(1 for d in deterministic_checks if d.passed) / len(deterministic_checks)
            return VerificationStatusEnum.PARTIAL, pass_ratio * 0.6
        else:
            avg_conf = sum(d.confidence for d in deterministic_checks) / len(deterministic_checks)
            return VerificationStatusEnum.FAILED, avg_conf

    def _compute_trust(
        self,
        verification_confidence: float,
        verification_status: VerificationStatusEnum,
        consistency_score: float,
        self_consistency_score: float,
    ) -> TrustScore:
        """
        Compute calibrated trust score as a weighted sum.
        Enforces floor/ceiling constraints based on deterministic verification.
        """
        aggregate = (
            verification_confidence * WEIGHT_VERIFICATION
            + consistency_score * WEIGHT_REASONING_CONSISTENCY
            + self_consistency_score * WEIGHT_SELF_CONSISTENCY
        )

        # Enforce calibration constraints
        if verification_status == VerificationStatusEnum.PASSED and verification_confidence >= 0.9:
            aggregate = max(aggregate, TRUST_VERIFIED_FLOOR)
        elif verification_status == VerificationStatusEnum.FAILED and verification_confidence >= 0.8:
            aggregate = min(aggregate, TRUST_FAILED_CEILING)

        aggregate = round(max(0.0, min(1.0, aggregate)), 4)

        return TrustScore(
            verification_confidence=round(verification_confidence, 4),
            reasoning_consistency_score=round(consistency_score, 4),
            self_consistency_score=round(self_consistency_score, 4),
            aggregate_score=aggregate,
            # Legacy fields — computed for backward compat
            logical_consistency=round(consistency_score, 4),
            evidence_alignment=round(verification_confidence, 4),
            entropy=round(max(0.0, 1.0 - aggregate), 4),
            contradiction_rate=round(max(0.0, 1.0 - consistency_score), 4),
        )

    def _compute_hallucination_risk(
        self,
        verification_status: VerificationStatusEnum,
        verification_confidence: float,
        consistency_score: float,
        flagged_count: int,
        step_count: int,
        query_type: str = "SEMI_STRUCTURED",
    ) -> float:
        """
        Compute grounded hallucination risk.
        For deterministic domains: risk ≈ 1 - verification_confidence when verification passes.
        For OPEN_ENDED: based on coverage gaps and coherence, NOT verification_confidence.
        For non-deterministic: use consistency + flagged ratio signals.
        """
        if query_type == "OPEN_ENDED":
            # Open-ended: risk based on coherence gaps and factual errors only
            flagged_ratio = flagged_count / max(step_count, 1)
            risk = (
                (1.0 - consistency_score) * 0.5  # coherence is primary signal
                + flagged_ratio * 0.3             # factual errors
                + (1.0 - verification_confidence) * 0.2  # soft coverage factor
            )
            return round(max(0.0, min(1.0, risk)), 4)

        if verification_status == VerificationStatusEnum.PASSED and verification_confidence >= 0.8:
            # Deterministic domain verified → near-zero risk
            return round(max(0.0, (1.0 - verification_confidence) * 0.3), 4)

        if verification_status == VerificationStatusEnum.FAILED:
            # High risk when verification explicitly fails
            return round(min(1.0, 0.7 + (1.0 - consistency_score) * 0.3), 4)

        # PARTIAL or no deterministic checks
        flagged_ratio = flagged_count / max(step_count, 1)
        risk = (
            (1.0 - verification_confidence) * 0.4
            + (1.0 - consistency_score) * 0.3
            + flagged_ratio * 0.3
        )
        return round(max(0.0, min(1.0, risk)), 4)

    def _compute_instruction_completion(self, query: str, content: str) -> float:
        """
        Check if the final content adheres to explicit instructions in the query.
        Returns a score 0.0 - 1.0. 
        """
        query_lower = query.lower()
        content_lower = content.lower()
        score = 1.0
        penalties = 0.0

        # Check for numeric constraints like "exactly 3" or "at least 2"
        # Since it's heuristic, we just verify the presence of numbers or enumeration patterns
        # if the query explicitly requested them.
        import re
        if "at least" in query_lower or "exactly" in query_lower or "identify" in query_lower:
            # Check if there are lists, numbers, or bullet points in the answer
            if not re.search(r'\d|\-|\*', content_lower):
                penalties += 0.3

        if "explain why" in query_lower or "justify" in query_lower:
            if "because" not in content_lower and "therefore" not in content_lower and "since" not in content_lower:
                penalties += 0.2
                
        if "acknowledge gaps" in query_lower or "limitations" in query_lower:
            if "however" not in content_lower and "limit" not in content_lower and "gap" not in content_lower:
                penalties += 0.3

        return max(0.0, score - penalties)

    # Public API
    def verify_and_score(
        self,
        query: str,
        llm_output: Dict[str, Any],
        pre_reasoning_risk: float,
        self_consistency_score: float = 0.0,
        query_type: str = "SEMI_STRUCTURED",
    ) -> Tuple[TrustScore, float, List[int], VerificationStatusEnum, float, List[VerificationDetail]]:
        """
        Full verification pipeline. Returns:
        (trust_score, completion_score, flagged_indices, verification_status, verification_confidence, verification_details)

        Safe fallback: any internal failure returns conservative scores, never raises.
        """
        try:
            return self._verify_and_score_inner(query, llm_output, pre_reasoning_risk, self_consistency_score, query_type)
        except Exception as exc:
            logger.error("Verification pipeline crashed (safe fallback): %s", exc, exc_info=True)
            return self._fallback_result(pre_reasoning_risk)

    def _verify_and_score_inner(
        self,
        query: str,
        llm_output: Dict[str, Any],
        pre_reasoning_risk: float,
        self_consistency_score: float,
        query_type: str = "SEMI_STRUCTURED",
    ) -> Tuple[TrustScore, float, List[int], VerificationStatusEnum, float, List[VerificationDetail]]:
        """Core verification logic, wrapped by safe fallback in verify_and_score."""

        steps = llm_output.get("reasoning_steps", [])
        final_answer = str(llm_output.get("final_answer", ""))
        all_text = " ".join(str(s.get("content", "")) for s in steps) + " " + final_answer
        
        # Bug E: If LLM output is totally empty/busted
        if not steps and not final_answer:
            trust = TrustScore(
                verification_confidence=0.0,
                reasoning_consistency_score=0.0,
                self_consistency_score=0.0,
                aggregate_score=0.10,  # 10% Trust
                logical_consistency=0.0,
                evidence_alignment=0.0,
                entropy=0.80, # 80% Risk
                contradiction_rate=0.0,
            )
            return trust, 0.0, [], VerificationStatusEnum.FAILED, 0.0, [
                VerificationDetail(
                    method="schema_check",
                    passed=False,
                    confidence=1.0,
                    message="LLM generated zero valid reasoning steps and empty final answer.",
                )
            ]
            
        completion_score = self._compute_instruction_completion(query, all_text)

        # OPEN_ENDED path: heuristic evaluation (no strict verification)
        if query_type == "OPEN_ENDED":
            trust, flagged, status, conf, details = self._heuristic_evaluate(query, steps, final_answer, self_consistency_score)
            return trust, completion_score, flagged, status, conf, details

        # DETERMINISTIC / SEMI_STRUCTURED path: strict verification
        # 1. Mathematical verification
        math_details, flagged_indices = self._verify_math_in_steps(steps, final_answer)

        # 2. Logical consistency (strict assumption alignment for deterministic)
        consistency_score, consistency_details = self._check_logical_consistency(steps)

        # Merge all details
        all_details = math_details + consistency_details

        # 3. Overall status & confidence
        verification_status, verification_confidence = self._compute_verification_status(
            math_details, consistency_score
        )

        # 4. Confidence Inversion Guard
        uncertainty_markers = ["paradox", "bias", "unknown", "cannot detect", "dilemma", "uncertain"]
        if any(m in query.lower() for m in uncertainty_markers) and verification_confidence > 0.8:
            verification_confidence = 0.8  # Cap confidence for highly uncertain prompts
            all_details.append(VerificationDetail(
                method="confidence_inversion_guard",
                passed=False,
                message="Confidence ceiling lowered due to high uncertainty/paradox language in query"
            ))

        # 5. Trust score
        trust = self._compute_trust(
            verification_confidence, verification_status,
            consistency_score, self_consistency_score,
        )

        # 5. Hallucination risk (post-reasoning, grounded)
        hallucination_risk = self._compute_hallucination_risk(
            verification_status, verification_confidence,
            consistency_score, len(flagged_indices), len(steps),
            query_type=query_type,
        )

        # Override the trust entropy with hallucination risk for consistency
        trust.entropy = hallucination_risk

        return trust, completion_score, flagged_indices, verification_status, verification_confidence, all_details

    # OPEN_ENDED heuristic evaluation
    def _heuristic_evaluate(
        self,
        query: str,
        steps: List[Dict[str, Any]],
        final_answer: str,
        self_consistency_score: float,
    ) -> Tuple[TrustScore, List[int], VerificationStatusEnum, float, List[VerificationDetail]]:
        """
        Heuristic evaluation for open-ended / conceptual queries.
        Computes: coverage_score, coherence_score, plausibility_score.
        Checks for evasion and caps trust accordingly.
        """
        details: List[VerificationDetail] = []
        
        all_text = " ".join(str(s.get("content", "")) for s in steps) + " " + final_answer
        all_text_lower = all_text.lower()
        query_lower = query.lower()

        # 1. Evasion check
        evasion_keywords = ["i am a machine learning model", "i cannot", "as an ai", "i am an ai model", "i do not have", "i'm just an ai"]
        is_evasive = False
        if any(kw in all_text_lower for kw in evasion_keywords) and len(all_text) < 300:
            is_evasive = True
            
        if is_evasive:
            details.append(VerificationDetail(
                method="evasion_check",
                passed=False,
                confidence=0.9,
                message="Output contained deflection/evasion phrases without substantive content",
            ))

        # Coverage: did the answer address all parts?
        step_count = len(steps)
        total_content_len = len(all_text)
        
        import re
        clauses = re.split(r'[,;]|\n-|\*', query_lower)
        clauses = [c.strip() for c in clauses if len(c.strip()) > 5]
        
        addressed = 0
        for clause in clauses:
            words = set(w for w in clause.split() if len(w) > 4)
            if words and any(w in all_text_lower for w in words):
                addressed += 1
                
        completeness_ratio = addressed / max(len(clauses), 1)
        
        # Main verb/task missing penalty
        main_tasks = ["construct", "produce", "list", "explain", "justify", "summarize", "derive"]
        task_missing = any(t in query_lower and t not in all_text_lower for t in main_tasks)
        
        coverage_score = min(1.0, 0.4 + completeness_ratio * 0.6)
        if task_missing:
            coverage_score = min(coverage_score, 0.6)

        details.append(VerificationDetail(
            method="heuristic_coverage",
            passed=coverage_score >= 0.5,
            confidence=coverage_score,
            message=f"Coverage ratio {completeness_ratio:.2f}; Task missing: {task_missing}",
        ))

        # Coherence: step-to-step flow (soft contradiction check)
        coherence_score = 1.0
        if len(steps) >= 2:
            for i in range(len(steps) - 1):
                content_a = str(steps[i].get("content", "")).lower()
                content_b = str(steps[i + 1].get("content", "")).lower()
                for pos, neg in self._negation_pairs:
                    if (pos in content_a and neg in content_b) or (neg in content_a and pos in content_b):
                        words_a = set(content_a.split())
                        words_b = set(content_b.split())
                        overlap = words_a & words_b - {"the", "a", "an", "is", "are", "to", "and", "or", "in", "of"}
                        if len(overlap) >= 4:  # Higher threshold for open-ended
                            coherence_score -= 0.15
            coherence_score = max(0.3, coherence_score)

        details.append(VerificationDetail(
            method="heuristic_coherence",
            passed=coherence_score >= 0.6,
            confidence=coherence_score,
            message=f"Step flow coherence: {coherence_score:.2f}",
        ))

        # Plausibility: no obviously incorrect claims
        # For open-ended, we don't have ground truth. Score based on structure.
        plausibility_score = 0.75  # default: plausible
        if final_answer and len(final_answer.strip()) > 20:
            plausibility_score = 0.8
        if step_count >= 2 and all(len(str(s.get("content", ""))) > 10 for s in steps):
            plausibility_score = min(1.0, plausibility_score + 0.1)

        details.append(VerificationDetail(
            method="heuristic_plausibility",
            passed=True,  # Never flag open-ended as implausible without evidence
            confidence=plausibility_score,
            message=f"Structural plausibility: {plausibility_score:.2f}",
        ))

        verification_confidence = round(
            (coverage_score + coherence_score + plausibility_score) / 3.0, 4
        )

        # Trust: different weights for open-ended
        if is_evasive:
            aggregate = 0.4  # Hard cap
            status = VerificationStatusEnum.FAILED
        else:
            aggregate = round(
                coverage_score * 0.4 + coherence_score * 0.3 + plausibility_score * 0.3,
                4,
            )
            # Cap trust at 0.6 if main task missing
            if task_missing:
                aggregate = min(aggregate, 0.6)
            aggregate = max(0.0, min(1.0, aggregate))
            status = VerificationStatusEnum.HEURISTIC

        trust = TrustScore(
            verification_confidence=verification_confidence,
            reasoning_consistency_score=round(coherence_score, 4),
            self_consistency_score=round(self_consistency_score, 4),
            aggregate_score=aggregate,
            logical_consistency=round(coherence_score, 4),
            evidence_alignment=round(coverage_score, 4),
            entropy=round(max(0.0, 1.0 - aggregate), 4),
            contradiction_rate=round(max(0.0, 1.0 - coherence_score), 4),
        )

        return trust, [], status, verification_confidence, details

    def _fallback_result(
        self, pre_reasoning_risk: float
    ) -> Tuple[TrustScore, float, List[int], VerificationStatusEnum, float, List[VerificationDetail]]:
        """Conservative safe fallback when the verification pipeline itself fails."""
        trust = TrustScore(
            verification_confidence=0.0,
            reasoning_consistency_score=0.5,
            self_consistency_score=0.0,
            aggregate_score=0.3,
            logical_consistency=0.5,
            evidence_alignment=0.0,
            entropy=pre_reasoning_risk,
            contradiction_rate=0.0,
        )
        details = [VerificationDetail(
            method="fallback",
            passed=False,
            confidence=0.0,
            message="Verification pipeline encountered an internal error; conservative scores applied",
        )]
        return trust, 0.5, [], VerificationStatusEnum.PARTIAL, 0.0, details
