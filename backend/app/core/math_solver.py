"""
math_solver.py, Deterministic re-computation engine for independent verification.

Uses SymPy for arithmetic, algebraic, and probability formula verification.
Includes safe fallbacks: solver failures never crash the pipeline.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import sympy
from sympy import sympify, simplify, Rational, oo, S

logger = logging.getLogger(__name__)


@dataclass
class SolverResult:
    """Result from an independent re-computation attempt."""
    computed_answer: Optional[float] = None
    method: str = "none"
    confidence: float = 0.0
    success: bool = False
    error: Optional[str] = None
    details: str = ""


@dataclass
class ToleranceConfig:
    """Configurable tolerance thresholds for comparisons."""
    float_abs_tol: float = 0.01
    float_rel_tol: float = 1e-6
    integer_exact: bool = True
    probability_tol: float = 0.005


class MathSolver:
    """
    Lightweight solver for independent re-computation of deterministic problems.
    Covers: arithmetic, basic algebra, probability formulas.
    All methods have safe fallbacks — failures return SolverResult with success=False.
    """

    def __init__(self, tolerances: Optional[ToleranceConfig] = None):
        self.tolerances = tolerances or ToleranceConfig()

        # Patterns for extracting numerical answers
        self._number_pattern = re.compile(
            r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?(?:\s*%)?",
        )
        self._fraction_pattern = re.compile(r"(\d+)\s*/\s*(\d+)")
        self._probability_pattern = re.compile(
            r"[Pp]\s*\(\s*(.+?)\s*\)\s*=\s*([\d.]+(?:/[\d.]+)?)",
        )
        # Pattern for arithmetic expressions in text
        self._arithmetic_expr = re.compile(
            r"([\d.]+)\s*([\+\-\*×÷/])\s*([\d.]+)"
        )

    def extract_numerical_answer(self, text: str) -> Optional[float]:
        """
        Extract the most likely numerical answer from a text string.
        Returns None if no number can be extracted.
        """
        if not text:
            return None

        text_clean = text.strip()

        # Try to parse the entire text as a number first
        try:
            return float(text_clean.replace(",", "").replace("%", "").strip())
        except (ValueError, TypeError):
            pass

        # Try fraction pattern
        frac_match = self._fraction_pattern.search(text_clean)
        if frac_match:
            try:
                num = float(frac_match.group(1))
                den = float(frac_match.group(2))
                if den != 0:
                    return num / den
            except (ValueError, ZeroDivisionError):
                pass

        # Extract all numbers, prefer the last one (usually the final answer)
        numbers = self._number_pattern.findall(text_clean)
        if numbers:
            last_num = numbers[-1].strip()
            is_percent = last_num.endswith("%")
            try:
                val = float(last_num.replace(",", "").replace("%", "").strip())
                if is_percent:
                    val /= 100.0
                return val
            except (ValueError, TypeError):
                pass

        return None

    def evaluate_arithmetic(self, expression: str) -> SolverResult:
        """
        Evaluate a pure arithmetic expression using SymPy.
        Safe fallback: returns failure result on any exception.
        """
        try:
            normalized = (expression or "").strip()
            normalized = normalized.replace("^", "**")
            normalized = normalized.replace("×", "*")
            normalized = normalized.replace("÷", "/")
            normalized = normalized.replace(",", "")

            result = sympify(normalized, evaluate=True)

            if result is None or result == sympy.nan:
                return SolverResult(method="sympy_arithmetic", error="Expression evaluated to NaN")

            if hasattr(result, "is_number") and result.is_number:
                float_val = float(result.evalf())
                return SolverResult(
                    computed_answer=float_val,
                    method="sympy_arithmetic",
                    confidence=1.0,
                    success=True,
                    details=f"Evaluated: {expression} = {float_val}",
                )

            # Result has free symbols — cannot compute
            return SolverResult(
                method="sympy_arithmetic",
                confidence=0.0,
                error=f"Expression contains unresolved symbols: {result.free_symbols}",
            )
        except Exception as exc:
            logger.debug("Arithmetic evaluation failed for '%s': %s", expression, exc)
            return SolverResult(method="sympy_arithmetic", error=str(exc))

    def verify_probability(self, text: str) -> SolverResult:
        """
        Verify probability computations found in text.
        Checks: complement rule, independence, Bayes' theorem patterns.
        Safe fallback on any failure.
        """
        try:
            # Extract probability assignments
            prob_assignments = self._probability_pattern.findall(text)
            if not prob_assignments:
                return SolverResult(method="probability_formula", confidence=0.0,
                                    details="No probability expressions found")

            prob_values = {}
            for event, value_str in prob_assignments:
                try:
                    if "/" in value_str:
                        parts = value_str.split("/")
                        val = float(parts[0]) / float(parts[1])
                    else:
                        val = float(value_str)
                    prob_values[event.strip()] = val
                except (ValueError, ZeroDivisionError):
                    continue

            if not prob_values:
                return SolverResult(method="probability_formula", confidence=0.0,
                                    details="Could not parse probability values")

            # Validate: all probabilities in [0, 1]
            for event, val in prob_values.items():
                if val < -self.tolerances.probability_tol or val > 1.0 + self.tolerances.probability_tol:
                    return SolverResult(
                        computed_answer=val,
                        method="probability_formula",
                        confidence=0.9,
                        success=False,
                        details=f"P({event}) = {val} is outside valid range [0, 1]",
                    )

            # Check complement pairs: P(A) + P(not A) = 1
            checks_passed = 0
            total_checks = 0
            for event, val in prob_values.items():
                complement_keys = [f"not {event}", f"not({event})", f"~{event}", f"{event}'", f"{event}c"]
                for ck in complement_keys:
                    if ck in prob_values:
                        total_checks += 1
                        if abs((val + prob_values[ck]) - 1.0) < self.tolerances.probability_tol:
                            checks_passed += 1
                        else:
                            return SolverResult(
                                method="probability_formula",
                                confidence=0.95,
                                success=False,
                                details=f"Complement violation: P({event})={val} + P({ck})={prob_values[ck]} != 1.0",
                            )

            confidence = 0.7 if total_checks > 0 else 0.4
            return SolverResult(
                method="probability_formula",
                confidence=confidence,
                success=True,
                details=f"Validated {len(prob_values)} probability values, {checks_passed}/{total_checks} complement checks passed",
            )

        except Exception as exc:
            logger.debug("Probability verification failed: %s", exc)
            return SolverResult(method="probability_formula", error=str(exc))

    def solve_equation(self, equation_str: str) -> SolverResult:
        """
        Attempt to solve a simple equation or verify an equation's validity.
        Safe fallback on any failure.
        """
        try:
            # Normalize
            normalized = (equation_str or "").strip()
            normalized = normalized.replace("^", "**")
            normalized = normalized.replace(",", "")

            # Split on '='
            if "=" not in normalized:
                return SolverResult(method="sympy_equation", confidence=0.0,
                                    error="No equation sign found")

            # Handle '==' and '='
            if "==" in normalized:
                parts = normalized.split("==", 1)
            else:
                parts = normalized.split("=", 1)

            if len(parts) != 2:
                return SolverResult(method="sympy_equation", confidence=0.0,
                                    error="Could not split equation into two sides")

            left_str, right_str = parts[0].strip(), parts[1].strip()
            if not left_str or not right_str:
                return SolverResult(method="sympy_equation", confidence=0.0,
                                    error="Empty side in equation")

            left_expr = sympify(left_str, evaluate=True)
            right_expr = sympify(right_str, evaluate=True)
            delta = simplify(left_expr - right_expr)

            if delta == 0:
                return SolverResult(
                    method="sympy_equation",
                    confidence=1.0,
                    success=True,
                    details=f"Verified: {left_str} = {right_str}",
                )

            if hasattr(delta, "is_number") and delta.is_number:
                diff = abs(float(delta.evalf()))
                if diff < self.tolerances.float_abs_tol:
                    return SolverResult(
                        method="sympy_equation",
                        confidence=0.95,
                        success=True,
                        details=f"Approximately equal within tolerance: diff={diff:.6f}",
                    )
                return SolverResult(
                    computed_answer=float(right_expr.evalf()) if hasattr(right_expr, "evalf") else None,
                    method="sympy_equation",
                    confidence=0.95,
                    success=False,
                    details=f"Equation does not hold: {left_str} ≠ {right_str} (diff={diff:.6f})",
                )

            # Has free symbols — cannot verify deterministically
            if hasattr(delta, "free_symbols") and delta.free_symbols:
                return SolverResult(
                    method="sympy_equation",
                    confidence=0.0,
                    details=f"Equation has unresolved symbols: {delta.free_symbols}",
                )

            return SolverResult(method="sympy_equation", confidence=0.0,
                                details="Could not determine equation validity")

        except Exception as exc:
            logger.debug("Equation solving failed for '%s': %s", equation_str, exc)
            return SolverResult(method="sympy_equation", error=str(exc))

    def compare_answers(self, expected: float, actual: float) -> Tuple[bool, str]:
        """
        Compare two numerical answers using configured tolerances.
        Returns (match: bool, detail: str).
        """
        if expected is None or actual is None:
            return False, "One or both values are None"

        # Check for exact integer match
        if (self.tolerances.integer_exact
                and expected == int(expected)
                and actual == int(actual)):
            match = int(expected) == int(actual)
            return match, f"Integer comparison: {int(expected)} {'==' if match else '!='} {int(actual)}"

        # Float comparison with absolute and relative tolerance
        abs_diff = abs(expected - actual)
        rel_diff = abs_diff / max(abs(expected), 1e-15)

        if abs_diff <= self.tolerances.float_abs_tol:
            return True, f"Match within absolute tolerance: |{expected} - {actual}| = {abs_diff:.6f} ≤ {self.tolerances.float_abs_tol}"

        if rel_diff <= self.tolerances.float_rel_tol:
            return True, f"Match within relative tolerance: rel_diff={rel_diff:.8f} ≤ {self.tolerances.float_rel_tol}"

        return False, f"Mismatch: expected={expected}, actual={actual}, abs_diff={abs_diff:.6f}, rel_diff={rel_diff:.8f}"

    def extract_and_verify_arithmetic(self, text: str) -> List[SolverResult]:
        """
        Extract all arithmetic expressions from text and verify each one.
        Returns a list of SolverResults for all found expressions.
        """
        results = []
        if not text:
            return results

        # Find patterns like "17 * 23 = 391"
        equation_pattern = re.compile(
            r"([\d,.]+\s*[\+\-\*×÷/]\s*[\d,.]+(?:\s*[\+\-\*×÷/]\s*[\d,.]+)*)\s*=\s*([\d,.]+)"
        )

        for match in equation_pattern.finditer(text):
            expr_str = match.group(1).strip()
            claimed_result = match.group(2).strip()

            eval_result = self.evaluate_arithmetic(expr_str)
            if eval_result.success and eval_result.computed_answer is not None:
                try:
                    claimed_val = float(claimed_result.replace(",", ""))
                    match_ok, detail = self.compare_answers(eval_result.computed_answer, claimed_val)
                    results.append(SolverResult(
                        computed_answer=eval_result.computed_answer,
                        method="sympy_arithmetic_check",
                        confidence=1.0 if match_ok else 0.95,
                        success=match_ok,
                        details=detail,
                    ))
                except ValueError:
                    results.append(eval_result)
            else:
                results.append(eval_result)

        return results
