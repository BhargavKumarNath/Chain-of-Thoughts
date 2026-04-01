import json
import logging
import os
import re
from collections import Counter
from typing import Any, Dict, List

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.schemas.reasoning import StrategyEnum, DifficultyLevel

logger = logging.getLogger(__name__)


class LLMReasoningStepPayload(BaseModel):
    step_index: int = Field(..., ge=1)
    content: str = Field(..., min_length=1)
    assumptions: List[str] = Field(default_factory=list)


class LLMReasoningPayload(BaseModel):
    reasoning_steps: List[LLMReasoningStepPayload] = Field(default_factory=list)
    final_answer: str = Field(..., min_length=1)


class LLMGenerator:
    """
    Interfaces with Ollama through async HTTP calls.
    Enforces API-level JSON mode and strict schema validation.
    Optimized for latency: dynamic context windows and token limits per strategy.
    """

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_URL", "http://ollama:11434")
        self.model = os.getenv("LLM_MODEL", "llama3.2:3b")
        self.timeout_seconds = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
        self.base_num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
        self.multi_sample_n = 3  # Forced to 3 for MULTI_SAMPLE

    # Dynamic context / token limits per strategy
    _CTX_LIMITS = {
        StrategyEnum.DIRECT: 200,
        StrategyEnum.SHORT_COT: 500,
        StrategyEnum.LONG_COT: 1500,
        StrategyEnum.TREE_OF_THOUGHTS: 2000,
        StrategyEnum.MULTI_SAMPLE: 500,
        StrategyEnum.EXTERNAL_SOLVER: 3072,
    }

    def _get_num_ctx(self, strategy: StrategyEnum) -> int:
        return self._CTX_LIMITS.get(strategy, self.base_num_ctx)

    def _build_system_prompt(self, strategy: StrategyEnum) -> str:
        strategy_hint = {
            StrategyEnum.DIRECT: (
                "Answer directly with 0 reasoning steps. 'reasoning_steps' array MUST be empty []. "
                "The final_answer must be a clear, complete sentence."
            ),
            StrategyEnum.SHORT_COT: (
                "Produce a short chain of reasoning with exactly 2 to 5 steps max. "
                "The final_answer must be a complete sentence summarising the conclusion."
            ),
            StrategyEnum.LONG_COT: (
                "Use a thorough, step-by-step chain of reasoning (max 15 steps). "
                "Each step must build logically on the previous one without circular repetition. "
                "Address second-order effects fully before proceeding to third-order effects. Proceed strictly by analytical depth. "
                "For optimisation or allocation problems, compute exact quantities per entity and verify the total adds up. "
                "For probabilistic problems, show each intermediate probability and state the formula used. "
                "The final_answer must be a self-contained, fully explained conclusion — "
                "never just a bare number. It must restate what the answer means in context."
            ),
            StrategyEnum.TREE_OF_THOUGHTS: (
                "Consider at least two candidate approaches before committing. Branching factor <= 3, depth <= 4. "
                "Each branch must be explicitly labeled with its decision node (e.g., `[Branch: <path name>]`). "
                "Steps within a branch must progress firmly toward that branch's terminal outcome. "
                "The final_answer must clearly state which approach was chosen and why."
            ),
            StrategyEnum.MULTI_SAMPLE: (
                "Produce one high-quality reasoning sample with max 5 steps. "
                "The final_answer must be a complete, standalone answer."
            ),
            StrategyEnum.EXTERNAL_SOLVER: (
                "Show all equations explicitly so they can be verified deterministically. "
                "Label every variable. The final_answer must state the numeric result with units."
            ),
        }.get(strategy, "Reason carefully and keep assumptions explicit. The final_answer must be a complete sentence.")

        return (
            "You are ReasonOps, a strict analytical reasoning engine.\n"
            "Return ONLY JSON with NO markdown, NO backticks, and NO additional keys.\n"
            "Required schema:\n"
            "{\n"
            '  "reasoning_steps": [\n'
            '    {"step_index": 1, "content": "string", "assumptions": ["string"]}\n'
            "  ],\n"
            '  "final_answer": "string"\n'
            "}\n"
            "Rules:\n"
            "- step_index must start at 1 and increase by 1.\n"
            "- assumptions must always be an array (empty array [] is fine, never omit it).\n"
            "- final_answer must NEVER be empty or a single bare number without context.\n"
            "- Each step MUST introduce a new constraint/variable OR logically resolve a previous one. DO NOT merely restate prior conclusions.\n"
            "- If the query involves scheduling, assignment, or resource allocation: Step 1 MUST strictly enumerate ONLY the explicit entities (e.g., Tasks A-K, Workers 1-4) named in the prompt. Treat this as a strictly closed world. Tasks MUST be distributed in parallel across available workers simultaneously, NOT assigned sequentially to a single worker.\n"
            f"- {strategy_hint}\n"
        )

    def _dump_model(self, model: BaseModel) -> Dict[str, Any]:
        if hasattr(model, "model_dump"):
            return model.model_dump()
        return model.dict()

    def _validate_payload(self, payload: Dict[str, Any]) -> LLMReasoningPayload:
        if hasattr(LLMReasoningPayload, "model_validate"):
            return LLMReasoningPayload.model_validate(payload)
        return LLMReasoningPayload.parse_obj(payload)

    def _extract_json_lenient(self, raw_content: str) -> Dict[str, Any]:
        """Tries to extract JSON, or forces extraction of steps via regex if severely broken."""
        raw_content = (raw_content or "").strip()
        if not raw_content:
            raise ValueError("Empty content")
            
        try:
            return json.loads(raw_content)
        except json.JSONDecodeError:
            pass
            
        # Try standard cleanup
        no_fence = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content, flags=re.IGNORECASE | re.DOTALL).strip()
        if no_fence:
            try:
                return json.loads(no_fence)
            except json.JSONDecodeError:
                pass
            
        # Try finding the largest valid JSON block
        first_brace = no_fence.find("{")
        last_brace = no_fence.rfind("}")
        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            try:
                return json.loads(no_fence[first_brace:last_brace + 1])
            except json.JSONDecodeError:
                pass
                
        # --- Extreme Lenient Parsing (Regex Fallback) ---
        # If we reached here, JSON is broken (likely truncated from timeout).
        extracted_steps = []
        matches = re.finditer(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_content)
        for i, match in enumerate(matches, 1):
            val = match.group(1).replace('\\"', '"').replace('\\n', '\n')
            extracted_steps.append({
                "step_index": i,
                "content": val,
                "assumptions": []
            })
            
        # Try to find a final answer
        final_ans = "Partial generation - final answer not reached."
        ans_match = re.search(r'"final_answer"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_content)
        if ans_match:
            final_ans = ans_match.group(1).replace('\\"', '"').replace('\\n', '\n')
            
        if not extracted_steps and not ans_match:
            # Maybe the whole output is just plain text? Wrap it.
            clean_text = raw_content[:1000] + "..." if len(raw_content) > 1000 else raw_content
            extracted_steps.append({
                "step_index": 1,
                "content": clean_text,
                "assumptions": []
            })
            
        return {
            "reasoning_steps": extracted_steps,
            "final_answer": final_ans
        }

    def _normalize_payload(self, payload: Dict[str, Any], query: str = "") -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}

        steps = payload.get("reasoning_steps")
        if steps is None:
            steps = payload.get("steps")

        final_answer = payload.get("final_answer")
        if final_answer is None:
            final_answer = payload.get("answer")

        step_items: List[Dict[str, Any]] = []
        if isinstance(steps, list):
            for idx, step in enumerate(steps, start=1):
                if isinstance(step, str):
                    step_items.append(
                        {"step_index": idx, "content": step.strip(), "assumptions": []}
                    )
                    continue

                if not isinstance(step, dict):
                    continue

                assumptions = step.get("assumptions", [])
                if isinstance(assumptions, str):
                    assumptions = [assumptions]
                if not isinstance(assumptions, list):
                    assumptions = []

                step_items.append(
                    {
                        "step_index": int(step.get("step_index", idx)),
                        "content": str(step.get("content", "")).strip(),
                        "assumptions": [str(a) for a in assumptions if str(a).strip()],
                    }
                )

        def jaccard_similarity(s1: str, s2: str) -> float:
            set1 = set(s1.lower().split())
            set2 = set(s2.lower().split())
            if not set1 or not set2:
                return 0.0
            return len(set1.intersection(set2)) / len(set1.union(set2))

        # Hallucination Guard Prep
        is_scheduling = any(kw in query.lower() for kw in ["schedule", "makespan", "allocate", "assign"])
        illegal_entities = []
        if is_scheduling:
            q_upper = query.upper()
            for i in range(ord('A'), ord('Z') + 1):
                letter = chr(i)
                if not re.search(r'\b' + letter + r'\b', q_upper):
                    illegal_entities.append(f"Task {letter}")
                    illegal_entities.append(f"task {letter.lower()}")
            for i in range(5, 20):
                if str(i) not in query:
                    illegal_entities.append(f"Worker {i}")
                    illegal_entities.append(f"worker {i}")

        filtered_steps = []
        redundant_streak = 0
        loop_halted = False

        for step in step_items:
            if loop_halted:
                break
            content = step["content"]
            
            # Bug 1 Hallucination Guard
            if is_scheduling and not loop_halted:
                hallucinated = False
                for illegal in illegal_entities:
                    if illegal in content:
                        hallucinated = True
                        break
                if hallucinated:
                    step["content"] = "[HALLUCINATED] " + content
                    loop_halted = True
                    final_answer = "[HALLUCINATED] Unknown entity introduced. Halting trace."
                    filtered_steps.append(step)
                    continue

            is_redundant = False
            for prev_step in filtered_steps:
                if jaccard_similarity(content, prev_step["content"]) > 0.85:
                    is_redundant = True
                    break
            
            if is_redundant:
                step["content"] = "[REDUNDANT] " + content
                redundant_streak += 1
                if redundant_streak >= 3:
                    loop_halted = True
                    final_answer = "[REASONING_LOOP_DETECTED] " + str(final_answer)
            else:
                redundant_streak = 0
            
            filtered_steps.append(step)

        normalized["reasoning_steps"] = filtered_steps
        normalized["final_answer"] = str(final_answer or "").strip()
        return normalized

    def _fallback_payload(self, reason: str, tokens_used: int = 0) -> Dict[str, Any]:
        return {
            "reasoning_steps": [
                {
                    "step_index": 1,
                    "content": f"LLM output validation fallback triggered: {reason}",
                    "assumptions": [],
                }
            ],
            "final_answer": "Unable to produce a fully validated structured response.",
            "tokens_used": max(tokens_used, 0),
        }

    def _temperature_for(self, strategy: StrategyEnum, sample_index: int) -> float:
        if strategy == StrategyEnum.MULTI_SAMPLE:
            return min(0.8, 0.55 + (sample_index * 0.05))
        return {
            StrategyEnum.DIRECT: 0.10,
            StrategyEnum.SHORT_COT: 0.10,
            StrategyEnum.LONG_COT: 0.10,
            StrategyEnum.TREE_OF_THOUGHTS: 0.30,
            StrategyEnum.EXTERNAL_SOLVER: 0.05,
        }.get(strategy, 0.15)

    async def _generate_single(
        self, query: str, strategy: StrategyEnum, sample_index: int = 0
    ) -> Dict[str, Any]:
        num_ctx = self._get_num_ctx(strategy)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._build_system_prompt(strategy)},
                {"role": "user", "content": query},
            ],
            "format": "json",
            "stream": True,
            "options": {
                "temperature": self._temperature_for(strategy, sample_index),
                "num_ctx": num_ctx,
                "num_predict": num_ctx,
            },
        }

        timeout_map = {
            StrategyEnum.DIRECT: 5.0,
            StrategyEnum.SHORT_COT: 10.0,
            StrategyEnum.LONG_COT: 90.0,
            StrategyEnum.TREE_OF_THOUGHTS: 120.0,
            StrategyEnum.MULTI_SAMPLE: 45.0,
            StrategyEnum.EXTERNAL_SOLVER: 60.0,
        }
        request_timeout = timeout_map.get(strategy, self.timeout_seconds)

        raw_content = ""
        tokens_used = 0
        is_timeout = False

        async with httpx.AsyncClient(timeout=request_timeout) as client:
            try:
                async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            raw_content += chunk.get("message", {}).get("content", "")
                            if chunk.get("done"):
                                tokens_used = chunk.get("eval_count", 0) + chunk.get("prompt_eval_count", 0)
                        except json.JSONDecodeError:
                            pass
            except httpx.ReadTimeout:
                logger.warning(f"Ollama timeout at {request_timeout}s. Stored {len(raw_content)} chars.")
                is_timeout = True
                if not raw_content.strip():
                    return self._fallback_payload("[TIMEOUT] Generation hit latency ceiling with no output.", tokens_used=num_ctx)
            except httpx.HTTPError as exc:
                logger.error("Ollama API error: %s", exc)
                return self._fallback_payload("ollama_http_error", tokens_used=0)

        # Bug B & C: Use lenient parsing and don't strict-fail on ValidationError
        try:
            extracted = self._extract_json_lenient(raw_content)
            normalized = self._normalize_payload(extracted, query)
            
            structured = {
                "reasoning_steps": normalized.get("reasoning_steps", []),
                "final_answer": normalized.get("final_answer", ""),
                "tokens_used": tokens_used or num_ctx
            }
            
            if not structured["reasoning_steps"] and not structured["final_answer"].strip():
                 return self._fallback_payload("empty_or_broken_output", tokens_used)
                 
            if is_timeout:
                step_count = len(structured["reasoning_steps"])
                structured["reasoning_steps"].append({
                    "step_index": step_count + 1,
                    "content": f"[TIMEOUT] Generation halted at step {step_count} — partial trace shown above.",
                    "assumptions": []
                })
                structured["_is_timeout"] = True
                
            return structured
        except Exception as exc:
            logger.warning("Invalid raw LLM output (sample %s): %s", sample_index, exc)
            return self._fallback_payload("schema_validation_failed", tokens_used=tokens_used)

    def _choose_consensus_candidate(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not candidates:
            return self._fallback_payload("no_candidates")

        answer_votes = Counter(
            str(candidate.get("final_answer", "")).strip().lower()
            for candidate in candidates
            if str(candidate.get("final_answer", "")).strip()
        )

        if answer_votes:
            winning_answer = answer_votes.most_common(1)[0][0]
            winning_candidates = [
                candidate
                for candidate in candidates
                if str(candidate.get("final_answer", "")).strip().lower() == winning_answer
            ]
            if winning_candidates:
                return max(
                    winning_candidates,
                    key=lambda c: (
                        len(c.get("reasoning_steps", [])),
                        len(str(c.get("final_answer", ""))),
                    ),
                )

        return max(
            candidates,
            key=lambda c: (
                len(c.get("reasoning_steps", [])),
                len(str(c.get("final_answer", ""))),
            ),
        )

    def compute_self_consistency_score(self, candidates: List[Dict[str, Any]]) -> float:
        """
        Compute agreement ratio among multiple reasoning sample outputs.
        Returns 1.0 if all agree, 0.0 if all differ.
        """
        if len(candidates) <= 1:
            return 0.0  # Cannot measure self-consistency with < 2 samples

        answers = [str(c.get("final_answer", "")).strip().lower() for c in candidates if c.get("final_answer")]
        if not answers:
            return 0.0

        answer_counts = Counter(answers)
        most_common_count = answer_counts.most_common(1)[0][1]
        return round(most_common_count / len(answers), 4)

    async def generate(self, query: str, strategy: StrategyEnum) -> Dict[str, Any]:
        """
        Executes async calls to Ollama with strict JSON mode and resilient validation.
        MULTI_SAMPLE executes sequentially (N=2 by default) to avoid GPU OOM.
        """
        if strategy != StrategyEnum.MULTI_SAMPLE:
            return await self._generate_single(query, strategy, sample_index=0)

        candidates: List[Dict[str, Any]] = []
        total_tokens = 0

        for idx in range(self.multi_sample_n):
            candidate = await self._generate_single(query, strategy, sample_index=idx + 1)
            total_tokens += int(candidate.get("tokens_used", 0) or 0)
            candidates.append(candidate)

        self_consistency = self.compute_self_consistency_score(candidates)
        selected = self._choose_consensus_candidate(candidates)
        
        # Merge sample traces into one unified list for UI observation
        merged_steps = []
        for i, cand in enumerate(candidates, 1):
            cand_steps = cand.get("reasoning_steps", [])
            merged_steps.append({
                "step_index": len(merged_steps) + 1,
                "content": f"[Sample {i}] started",
                "assumptions": []
            })
            for s in cand_steps:
               s_copy = dict(s)
               s_copy["step_index"] = len(merged_steps) + 1
               merged_steps.append(s_copy)

        selected["tokens_used"] = total_tokens
        selected["_self_consistency_score"] = self_consistency
        selected["_all_candidates"] = candidates
        
        # Surface divergence flag
        if self_consistency < 1.0:
            selected["_divergence_flag"] = True
            
        # Bug 2 Synthesis Step & Overwrite final_answer
        majority_conclusion = str(selected.get("final_answer", "")).strip()
        final_conclusions = [str(c.get("final_answer", "")).strip() for c in candidates]
        
        if self_consistency == 1.0:
            synthesis_content = f"Samples 1-3 compared. Consensus Reached: {majority_conclusion}"
            selected["final_answer"] = majority_conclusion
        else:
            dissenting_list = [c for c in final_conclusions if c.lower() != majority_conclusion.lower()]
            dissent = dissenting_list[0] if dissenting_list else "No clear dissent."
            synthesis_content = f"Samples 1-3 compared. Divergence detected.\nMajority conclusion: {majority_conclusion}\nDissenting sample conclusion: {dissent}"
            selected["final_answer"] = synthesis_content

        merged_steps.append({
            "step_index": len(merged_steps) + 1,
            "content": f"[Synthesis] {synthesis_content}",
            "assumptions": []
        })
        selected["reasoning_steps"] = merged_steps
            
        return selected

    async def generate_for_self_consistency(
        self, query: str, strategy: StrategyEnum, n: int = 2
    ) -> Dict[str, Any]:
        """
        Generate multiple samples for self-consistency checking (conditional).
        Returns the consensus candidate with self_consistency metadata.
        """
        candidates: List[Dict[str, Any]] = []
        total_tokens = 0

        for idx in range(n):
            candidate = await self._generate_single(query, strategy, sample_index=idx + 1)
            total_tokens += int(candidate.get("tokens_used", 0) or 0)
            candidates.append(candidate)

        self_consistency = self.compute_self_consistency_score(candidates)
        selected = self._choose_consensus_candidate(candidates)
        selected["tokens_used"] = total_tokens
        selected["_self_consistency_score"] = self_consistency
        return selected
