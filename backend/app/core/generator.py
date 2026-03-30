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
    reasoning_steps: List[LLMReasoningStepPayload] = Field(..., min_length=1)
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
        self.multi_sample_n = max(1, int(os.getenv("MULTI_SAMPLE_N", "2")))

    # Dynamic context / token limits per strategy
    _CTX_LIMITS = {
        StrategyEnum.DIRECT: 2048,
        StrategyEnum.SHORT_COT: 3072,
        StrategyEnum.LONG_COT: 4096,
        StrategyEnum.TREE_OF_THOUGHTS: 4096,
        StrategyEnum.MULTI_SAMPLE: 3072,
        StrategyEnum.EXTERNAL_SOLVER: 3072,
    }

    def _get_num_ctx(self, strategy: StrategyEnum) -> int:
        return self._CTX_LIMITS.get(strategy, self.base_num_ctx)

    def _build_system_prompt(self, strategy: StrategyEnum) -> str:
        strategy_hint = {
            StrategyEnum.DIRECT: (
                "Answer directly in 1-2 concise steps. "
                "The final_answer must be a clear, complete sentence."
            ),
            StrategyEnum.SHORT_COT: (
                "Produce a short chain of reasoning with 2-4 steps. "
                "The final_answer must be a complete sentence summarising the conclusion."
            ),
            StrategyEnum.LONG_COT: (
                "Use a thorough, step-by-step chain of reasoning. "
                "Each step must build logically on the previous one. "
                "For optimisation or allocation problems, compute exact quantities per entity and verify the total adds up. "
                "For probabilistic problems, show each intermediate probability and state the formula used. "
                "The final_answer must be a self-contained, fully explained conclusion — "
                "never just a bare number. It must restate what the answer means in context."
            ),
            StrategyEnum.TREE_OF_THOUGHTS: (
                "Consider at least two candidate approaches before committing to the best one. "
                "The final_answer must clearly state which approach was chosen and why."
            ),
            StrategyEnum.MULTI_SAMPLE: (
                "Produce one high-quality reasoning sample. "
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
            "- reasoning_steps must be a non-empty array of at least 2 items.\n"
            "- step_index must start at 1 and increase by 1.\n"
            "- assumptions must always be an array (empty array [] is fine, never omit it).\n"
            "- final_answer must NEVER be empty or a single bare number without context.\n"
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

    def _extract_json_candidate(self, raw_content: str) -> Dict[str, Any]:
        raw_content = (raw_content or "").strip()
        if not raw_content:
            raise json.JSONDecodeError("Empty content", "", 0)

        try:
            return json.loads(raw_content)
        except json.JSONDecodeError:
            pass

        no_fence = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content, flags=re.IGNORECASE | re.DOTALL).strip()
        if no_fence and no_fence != raw_content:
            try:
                return json.loads(no_fence)
            except json.JSONDecodeError:
                pass

        first_brace = no_fence.find("{")
        last_brace = no_fence.rfind("}")
        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            sliced = no_fence[first_brace:last_brace + 1]
            return json.loads(sliced)

        raise json.JSONDecodeError("Unable to extract JSON object", raw_content, 0)

    def _normalize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
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

        normalized["reasoning_steps"] = step_items
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
            "stream": False,
            "options": {
                "temperature": self._temperature_for(strategy, sample_index),
                "num_ctx": num_ctx,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            try:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPError as exc:
                logger.error("Ollama API error: %s", exc)
                return self._fallback_payload("ollama_http_error")

        tokens_used = int((data or {}).get("eval_count", 0) or 0) + int((data or {}).get("prompt_eval_count", 0) or 0)
        raw_content = ((data or {}).get("message") or {}).get("content", "")

        try:
            extracted = self._extract_json_candidate(raw_content)
            normalized = self._normalize_payload(extracted)
            validated = self._validate_payload(normalized)
            structured = self._dump_model(validated)
            structured["tokens_used"] = tokens_used
            return structured
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
            logger.warning("Invalid structured LLM output (sample %s): %s", sample_index, exc)
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
        selected["tokens_used"] = total_tokens
        selected["_self_consistency_score"] = self_consistency
        selected["_all_candidates"] = candidates
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
