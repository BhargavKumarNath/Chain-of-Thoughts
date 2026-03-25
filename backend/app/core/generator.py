import os
import json
import httpx
import logging
from typing import Dict, Any, List
from pydantic import ValidationError
from app.schemas.reasoning import StrategyEnum, ReasoningStep

logger = logging.getLogger(__name__)

class LLMGenerator:
    """
    Interfaces with local Ollama instance. 
    Enforces structured JSON generation and handles memory-constrained sampling.
    """
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_URL", "http://ollama:11434")
        self.model = os.getenv("LLM_MODEL", "llama3") 

    def _build_system_prompt(self, strategy: StrategyEnum) -> str:
        base_prompt = (
            "You are ReasonOps, a strict analytical reasoning engine. "
            "You must respond ONLY in valid JSON. Do not include markdown formatting, code blocks, or conversational text. "
            "Your JSON must exactly match this schema:\n"
            "{\n"
            '  "reasoning_steps":[\n'
            '    {"step_index": 1, "content": "step description", "assumptions": ["assumption1"]}\n'
            "  ],\n"
            '  "final_answer": "your definitive answer"\n'
            "}\n"
        )

        if strategy == StrategyEnum.DIRECT:
            return base_prompt + "Provide a direct, factual answer with minimal steps."
        elif strategy == StrategyEnum.LONG_COT:
            return base_prompt + "Break down the problem extensively. Verify edge cases in your steps."
        
        return base_prompt + "Think step-by-step and explicitly state your assumptions."

    async def generate(self, query: str, strategy: StrategyEnum) -> Dict[str, Any]:
        """
        Executes an async call to Ollama enforcing JSON output.
        """
        system_prompt = self._build_system_prompt(strategy)
        
        payload = {
            "model": self.model,
            "prompt": f"System: {system_prompt}\n\nUser Query: {query}",
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.2 if strategy != StrategyEnum.MULTI_SAMPLE else 0.7,
                "num_ctx": 4096 
            }
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(f"{self.base_url}/api/generate", json=payload)
                response.raise_for_status()
                data = response.json()
                
                # Parse the guaranteed JSON string from Ollama
                result_dict = json.loads(data.get("response", "{}"))
                
                # Attach token telemetry directly from Ollama
                result_dict["tokens_used"] = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
                
                return result_dict
            
            except httpx.HTTPError as e:
                logger.error(f"Ollama API Error: {e}")
                raise
            except json.JSONDecodeError:
                logger.error("LLM failed to produce valid JSON.")
                # Fallback to prevent complete pipeline crash
                return {
                    "reasoning_steps":[{"step_index": 1, "content": "Failed to parse LLM JSON output.", "assumptions": []}],
                    "final_answer": "Error generating structured response.",
                    "tokens_used": 0
                }
