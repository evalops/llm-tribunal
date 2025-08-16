"""
LLM-as-Evaluator nodes for DAG-based evaluation system.
Provides composable evaluation protocols using multiple LLM critics.
"""

import json
import re
import logging
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import os
try:
    import openai
except Exception:
    class _OpenAIStub:
        class OpenAI:
            pass
    openai = _OpenAIStub()

try:
    from anthropic import Anthropic
except Exception:
    class Anthropic:  # minimal stub
        def __init__(self, *args, **kwargs):
            pass
import os
import requests

from dag_engine import BaseNode, ExecutionContext, NodeResult, NodeStatus
from config import get_config

# Optional Jinja2 templating for prompts
try:
    from jinja2 import Template as JinjaTemplate
    _JINJA_AVAILABLE = True
except Exception:
    _JINJA_AVAILABLE = False


class _RateLimiter:
    """Very simple token-bucket style limiter (per provider)."""
    def __init__(self):
        self.allow = {}
    def acquire(self, key: str):
        # No-op placeholder for now
        return True


class _TokenTracker:
    def __init__(self):
        self.totals = {"prompt": 0, "completion": 0}
    def add(self, prompt_tokens: int, completion_tokens: int):
        self.totals["prompt"] += prompt_tokens
        self.totals["completion"] += completion_tokens


_rate_limiter = _RateLimiter()
_token_tracker = _TokenTracker()

def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))
from config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Scale abstractions for evaluator outputs
class EvaluationScale(ABC):
    """Abstract base class for evaluator output scales."""
    
    @abstractmethod
    def validate(self, value: Any) -> bool:
        """Validate if a value fits this scale."""
        pass
    
    @abstractmethod
    def parse(self, response: str) -> Any:
        """Parse a response string to extract the scale value."""
        pass
    
    @abstractmethod
    def format_prompt(self) -> str:
        """Format scale description for prompts."""
        pass

class RatingScale(EvaluationScale):
    """Numeric rating scale (e.g., 1-5, 1-10)."""
    
    def __init__(self, range_tuple: Tuple[int, int], labels: Optional[Dict[int, str]] = None):
        self.min_val, self.max_val = range_tuple
        self.labels = labels or {}
    
    def validate(self, value: Any) -> bool:
        return isinstance(value, int) and self.min_val <= value <= self.max_val
    
    def parse(self, response: str) -> int:
        # Look for rating patterns first (more specific)
        rating_patterns = [
            r'(?:rate|rating|score)[^\d]*(\d+)',  # "rate this 4", "rating: 3"
            r'(\d+)\s*(?:out of|/)\s*\d+',        # "4 out of 5", "3/5"
            r'(\d+)\s*(?:stars?|points?)',        # "4 stars", "3 points"
        ]
        
        for pattern in rating_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                num = int(match)
                if self.validate(num):
                    return num
                else:
                    # Found a rating but it's out of range
                    raise ValueError(f"Rating {num} is outside valid range {self.min_val}-{self.max_val}")
        
        # Fallback: look for any numbers in the response
        numbers = re.findall(r'\b(\d+)\b', response)
        for num_str in numbers:
            num = int(num_str)
            if self.validate(num):
                return num
        
        raise ValueError(f"No valid score found in range {self.min_val}-{self.max_val}")
    
    def format_prompt(self) -> str:
        prompt = f"Respond with a score between {self.min_val} and {self.max_val}."
        if self.labels:
            for score, label in self.labels.items():
                prompt += f"\n- {score}: {label}"
        return prompt

class BinaryScale(EvaluationScale):
    """Binary scale (yes/no, true/false)."""
    
    def __init__(self, true_values: Optional[List[str]] = None, false_values: Optional[List[str]] = None):
        self.true_values = true_values or ["yes", "true", "correct", "valid"]
        self.false_values = false_values or ["no", "false", "incorrect", "invalid"]
    
    def validate(self, value: Any) -> bool:
        return isinstance(value, bool)
    
    def parse(self, response: str) -> bool:
        response_lower = response.lower().strip()
        
        for true_val in self.true_values:
            if true_val in response_lower:
                return True
        
        for false_val in self.false_values:
            if false_val in response_lower:
                return False
        
        raise ValueError(f"No boolean value found in response")
    
    def format_prompt(self) -> str:
        return f"Respond with '{self.true_values[0]}' or '{self.false_values[0]}'."

class CategoryScale(EvaluationScale):
    """Categorical scale with predefined categories."""
    
    def __init__(self, categories: List[str]):
        self.categories = categories
    
    def validate(self, value: Any) -> bool:
        return isinstance(value, str) and value in self.categories
    
    def parse(self, response: str) -> str:
        response_lower = response.lower().strip()
        
        # Look for exact matches first
        for category in self.categories:
            if category.lower() in response_lower:
                return category
        
        raise ValueError(f"No valid category found from: {self.categories}")
    
    def format_prompt(self) -> str:
        return f"Respond with one of: {', '.join(self.categories)}"

@dataclass
class EvaluationResult:
    """Result from an evaluator node."""
    score: Any
    explanation: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class EvaluatorNode(BaseNode):
    """Individual evaluator that assesses input and produces scored output."""
    
    def __init__(self, node_id: str, scale: EvaluationScale, prompt_template: str,
                 model: str = "gpt-4o-mini", temperature: float = 0.0,
                 explanation: bool = False, retries: int = 3,
                 template_engine: str = "auto",
                 inputs: Optional[List[str]] = None, outputs: Optional[List[str]] = None):
        # Allow explicit IO contracts for YAML-driven DAGs
        super().__init__(node_id, inputs=inputs or [], outputs=outputs or [])
        self.scale = scale
        self.prompt_template = prompt_template
        self.model = model
        self.temperature = temperature
        self.explanation = explanation
        self.retries = retries
        self.template_engine = template_engine
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with the given prompt."""
        cfg = get_config()
        backoff = getattr(cfg.evaluation, "retry_delay_seconds", 1.0)
        # Simple on-disk response cache
        def _cache_dir() -> Optional[str]:
            base = os.getenv("LLM_CACHE_DIR", os.path.join(".dag_cache", "llm_responses"))
            try:
                os.makedirs(base, exist_ok=True)
            except Exception:
                return None
            return base
        def _cache_key(model: str, temperature: float, prompt: str) -> str:
            h = hashlib.sha256()
            h.update(model.encode())
            h.update(str(temperature).encode())
            h.update(prompt.encode())
            return h.hexdigest()
        def _cache_get(model: str, temperature: float, prompt: str) -> Optional[str]:
            if os.getenv("LLM_CACHE", "1") in ("0", "false", "False"):
                return None
            d = _cache_dir()
            if not d:
                return None
            k = _cache_key(model, temperature, prompt)
            path = os.path.join(d, f"{k}.txt")
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception:
                    return None
            return None
        def _cache_set(model: str, temperature: float, prompt: str, text: str) -> None:
            if os.getenv("LLM_CACHE", "1") in ("0", "false", "False"):
                return
            d = _cache_dir()
            if not d:
                return
            k = _cache_key(model, temperature, prompt)
            path = os.path.join(d, f"{k}.txt")
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(text)
            except Exception:
                pass
        for attempt in range(self.retries):
            try:
                # Use trivial rate limiter hook
                _rate_limiter.acquire(self.model.split(":")[0])

                # Cache lookup
                cached = _cache_get(self.model, self.temperature, prompt)
                if cached is not None:
                    return cached

                if self.model.startswith("gpt") and ":" not in self.model:
                    client = openai.OpenAI(
                        base_url=cfg.api.openai_base_url or None,
                        timeout=cfg.api.default_timeout or None,
                    )
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature
                    )
                    text = response.choices[0].message.content.strip()
                    _token_tracker.add(_estimate_tokens(prompt), _estimate_tokens(text))
                    _cache_set(self.model, self.temperature, prompt, text)
                    return text
                
                elif self.model.startswith("claude"):
                    client = Anthropic(
                        base_url=cfg.api.anthropic_base_url or None,
                        timeout=cfg.api.default_timeout or None,
                    )
                    response = client.messages.create(
                        model=self.model,
                        max_tokens=1000,
                        temperature=self.temperature,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    text = response.content[0].text.strip()
                    _token_tracker.add(_estimate_tokens(prompt), _estimate_tokens(text))
                    _cache_set(self.model, self.temperature, prompt, text)
                    return text
                else:
                    # Ollama local model
                    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                    url = f"{base_url.rstrip('/')}/api/generate"
                    use_stream = os.getenv("OLLAMA_STREAM", "0").lower() in ("1", "true", "yes")
                    if use_stream:
                        try:
                            resp = requests.post(url, json={
                                "model": self.model,
                                "prompt": prompt,
                                "stream": True
                            }, timeout=cfg.api.default_timeout, stream=True)
                            resp.raise_for_status()
                            chunks = []
                            for line in resp.iter_lines(decode_unicode=True):
                                if not line:
                                    continue
                                try:
                                    obj = json.loads(line)
                                except Exception:
                                    continue
                                piece = obj.get("response")
                                if piece:
                                    chunks.append(piece)
                                    if os.getenv("OLLAMA_STREAM_PRINT", "0") in ("1", "true", "True"):
                                        try:
                                            print(piece, end="", flush=True)
                                        except Exception:
                                            pass
                                if obj.get("done"):
                                    break
                            text = ("".join(chunks)).strip()
                            _token_tracker.add(_estimate_tokens(prompt), _estimate_tokens(text))
                            _cache_set(self.model, self.temperature, prompt, text)
                            return text
                        except Exception:
                            # Fallback to non-streaming if streaming not supported/available
                            pass
                    resp = requests.post(url, json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False
                    }, timeout=cfg.api.default_timeout)
                    resp.raise_for_status()
                    data = resp.json()
                    text = data.get("response", "").strip()
                    _token_tracker.add(_estimate_tokens(prompt), _estimate_tokens(text))
                    _cache_set(self.model, self.temperature, prompt, text)
                    return text
                
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt == self.retries - 1:
                    raise e
                # Exponential backoff
                try:
                    time.sleep(backoff * (2 ** attempt))
                except Exception:
                    pass
        
        raise Exception("All LLM call attempts failed")
    
    def run(self, context: ExecutionContext) -> NodeResult:
        """Run the evaluator node."""
        try:
            # First, extract required variables from the prompt template
            import re
            required_vars = set(re.findall(r'\{(\w+)\}', self.prompt_template))
            
            # Check for missing required variables
            missing_vars = []
            for var in required_vars:
                if var not in context.artifacts:
                    missing_vars.append(var)
            
            if missing_vars:
                return NodeResult(
                    status=NodeStatus.FAILED,
                    error=f"Required template variables not found in context: {missing_vars}"
                )
            
            # Format the prompt with context artifacts
            # Safer prompt formatting: tolerate missing keys and support dotted access
            class _DotMap(dict):
                def __init__(self, data):
                    super().__init__()
                    self._data = data
                def __missing__(self, key):
                    return ""
                def __getitem__(self, key):
                    try:
                        return self._resolve(key)
                    except Exception:
                        return ""
                def _resolve(self, path: str):
                    cur = self._data
                    for part in path.split('.'):
                        if isinstance(cur, dict) and part in cur:
                            cur = cur[part]
                        elif hasattr(cur, part):
                            cur = getattr(cur, part)
                        else:
                            return ""
                    return cur

            format_dict: Dict[str, Any] = {}
            format_dict.update(context.artifacts)
            
            # Add previous node results for referencing
            for node_id, result in context.node_results.items():
                if result.status == NodeStatus.SUCCESS:
                    format_dict[node_id] = result.outputs
            
            # Jinja2 optional templating
            if self.template_engine == "jinja2" and _JINJA_AVAILABLE:
                try:
                    prompt = JinjaTemplate(self.prompt_template).render(**format_dict)
                except Exception:
                    prompt = self.prompt_template
            else:
                try:
                    prompt = self.prompt_template.format_map(_DotMap(format_dict))
                except Exception:
                    # As a last resort, use the raw template
                    prompt = self.prompt_template
            
            # Add scale instructions
            if self.explanation:
                prompt += f"\n\n{self.scale.format_prompt()}\n\nProvide your reasoning first, then give your final answer."
            else:
                prompt += f"\n\n{self.scale.format_prompt()}"
            
            # Get LLM response
            response = self._call_llm(prompt)
            
            # Parse the response
            score = self.scale.parse(response)
            
            # Extract explanation if requested
            explanation = None
            if self.explanation:
                # Try to separate reasoning from final answer
                lines = response.split('\n')
                explanation = '\n'.join(lines[:-1]).strip() if len(lines) > 1 else response
            
            result = EvaluationResult(
                score=score,
                explanation=explanation,
                metadata={"model": self.model, "temperature": self.temperature}
            )
            
            logger.info(f"Evaluator {self.id} scored: {score}")
            return NodeResult(
                status=NodeStatus.SUCCESS,
                outputs={
                    "score": result.score,
                    "explanation": result.explanation,
                    "metadata": {**(result.metadata or {}), "tokens": getattr(_token_tracker, 'totals', {})},
                    "raw_response": response
                }
            )
            
        except Exception as e:
            logger.error(f"Evaluator {self.id} failed: {e}")
            return NodeResult(
                status=NodeStatus.FAILED,
                outputs={},
                error=str(e)
            )

class MultiCriticNode(BaseNode):
    """Node that facilitates deliberation between multiple critic evaluators."""
    
    def __init__(self, node_id: str, critic_models: List[str], debate_prompt: str,
                 rounds: int = 2, scale: Optional[EvaluationScale] = None,
                 inputs: Optional[List[str]] = None, outputs: Optional[List[str]] = None):
        super().__init__(node_id, inputs=inputs or [], outputs=outputs or [])
        self.critic_models = critic_models
        self.debate_prompt = debate_prompt
        self.rounds = rounds
        self.scale = scale or BinaryScale()
    
    def run(self, context: ExecutionContext) -> NodeResult:
        """Run deliberation between multiple critics."""
        try:
            debate_history = []
            critic_positions = {}
            evaluators = {}
            
            # Initial positions from each critic
            for i, model in enumerate(self.critic_models):
                critic_id = f"critic_{i}"
                
                # Create an evaluator for this model (reuse across rounds)
                evaluator = EvaluatorNode(
                    f"{self.id}_{critic_id}",
                    self.scale,
                    self.debate_prompt,
                    model=model,
                    explanation=True
                )
                evaluators[critic_id] = evaluator
                
                # Get initial judgment
                result = evaluator.run(context)
                if result.status == NodeStatus.SUCCESS:
                    critic_positions[critic_id] = {
                        "model": model,
                        "score": result.outputs["score"],
                        "reasoning": result.outputs["explanation"]
                    }
                
                    debate_history.append({
                        "round": 0,
                        "judge": critic_id,
                        "position": result.outputs["score"],
                        "reasoning": result.outputs["explanation"]
                    })
            
            # Debate rounds
            for round_num in range(1, self.rounds + 1):
                for i, model in enumerate(self.critic_models):
                    critic_id = f"critic_{i}"
                    
                    # Format debate context
                    other_positions = []
                    for other_id, pos in critic_positions.items():
                        if other_id != critic_id:
                            other_positions.append(f"{other_id}: {pos['score']} - {pos['reasoning']}")
                    
                    # Create new context with debate info
                    debate_context = ExecutionContext()
                    debate_context.artifacts.update(context.artifacts)
                    debate_context.artifacts["other_critic_positions"] = "\n".join(other_positions)
                    debate_context.artifacts["previous_position"] = critic_positions[critic_id]["score"]
                    
                    # Updated debate prompt
                    updated_prompt = f"""{self.debate_prompt}

PREVIOUS POSITIONS FROM OTHER CRITICS:
{debate_context.artifacts["other_critic_positions"]}

YOUR PREVIOUS POSITION: {debate_context.artifacts["previous_position"]}

Consider the other critics' reasoning. You may maintain or revise your position."""
                    
                    # Reuse the same evaluator instance; update prompt template
                    evaluator = evaluators.get(critic_id)
                    if evaluator is None:
                        evaluator = EvaluatorNode(
                            f"{self.id}_{critic_id}",
                            self.scale,
                            updated_prompt,
                            model=model,
                            explanation=True
                        )
                        evaluators[critic_id] = evaluator
                    else:
                        try:
                            evaluator.prompt_template = updated_prompt
                        except Exception:
                            pass
                    result = evaluator.run(debate_context)
                    if result.status == NodeStatus.SUCCESS:
                        critic_positions[critic_id] = {
                            "model": model,
                            "score": result.outputs["score"],
                            "reasoning": result.outputs["explanation"]
                        }
                        
                        debate_history.append({
                            "round": round_num,
                            "judge": critic_id,
                            "position": result.outputs["score"],
                            "reasoning": result.outputs["explanation"]
                        })
            
            logger.info(f"Debate {self.id} completed with {len(debate_history)} exchanges")
            return NodeResult(
                status=NodeStatus.SUCCESS,
                outputs={
                    "debate_history": debate_history,
                    "final_positions": critic_positions,
                    "critics": self.critic_models,
                    "rounds": self.rounds
                }
            )
            
        except Exception as e:
            logger.error(f"Debate {self.id} failed: {e}")
            return NodeResult(
                status=NodeStatus.FAILED,
                outputs={},
                error=str(e)
            )

class SynthesisNode(BaseNode):
    """Node that synthesizes multiple evaluator results into a final assessment."""
    
    def __init__(self, node_id: str, aggregation_method: str = "majority_vote",
                 inputs: Optional[List[str]] = None, outputs: Optional[List[str]] = None,
                 method: Optional[str] = None):
        super().__init__(node_id, inputs=inputs or [], outputs=outputs or [])
        self.aggregation_method = method or aggregation_method
    
    def _majority_vote(self, scores: List[Any]) -> Any:
        """Simple majority vote aggregation."""
        from collections import Counter
        vote_counts = Counter(scores)
        return vote_counts.most_common(1)[0][0]
    
    def _weighted_average(self, scores: List[float], weights: Optional[List[float]] = None) -> float:
        """Weighted average for numeric scores."""
        if weights is None:
            weights = [1.0] * len(scores)
        return sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    
    def run(self, context: ExecutionContext) -> NodeResult:
        """Synthesize evaluator results."""
        try:
            # Extract scores from various possible input formats
            scores = []
            explanations = []
            
            # Handle debate results from previous nodes
            for node_id, node_result in context.node_results.items():
                if node_result.status == NodeStatus.SUCCESS:
                    outputs = node_result.outputs
                    
                    # Handle debate results
                    if "final_positions" in outputs:
                        for judge_id, position in outputs["final_positions"].items():
                            scores.append(position["score"])
                            explanations.append(position["reasoning"])
            
                    # Handle direct evaluator results
                    elif "score" in outputs:
                        scores.append(outputs["score"])
                        if "explanation" in outputs:
                            explanations.append(outputs["explanation"])
            
            # Fallback: check if a 'deliberation' artifact is present
            if not scores and context.has_artifact("deliberation"):
                deliberation = context.get_artifact("deliberation")
                if isinstance(deliberation, dict):
                    for round_data in deliberation.values():
                        if isinstance(round_data, dict):
                            scores.extend(list(round_data.values()))
            
            if not scores:
                raise ValueError("No scores found for aggregation")
            
            # Apply aggregation method
            if self.aggregation_method == "majority_vote":
                final_score = self._majority_vote(scores)
            elif self.aggregation_method == "average" and all(isinstance(s, (int, float)) for s in scores):
                final_score = self._weighted_average(scores)
            else:
                # Default to majority vote
                final_score = self._majority_vote(scores)
            
            # Calculate confidence (percentage agreement)
            from collections import Counter
            vote_counts = Counter(scores)
            max_count = max(vote_counts.values())
            confidence = max_count / len(scores)
            
            logger.info(f"Aggregation {self.id} result: {final_score} (confidence: {confidence:.2f})")
            return NodeResult(
                status=NodeStatus.SUCCESS,
                outputs={
                    "final_score": final_score,
                    "individual_scores": scores,
                    "confidence": confidence,
                    "vote_distribution": dict(vote_counts),
                    "explanations": explanations,
                    "aggregation_method": self.aggregation_method
                }
            )
            
        except Exception as e:
            logger.error(f"Aggregation {self.id} failed: {e}")
            return NodeResult(
                status=NodeStatus.FAILED,
                outputs={},
                error=str(e)
            )

class ValidationNode(BaseNode):
    """Node that validates the reasoning of previous evaluations."""
    
    def __init__(self, node_id: str, verification_prompt: str,
                 verifier_model: str = "gpt-4o-mini",
                 inputs: Optional[List[str]] = None, outputs: Optional[List[str]] = None):
        super().__init__(node_id, inputs=inputs or [], outputs=outputs or [])
        self.verification_prompt = verification_prompt
        self.verifier_model = verifier_model
    
    def run(self, context: ExecutionContext) -> NodeResult:
        """Validate the reasoning of a previous evaluation."""
        try:
            # Create validator evaluator
            validator = EvaluatorNode(
                f"{self.id}_validator",
                BinaryScale(),
                self.verification_prompt,
                model=self.verifier_model,
                explanation=True
            )
            
            # Run validation
            result = validator.run(context)
            
            if result.status == NodeStatus.SUCCESS:
                verification_passed = result.outputs["score"]
                
                logger.info(f"Verification {self.id}: {'PASSED' if verification_passed else 'FAILED'}")
                return NodeResult(
                    status=NodeStatus.SUCCESS,
                    outputs={
                        "verification_passed": verification_passed,
                        "verification_reasoning": result.outputs.get("explanation"),
                        "verifier_model": self.verifier_model,
                        # Backwards-compatible synonyms
                        "validation": verification_passed,
                        "explanation": result.outputs.get("explanation"),
                    }
                )
            else:
                return result
            
        except Exception as e:
            logger.error(f"Verification {self.id} failed: {e}")
            return NodeResult(
                status=NodeStatus.FAILED,
                outputs={},
                error=str(e)
            )
