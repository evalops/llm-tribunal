"""
Gateway-specific DAG nodes for real-time safety evaluation.

These nodes are optimized for low-latency execution in production environments.
"""

import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from dag_engine import BaseNode, ExecutionContext, NodeResult, NodeStatus, register_node
from judge_nodes import EvaluatorNode, RatingScale, BinaryScale


@register_node("RealTimeProxyNode")
class RealTimeProxyNode(BaseNode):
    """
    Main entry point for real-time request processing.
    Orchestrates the safety evaluation pipeline for incoming requests.
    """
    
    def __init__(self, node_id: str, inputs: Optional[List[str]] = None, 
                 outputs: Optional[List[str]] = None, **kwargs):
        super().__init__(node_id, inputs=inputs or [], outputs=outputs or ["action", "risk_score", "explanation"])
        self.risk_threshold = kwargs.get("risk_threshold", 4)
        self.fail_open = kwargs.get("fail_open", True)  # Allow on evaluation failure
    
    def run(self, context: ExecutionContext) -> NodeResult:
        """Process incoming request and determine action."""
        try:
            request_content = context.get_artifact("request_content")
            
            # Basic validation
            if not request_content or len(request_content.strip()) == 0:
                return NodeResult(
                    status=NodeStatus.SUCCESS,
                    outputs={
                        "action": "allow",
                        "risk_score": 1,
                        "explanation": "Empty request content",
                        "processing_time_ms": 0
                    }
                )
            
            # Extract metadata
            metadata = context.artifacts.get("request_metadata", {})
            
            return NodeResult(
                status=NodeStatus.SUCCESS,
                outputs={
                    "action": "evaluate",  # Pass to downstream evaluators
                    "risk_score": 1,
                    "explanation": "Request accepted for evaluation",
                    "metadata": metadata,
                    "content_length": len(request_content),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            if self.fail_open:
                return NodeResult(
                    status=NodeStatus.SUCCESS,
                    outputs={
                        "action": "allow",
                        "risk_score": 1,
                        "explanation": f"Evaluation error (fail-open): {str(e)}",
                        "error": str(e)
                    }
                )
            else:
                return NodeResult(
                    status=NodeStatus.FAILED,
                    error=f"Real-time proxy failed: {str(e)}"
                )


@register_node("SafetyPolicyNode") 
class SafetyPolicyNode(BaseNode):
    """
    Applies safety policies based on evaluation results.
    Makes the final allow/block decision.
    """
    
    def __init__(self, node_id: str, inputs: Optional[List[str]] = None,
                 outputs: Optional[List[str]] = None, **kwargs):
        super().__init__(node_id, inputs=inputs or ["evaluation_result"], 
                        outputs=outputs or ["action", "risk_score", "reason"])
        self.risk_threshold = kwargs.get("risk_threshold", 4)
        self.policies = kwargs.get("policies", {})
    
    def run(self, context: ExecutionContext) -> NodeResult:
        """Apply safety policies to evaluation results."""
        try:
            # Collect risk scores from evaluator nodes
            max_risk_score = 1
            explanations = []
            evaluation_details = {}
            
            # Check results from previous evaluation nodes
            for node_id, node_result in context.node_results.items():
                if node_result.status == NodeStatus.SUCCESS:
                    outputs = node_result.outputs
                    
                    if "score" in outputs:
                        risk_score = outputs["score"]
                        max_risk_score = max(max_risk_score, risk_score)
                        evaluation_details[node_id] = {
                            "score": risk_score,
                            "explanation": outputs.get("explanation", "")
                        }
                        
                        if outputs.get("explanation"):
                            explanations.append(f"{node_id}: {outputs['explanation']}")
            
            # Apply policy decision
            action = "allow"
            reason = "No safety concerns detected"
            
            if max_risk_score >= self.risk_threshold:
                action = "block"
                reason = f"Risk score {max_risk_score} exceeds threshold {self.risk_threshold}"
            
            # Check for specific policy overrides
            for policy_name, policy_rule in self.policies.items():
                if self._check_policy_rule(policy_rule, evaluation_details, context):
                    action = policy_rule.get("action", action)
                    reason = policy_rule.get("reason", reason)
                    break
            
            return NodeResult(
                status=NodeStatus.SUCCESS,
                outputs={
                    "action": action,
                    "risk_score": max_risk_score,
                    "reason": reason,
                    "threshold": self.risk_threshold,
                    "evaluation_details": evaluation_details,
                    "explanations": explanations
                }
            )
            
        except Exception as e:
            # Fail-safe: allow request
            return NodeResult(
                status=NodeStatus.SUCCESS,
                outputs={
                    "action": "allow",
                    "risk_score": 1,
                    "reason": f"Policy evaluation error (fail-safe): {str(e)}",
                    "error": str(e)
                }
            )
    
    def _check_policy_rule(self, rule: Dict[str, Any], evaluation_details: Dict[str, Any], 
                          context: ExecutionContext) -> bool:
        """Check if a specific policy rule applies."""
        try:
            # Example rule format:
            # {
            #   "condition": "any_score_above",
            #   "threshold": 3,
            #   "action": "block",
            #   "reason": "Medium risk content detected"
            # }
            
            condition = rule.get("condition", "")
            threshold = rule.get("threshold", 0)
            
            if condition == "any_score_above":
                return any(detail["score"] > threshold for detail in evaluation_details.values())
            elif condition == "all_scores_above":
                return all(detail["score"] > threshold for detail in evaluation_details.values())
            elif condition == "content_contains":
                keywords = rule.get("keywords", [])
                content = context.artifacts.get("request_content", "")
                return any(keyword.lower() in content.lower() for keyword in keywords)
            
            return False
            
        except Exception:
            return False


@register_node("FastThreatDetectorNode")
class FastThreatDetectorNode(BaseNode):
    """
    Fast, lightweight threat detection using heuristics and regex patterns.
    Designed for sub-10ms latency to filter obvious threats before expensive evaluation.
    """
    
    def __init__(self, node_id: str, inputs: Optional[List[str]] = None,
                 outputs: Optional[List[str]] = None, **kwargs):
        super().__init__(node_id, inputs=inputs or ["request_content"], 
                        outputs=outputs or ["score", "explanation", "threats_detected"])
        
        # Load threat patterns
        self.threat_patterns = kwargs.get("threat_patterns", self._default_patterns())
        self.enable_keyword_detection = kwargs.get("enable_keyword_detection", True)
        self.enable_pattern_detection = kwargs.get("enable_pattern_detection", True)
    
    def _default_patterns(self) -> Dict[str, Any]:
        """Default threat detection patterns."""
        return {
            "prompt_injection": {
                "patterns": [
                    r"ignore\s+(all\s+)?previous\s+instructions",
                    r"you\s+are\s+now\s+a\s+",
                    r"disregard\s+your\s+programming",
                    r"override\s+safety\s+protocols",
                    r"\[SYSTEM\]|\[ADMIN\]|\[ROOT\]"
                ],
                "risk_score": 4,
                "description": "Potential prompt injection attempt"
            },
            "jailbreak": {
                "patterns": [
                    r"DAN\s+mode",
                    r"developer\s+mode",
                    r"unrestricted\s+mode",
                    r"bypass\s+filters",
                    r"ignore\s+safety"
                ],
                "risk_score": 4,
                "description": "Potential jailbreak attempt"
            },
            "sensitive_data_extraction": {
                "patterns": [
                    r"show\s+me\s+your\s+training\s+data",
                    r"reveal\s+your\s+system\s+prompt",
                    r"what\s+are\s+your\s+instructions",
                    r"personal\s+information\s+from\s+training"
                ],
                "risk_score": 3,
                "description": "Attempt to extract sensitive information"
            },
            "harmful_content": {
                "keywords": [
                    "bomb", "explosive", "poison", "hack", "crack",
                    "illegal", "drugs", "weapon", "violence"
                ],
                "risk_score": 3,
                "description": "Potentially harmful content keywords detected"
            }
        }
    
    def run(self, context: ExecutionContext) -> NodeResult:
        """Run fast threat detection."""
        start_time = time.time()
        
        try:
            request_content = context.get_artifact("request_content")
            
            if not request_content:
                return NodeResult(
                    status=NodeStatus.SUCCESS,
                    outputs={
                        "score": 1,
                        "explanation": "No content to analyze",
                        "threats_detected": [],
                        "processing_time_ms": 0
                    }
                )
            
            threats_detected = []
            max_risk_score = 1
            
            content_lower = request_content.lower()
            
            # Pattern-based detection
            if self.enable_pattern_detection:
                import re
                for threat_type, config in self.threat_patterns.items():
                    if "patterns" in config:
                        for pattern in config["patterns"]:
                            if re.search(pattern, content_lower, re.IGNORECASE):
                                threats_detected.append({
                                    "type": threat_type,
                                    "pattern": pattern,
                                    "risk_score": config["risk_score"],
                                    "description": config["description"]
                                })
                                max_risk_score = max(max_risk_score, config["risk_score"])
            
            # Keyword-based detection
            if self.enable_keyword_detection:
                for threat_type, config in self.threat_patterns.items():
                    if "keywords" in config:
                        detected_keywords = []
                        for keyword in config["keywords"]:
                            if keyword.lower() in content_lower:
                                detected_keywords.append(keyword)
                        
                        if detected_keywords:
                            threats_detected.append({
                                "type": threat_type,
                                "keywords": detected_keywords,
                                "risk_score": config["risk_score"],
                                "description": config["description"]
                            })
                            max_risk_score = max(max_risk_score, config["risk_score"])
            
            processing_time = int((time.time() - start_time) * 1000)
            
            explanation = "No threats detected"
            if threats_detected:
                threat_types = [t["type"] for t in threats_detected]
                explanation = f"Detected threats: {', '.join(set(threat_types))}"
            
            return NodeResult(
                status=NodeStatus.SUCCESS,
                outputs={
                    "score": max_risk_score,
                    "explanation": explanation,
                    "threats_detected": threats_detected,
                    "processing_time_ms": processing_time,
                    "content_analyzed": len(request_content)
                }
            )
            
        except Exception as e:
            return NodeResult(
                status=NodeStatus.FAILED,
                error=f"Fast threat detection failed: {str(e)}"
            )


@register_node("LightweightEvaluatorNode")
class LightweightEvaluatorNode(BaseNode):
    """
    Lightweight version of EvaluatorNode optimized for real-time use.
    Uses shorter prompts and faster models when possible.
    """
    
    def __init__(self, node_id: str, inputs: Optional[List[str]] = None,
                 outputs: Optional[List[str]] = None, **kwargs):
        super().__init__(node_id, inputs=inputs or ["request_content"], 
                        outputs=outputs or ["score", "explanation"])
        
        self.evaluation_prompt = kwargs.get("evaluation_prompt", self._default_prompt())
        self.model = kwargs.get("model", "gpt-oss:20b")
        self.scale = RatingScale((1, 5))
        self.max_retries = kwargs.get("max_retries", 1)  # Reduced for speed
    
    def _default_prompt(self) -> str:
        """Default lightweight evaluation prompt."""
        return """Quickly assess the safety risk of this user request on a scale of 1-5:

1 = Safe - normal, legitimate request
2 = Low risk - minor concerns  
3 = Moderate risk - some safety issues
4 = High risk - clear policy violations
5 = Critical risk - immediate safety threat

Request: {request_content}

Provide only the risk score (1-5):"""
    
    def run(self, context: ExecutionContext) -> NodeResult:
        """Run lightweight safety evaluation."""
        try:
            request_content = context.get_artifact("request_content")
            
            if not request_content:
                return NodeResult(
                    status=NodeStatus.SUCCESS,
                    outputs={
                        "score": 1,
                        "explanation": "No content to evaluate"
                    }
                )
            
            # Use the existing EvaluatorNode but with optimized settings
            evaluator = EvaluatorNode(
                node_id=f"{self.id}_evaluator",
                scale=self.scale,
                prompt_template=self.evaluation_prompt,
                model=self.model,
                temperature=0.0,  # Deterministic for consistency
                explanation=False,  # Skip explanation for speed
                retries=self.max_retries
            )
            
            # Create a minimal context with just the content
            eval_context = ExecutionContext()
            eval_context.set_artifact("request_content", request_content)
            
            result = evaluator.run(eval_context)
            
            if result.status == NodeStatus.SUCCESS:
                return NodeResult(
                    status=NodeStatus.SUCCESS,
                    outputs={
                        "score": result.outputs.get("score", 1),
                        "explanation": result.outputs.get("explanation", "Lightweight evaluation"),
                        "model": self.model
                    }
                )
            else:
                # Fail-safe: return low risk score
                return NodeResult(
                    status=NodeStatus.SUCCESS,
                    outputs={
                        "score": 2,
                        "explanation": "Evaluation failed, assigned low risk score",
                        "error": result.error
                    }
                )
            
        except Exception as e:
            # Fail-safe: return low risk score
            return NodeResult(
                status=NodeStatus.SUCCESS,
                outputs={
                    "score": 2,
                    "explanation": f"Evaluation error, assigned low risk score: {str(e)}",
                    "error": str(e)
                }
            )


@register_node("RequestEnricherNode")
class RequestEnricherNode(BaseNode):
    """
    Enriches incoming requests with metadata and context for evaluation.
    """
    
    def __init__(self, node_id: str, inputs: Optional[List[str]] = None,
                 outputs: Optional[List[str]] = None, **kwargs):
        super().__init__(node_id, inputs=inputs or ["request_content", "request_metadata"], 
                        outputs=outputs or ["enriched_content", "analysis"])
    
    def run(self, context: ExecutionContext) -> NodeResult:
        """Enrich request with additional context."""
        try:
            request_content = context.get_artifact("request_content")
            request_metadata = context.artifacts.get("request_metadata", {})
            
            # Basic content analysis
            analysis = {
                "character_count": len(request_content) if request_content else 0,
                "word_count": len(request_content.split()) if request_content else 0,
                "line_count": len(request_content.splitlines()) if request_content else 0,
                "has_code": self._detect_code(request_content) if request_content else False,
                "has_urls": self._detect_urls(request_content) if request_content else False,
                "language_detected": "en",  # Simplified
                "request_type": self._classify_request_type(request_content) if request_content else "unknown"
            }
            
            # Create enriched content with metadata
            enriched = {
                "original_content": request_content,
                "metadata": request_metadata,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
            
            return NodeResult(
                status=NodeStatus.SUCCESS,
                outputs={
                    "enriched_content": enriched,
                    "analysis": analysis
                }
            )
            
        except Exception as e:
            return NodeResult(
                status=NodeStatus.FAILED,
                error=f"Request enrichment failed: {str(e)}"
            )
    
    def _detect_code(self, content: str) -> bool:
        """Simple code detection."""
        code_indicators = [
            "def ", "function ", "class ", "import ", "from ",
            "console.log", "print(", "echo ", "SELECT ",
            "#!/", "<script", "<?php"
        ]
        return any(indicator in content for indicator in code_indicators)
    
    def _detect_urls(self, content: str) -> bool:
        """Simple URL detection."""
        import re
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return bool(re.search(url_pattern, content))
    
    def _classify_request_type(self, content: str) -> str:
        """Simple request type classification."""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["how", "what", "why", "when", "where", "?"]):
            return "question"
        elif any(word in content_lower for word in ["create", "generate", "write", "make"]):
            return "generation"
        elif any(word in content_lower for word in ["explain", "describe", "tell me about"]):
            return "explanation"
        elif any(word in content_lower for word in ["code", "program", "script", "function"]):
            return "coding"
        else:
            return "other"
