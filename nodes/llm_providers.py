#!/usr/bin/env python3
"""
LLM provider setup and configuration nodes.
"""

import os
import dspy
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from dag_engine import BaseNode, NodeResult, NodeStatus, ExecutionContext, register_node
from config import get_config

# Load environment variables
load_dotenv()


@register_node("DSPySetup")
class DSPySetupNode(BaseNode):
    """Setup DSPy with specified language model provider."""
    
    def run(self, context: ExecutionContext) -> NodeResult:
        try:
            config = get_config()
            provider = self.config.get("provider", "openai")
            model = self.config.get("model", config.evaluation.default_model)
            
            if provider == "openai":
                api_key = self.config.get("api_key") or config.api.openai_api_key
                if not api_key:
                    return NodeResult(
                        status=NodeStatus.FAILED,
                        error="OpenAI API key not found. Set OPENAI_API_KEY environment variable or provide in config."
                    )
                
                base_url = self.config.get("base_url") or config.api.openai_base_url
                
                # Create OpenAI language model
                lm_kwargs = {
                    "model": model,
                    "api_key": api_key,
                    "max_tokens": self.config.get("max_tokens", config.evaluation.default_max_tokens),
                    "temperature": self.config.get("temperature", config.evaluation.default_temperature)
                }
                
                if base_url:
                    lm_kwargs["base_url"] = base_url
                
                lm = dspy.OpenAI(**lm_kwargs)
                
            elif provider == "anthropic":
                api_key = self.config.get("api_key") or config.api.anthropic_api_key
                if not api_key:
                    return NodeResult(
                        status=NodeStatus.FAILED,
                        error="Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable or provide in config."
                    )
                
                # Create Anthropic language model
                lm = dspy.Claude(
                    model=model,
                    api_key=api_key,
                    max_tokens=self.config.get("max_tokens", config.evaluation.default_max_tokens),
                    temperature=self.config.get("temperature", config.evaluation.default_temperature)
                )
                
            elif provider == "ollama":
                base_url = self.config.get("base_url", "http://localhost:11434")
                
                # Create Ollama language model using unified DSPy LM interface
                lm = dspy.LM(
                    model=f"ollama/{model}",
                    max_tokens=self.config.get("max_tokens", config.evaluation.default_max_tokens),
                    temperature=self.config.get("temperature", config.evaluation.default_temperature),
                    **({'api_base': base_url} if base_url != "http://localhost:11434" else {})
                )
                
            else:
                return NodeResult(
                    status=NodeStatus.FAILED,
                    error=f"Unsupported provider: {provider}. Supported: openai, anthropic, ollama"
                )
            
            # Configure DSPy with the language model
            dspy.configure(lm=lm)
            
            outputs = {
                "dspy_lm": lm,
                "provider": provider,
                "model": model
            }
            
            context.set_artifact("model_info", {
                "provider": provider,
                "model": model,
                "max_tokens": lm_kwargs.get("max_tokens") if provider == "openai" else self.config.get("max_tokens"),
                "temperature": lm_kwargs.get("temperature") if provider == "openai" else self.config.get("temperature")
            })
            
            return NodeResult(
                status=NodeStatus.SUCCESS,
                outputs=outputs
            )
            
        except Exception as e:
            return NodeResult(
                status=NodeStatus.FAILED,
                error=f"Failed to setup DSPy: {str(e)}"
            )


@register_node("SetupOllama")
class SetupOllamaNode(BaseNode):
    """Setup Ollama local language model."""
    
    def run(self, context: ExecutionContext) -> NodeResult:
        try:
            model = self.config.get("model", "llama3.2")
            base_url = self.config.get("base_url", "http://localhost:11434")
            
            # Test Ollama connection (prefer requests if available; fallback to urllib)
            url = f"{base_url}/api/tags"
            def _fail(msg):
                return NodeResult(status=NodeStatus.FAILED, error=msg)
            try:
                import requests  # may be a shim in tests
                try:
                    response = requests.get(url, timeout=5)
                    if getattr(response, 'status_code', None) != 200:
                        return _fail(f"Ollama server not available at {base_url}")
                except NotImplementedError:
                    # Fallback to urllib if shim is active
                    from urllib import request as _urlreq
                    try:
                        with _urlreq.urlopen(url, timeout=5) as resp:
                            if getattr(resp, 'status', None) != 200:
                                return _fail(f"Ollama server not available at {base_url}")
                    except Exception as e:
                        return _fail(f"Failed to connect to Ollama server: {e}")
                except Exception as e:
                    # requests present but failed
                    return _fail(f"Failed to connect to Ollama server: {e}")
            except Exception:
                # No requests; use urllib
                from urllib import request as _urlreq
                try:
                    with _urlreq.urlopen(url, timeout=5) as resp:
                        if getattr(resp, 'status', None) != 200:
                            return _fail(f"Ollama server not available at {base_url}")
                except Exception as e:
                    return _fail(f"Failed to connect to Ollama server: {e}")
            
            # Create Ollama language model using unified DSPy LM interface
            lm = dspy.LM(
                model=f"ollama/{model}",
                max_tokens=self.config.get("max_tokens", 1000),
                temperature=self.config.get("temperature", 0.0),
                **({'api_base': base_url} if base_url != "http://localhost:11434" else {})
            )
            
            # Configure DSPy
            dspy.configure(lm=lm)
            
            outputs = {
                "dspy_lm": lm,
                "provider": "ollama",
                "model": model,
                "base_url": base_url
            }
            
            # Record model info for downstream reporting
            try:
                context.set_artifact("model_info", {
                    "provider": "ollama",
                    "model": model,
                    "base_url": base_url,
                    "max_tokens": self.config.get("max_tokens", 1000),
                    "temperature": self.config.get("temperature", 0.0)
                })
            except Exception:
                pass
            
            return NodeResult(
                status=NodeStatus.SUCCESS,
                outputs=outputs
            )
            
        except Exception as e:
            return NodeResult(
                status=NodeStatus.FAILED,
                error=f"Failed to setup Ollama: {str(e)}"
            )
