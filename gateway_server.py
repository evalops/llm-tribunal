#!/usr/bin/env python3
"""
LLM Tribunal Safety Gateway - Real-time LLM firewall and safety proxy

This server acts as a drop-in replacement for OpenAI-compatible endpoints,
performing real-time safety evaluation using the existing DAG pipeline
before forwarding requests to upstream LLM providers.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Any, Optional, List, Union, AsyncIterator
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx

from dag_engine import DAGExecutor, ExecutionContext
from config import get_config
from gateway_nodes import RealTimeProxyNode, SafetyPolicyNode
from utils.logging_setup import get_logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

# Pydantic models for OpenAI-compatible API
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None

class SafetyGateway:
    """Core safety gateway that wraps DAG execution for real-time processing."""
    
    def __init__(self, policy_config: Optional[str] = None):
        self.config = get_config()
        self.policy_config = policy_config or "gateway/policies/default_safety_policy.yaml"
        self.executor = None
        self.load_safety_policy()
        
        # Metrics tracking
        self.stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "allowed_requests": 0,
            "avg_latency_ms": 0,
            "policy_violations": {}
        }
    
    def load_safety_policy(self):
        """Load and configure the safety evaluation pipeline."""
        try:
            with open(self.policy_config, 'r') as f:
                policy_spec = f.read()
            
            self.executor = DAGExecutor(config=self.config)
            self.executor.load_from_spec(policy_spec)
            self.executor.validate(strict=False)  # Allow external inputs
            
            logger.info(f"Safety policy loaded: {self.policy_config}")
            logger.info(f"Pipeline nodes: {list(self.executor.nodes.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to load safety policy: {e}")
            raise
    
    async def evaluate_safety(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Evaluate request safety using the DAG pipeline."""
        start_time = time.time()
        
        try:
            # Create execution context with request data
            context = ExecutionContext()
            
            # Extract content from messages for evaluation
            content_parts = []
            for message in request.messages:
                if message.role == "user":
                    content_parts.append(message.content)
            
            full_content = "\n".join(content_parts)
            
            # Set artifacts for the pipeline
            context.set_artifact("request_content", full_content)
            context.set_artifact("model", request.model)
            context.set_artifact("temperature", request.temperature or 0.7)
            context.set_artifact("request_metadata", {
                "timestamp": datetime.now().isoformat(),
                "request_id": str(uuid.uuid4()),
                "message_count": len(request.messages)
            })
            
            # Execute safety evaluation pipeline
            result_context = self.executor.execute(context, use_cache=True)
            
            # Extract safety decision
            safety_result = {
                "allowed": True,
                "risk_score": 1,
                "action": "allow",
                "explanation": "No safety concerns detected",
                "latency_ms": int((time.time() - start_time) * 1000),
                "pipeline_results": {}
            }
            
            # Check results from safety nodes
            for node_id, node_result in result_context.node_results.items():
                if node_result.status.name == "SUCCESS":
                    outputs = node_result.outputs
                    safety_result["pipeline_results"][node_id] = outputs
                    
                    # Check for blocking conditions
                    if "action" in outputs and outputs["action"] == "block":
                        safety_result["allowed"] = False
                        safety_result["action"] = "block"
                        safety_result["explanation"] = outputs.get("reason", "Safety policy violation")
                    
                    if "risk_score" in outputs:
                        safety_result["risk_score"] = max(safety_result["risk_score"], outputs["risk_score"])
            
            return safety_result
            
        except Exception as e:
            logger.error(f"Safety evaluation failed: {e}")
            # Fail-safe: allow request but log the error
            return {
                "allowed": True,
                "risk_score": 1,
                "action": "allow",
                "explanation": f"Safety evaluation error: {str(e)}",
                "latency_ms": int((time.time() - start_time) * 1000),
                "error": str(e)
            }
    
    async def forward_request(self, request: ChatCompletionRequest) -> Union[Dict, AsyncIterator]:
        """Forward allowed requests to upstream LLM provider."""
        # For demo, we'll use a mock response or forward to Ollama
        upstream_url = self.config.api.upstream_url or "http://localhost:11434"
        
        if request.model.startswith("gpt") and ":" not in request.model:
            # OpenAI API
            return await self._forward_to_openai(request)
        else:
            # Ollama local model
            return await self._forward_to_ollama(request, upstream_url)
    
    async def _forward_to_ollama(self, request: ChatCompletionRequest, base_url: str):
        """Forward request to Ollama endpoint."""
        # Convert OpenAI format to Ollama format
        prompt_parts = []
        for message in request.messages:
            if message.role == "user":
                prompt_parts.append(message.content)
            elif message.role == "system":
                prompt_parts.insert(0, f"System: {message.content}")
        
        full_prompt = "\n".join(prompt_parts)
        
        ollama_request = {
            "model": request.model,
            "prompt": full_prompt,
            "stream": request.stream,
            "options": {
                "temperature": request.temperature or 0.7
            }
        }
        
        if request.max_tokens:
            ollama_request["options"]["num_predict"] = request.max_tokens
        
        url = f"{base_url.rstrip('/')}/api/generate"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=ollama_request, timeout=30.0)
            response.raise_for_status()
            
            if request.stream:
                # TODO: Implement streaming response
                pass
            
            data = response.json()
            
            # Convert Ollama response to OpenAI format
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": data.get("response", "")
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(full_prompt.split()),
                    "completion_tokens": len(data.get("response", "").split()),
                    "total_tokens": len(full_prompt.split()) + len(data.get("response", "").split())
                }
            }
    
    async def _forward_to_openai(self, request: ChatCompletionRequest):
        """Forward request to OpenAI API."""
        # This would integrate with OpenAI API
        # For now, return a mock response
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a mock response from the safety gateway. In production, this would be forwarded to the actual OpenAI API after safety evaluation."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 25,
                "total_tokens": 35
            }
        }
    
    def update_stats(self, safety_result: Dict[str, Any]):
        """Update gateway statistics."""
        self.stats["total_requests"] += 1
        
        if safety_result["allowed"]:
            self.stats["allowed_requests"] += 1
        else:
            self.stats["blocked_requests"] += 1
            violation_type = safety_result.get("explanation", "unknown")
            self.stats["policy_violations"][violation_type] = \
                self.stats["policy_violations"].get(violation_type, 0) + 1
        
        # Update rolling average latency
        latency = safety_result["latency_ms"]
        self.stats["avg_latency_ms"] = \
            (self.stats["avg_latency_ms"] * (self.stats["total_requests"] - 1) + latency) / \
            self.stats["total_requests"]

# Initialize the gateway
gateway = SafetyGateway()

# FastAPI app
app = FastAPI(
    title="LLM Tribunal Safety Gateway",
    description="Real-time LLM safety evaluation and policy enforcement",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint with safety evaluation."""
    try:
        # Evaluate safety
        safety_result = await gateway.evaluate_safety(request)
        gateway.update_stats(safety_result)
        
        logger.info(f"Safety evaluation: {safety_result['action']} (risk: {safety_result['risk_score']}, latency: {safety_result['latency_ms']}ms)")
        
        if not safety_result["allowed"]:
            # Return blocked response
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": safety_result["explanation"],
                        "type": "safety_violation",
                        "code": "content_blocked",
                        "details": {
                            "risk_score": safety_result["risk_score"],
                            "action": safety_result["action"],
                            "evaluation_time_ms": safety_result["latency_ms"]
                        }
                    }
                }
            )
        
        # Forward to upstream LLM
        response = await gateway.forward_request(request)
        
        # Add safety metadata to response
        if isinstance(response, dict) and "choices" in response:
            response["safety"] = {
                "evaluation_time_ms": safety_result["latency_ms"],
                "risk_score": safety_result["risk_score"],
                "policy_version": gateway.policy_config
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Request processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": "gpt-oss:20b",
                "object": "model",
                "owned_by": "tribunal-gateway",
                "permission": []
            },
            {
                "id": "gpt-oss:120b", 
                "object": "model",
                "owned_by": "tribunal-gateway",
                "permission": []
            }
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "policy_loaded": gateway.executor is not None,
        "stats": gateway.stats
    }

@app.get("/admin/stats")
async def get_stats():
    """Get gateway statistics."""
    return {
        "stats": gateway.stats,
        "policy_config": gateway.policy_config,
        "nodes": list(gateway.executor.nodes.keys()) if gateway.executor else []
    }

@app.post("/admin/reload-policy")
async def reload_policy():
    """Reload safety policy configuration."""
    try:
        gateway.load_safety_policy()
        return {"status": "success", "message": "Policy reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload policy: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Tribunal Safety Gateway")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--policy", help="Path to safety policy configuration")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    if args.policy:
        gateway = SafetyGateway(args.policy)
    
    logger.info(f"Starting LLM Tribunal Safety Gateway on {args.host}:{args.port}")
    
    uvicorn.run(
        "gateway_server:app" if not args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )
