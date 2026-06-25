# agents/agent.py
# Agentic AI system - decision-making and tool routing

from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import json
from core.chains.base_chain import (
    BaseChain,
    InterviewQuestionChain,
    EvaluationChain,
    RAGChain,
    SummaryChain,
    ChainResult
)
from core.llm.llm_service import LLMService
from core.prompts.prompt_manager import PromptManager
from services.rag.retriever import Retriever, get_retriever
from utils.logger import Logger

logger = Logger(__name__)


class AgentAction(Enum):
    """Possible agent actions."""
    DIRECT_LLM = "direct_llm"
    USE_RAG = "use_rag"
    GENERATE_QUESTIONS = "generate_questions"
    EVALUATE_ANSWER = "evaluate_answer"
    GENERATE_SUMMARY = "generate_summary"
    RETRIEVE_INFO = "retrieve_info"


class ToolRegistry:
    """Registry of available tools."""
    
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
    
    def register(self, name: str, tool_func: Callable, description: str = ""):
        """Register a tool."""
        self.tools[name] = {
            "func": tool_func,
            "description": description
        }
        logger.debug(f"Tool registered: {name}")
    
    def get_tool(self, name: str) -> Optional[Callable]:
        """Get tool by name."""
        return self.tools.get(name, {}).get("func")
    
    def list_tools(self) -> List[str]:
        """List all available tools."""
        return list(self.tools.keys())


class Agent:
    """
    Agentic AI system that decides which chain/tool to use.
    
    Decides between:
    - Direct LLM call
    - RAG-augmented response
    - Interview question generation
    - Answer evaluation
    - Summary generation
    """
    
    def __init__(
        self,
        llm_service: LLMService = None,
        prompt_manager: PromptManager = None,
        retriever: Retriever = None,
        name: str = "InterviewAgent"
    ):
        self.name = name
        self.llm_service = llm_service or LLMService()
        self.prompt_manager = prompt_manager or PromptManager()
        self.retriever = retriever or get_retriever()
        self.tool_registry = ToolRegistry()
        
        # Initialize chains
        self.chains = {
            "interview": InterviewQuestionChain(llm_service, prompt_manager, retriever),
            "evaluation": EvaluationChain(llm_service, prompt_manager, retriever),
            "rag": RAGChain(llm_service, prompt_manager, retriever),
            "summary": SummaryChain(llm_service, prompt_manager, retriever),
            "base": BaseChain(llm_service, prompt_manager, retriever)
        }
        
        self._register_tools()
        logger.info(f"Agent initialized: {name}")
    
    def _register_tools(self):
        """Register available tools."""
        
        self.tool_registry.register(
            "retrieve_documents",
            self._tool_retrieve,
            "Retrieve relevant documents using semantic search"
        )
        
        self.tool_registry.register(
            "generate_questions",
            self._tool_generate_questions,
            "Generate interview questions"
        )
        
        self.tool_registry.register(
            "evaluate_answer",
            self._tool_evaluate_answer,
            "Evaluate interview answer"
        )
        
        self.tool_registry.register(
            "generate_summary",
            self._tool_generate_summary,
            "Generate performance summary"
        )
    
    async def decide_action(self, user_input: Dict[str, Any]) -> AgentAction:
        """
        Decide which action/chain to use based on user input.
        
        Args:
            user_input: User request dict with "intent" or "action"
        
        Returns:
            AgentAction enum
        """
        
        intent = user_input.get("intent", "").lower()
        action = user_input.get("action", "").lower()
        
        # Explicit action override
        if action:
            if "question" in action:
                return AgentAction.GENERATE_QUESTIONS
            elif "evaluate" in action or "score" in action:
                return AgentAction.EVALUATE_ANSWER
            elif "summary" in action:
                return AgentAction.GENERATE_SUMMARY
            elif "retrieve" in action or "search" in action:
                return AgentAction.USE_RAG
        
        # Intent-based decision
        if "generate" in intent and "question" in intent:
            return AgentAction.GENERATE_QUESTIONS
        elif "evaluate" in intent or "score" in intent:
            return AgentAction.EVALUATE_ANSWER
        elif "summary" in intent or "report" in intent:
            return AgentAction.GENERATE_SUMMARY
        elif "retrieve" in intent or "search" in intent:
            return AgentAction.USE_RAG
        elif "rag" in intent or "context" in intent or "background" in intent:
            return AgentAction.USE_RAG
        
        # Check vector store content availability
        if self.retriever.vector_store.size() > 0:
            return AgentAction.USE_RAG
        
        return AgentAction.DIRECT_LLM
    
    async def run(
        self,
        user_input: Dict[str, Any],
        temperature: float = 0.0,
        max_tokens: int = 4000
    ) -> Dict[str, Any]:
        """
        Execute agent - decide action and run appropriate chain.
        
        Args:
            user_input: User request
            temperature: LLM sampling temperature
            max_tokens: Max tokens
        
        Returns:
            Result dict with response and metadata
        """
        
        try:
            # Decide action
            action = await self.decide_action(user_input)
            logger.info(f"Agent decided action: {action.value}")
            
            # Execute appropriate chain
            result = None
            
            if action == AgentAction.GENERATE_QUESTIONS:
                result = await self.chains["interview"].invoke(
                    user_input,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=True
                )
            
            elif action == AgentAction.EVALUATE_ANSWER:
                result = await self.chains["evaluation"].invoke(
                    user_input,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=True
                )
            
            elif action == AgentAction.GENERATE_SUMMARY:
                result = await self.chains["summary"].invoke(
                    user_input,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=False
                )
            
            elif action == AgentAction.USE_RAG:
                result = await self.chains["rag"].invoke(
                    user_input,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            
            else:  # DIRECT_LLM
                result = await self.chains["base"].invoke(
                    user_input,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            
            return {
                "status": result.status,
                "response": result.output,
                "action": action.value,
                "metadata": result.metadata,
                "intermediate_steps": result.intermediate_steps
            }
        
        except Exception as e:
            logger.error(f"Agent execution failed: {str(e)}")
            
            return {
                "status": "error",
                "response": f"Agent error: {str(e)}",
                "action": "error",
                "metadata": {"agent": self.name, "error": str(e)},
                "intermediate_steps": []
            }
    
    async def _tool_retrieve(self, query: str, k: int = 5) -> str:
        """Tool: Retrieve documents."""
        results = await self.retriever.retrieve(query, k)
        return json.dumps([r.to_dict() for r in results])
    
    async def _tool_generate_questions(self, **kwargs) -> str:
        """Tool: Generate questions."""
        result = await self.chains["interview"].invoke(kwargs)
        return result.output
    
    async def _tool_evaluate_answer(self, **kwargs) -> str:
        """Tool: Evaluate answer."""
        result = await self.chains["evaluation"].invoke(kwargs)
        return result.output
    
    async def _tool_generate_summary(self, **kwargs) -> str:
        """Tool: Generate summary."""
        result = await self.chains["summary"].invoke(kwargs)
        return result.output
    
    def add_rag_documents(self, documents: List[Dict]) -> List[str]:
        """Add documents to RAG system."""
        return self.retriever.add_documents(documents)
    
    def get_agent_info(self) -> Dict:
        """Get agent capabilities and status."""
        return {
            "name": self.name,
            "chains": list(self.chains.keys()),
            "tools": self.tool_registry.list_tools(),
            "vector_store_size": self.retriever.vector_store.size(),
            "available_prompts": len(self.prompt_manager.list_prompts())
        }
