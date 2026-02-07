"""Codebase Intelligence - Constraint-aware code generation."""

from codebase_intelligence.graph import SemanticGraph
from codebase_intelligence.nodes import (
    ClassNode,
    FunctionNode,
    ModuleNode,
    NodeType,
    SemanticNode,
)
from codebase_intelligence.edges import EdgeType, SemanticEdge
from codebase_intelligence.parser import ASTParser
from codebase_intelligence.patterns import Pattern, PatternExtractor
from codebase_intelligence.constraints import (
    Constraint,
    ConstraintCompiler,
    ConstraintSet,
    ConstraintViolation,
    ErrorFormatConstraint,
    MustUseConstraint,
    MustNotCrossConstraint,
    NamingConstraint,
)
from codebase_intelligence.dsl import ConstraintDSL, DSLError
from codebase_intelligence.compiler import CompiledPrompt, PromptCompiler
from codebase_intelligence.llm import LLMProvider, LLMRequest, LLMResponse, StubLLMProvider
from codebase_intelligence.validator import CodeValidator, ValidationResult, LintResult, TestResult
from codebase_intelligence.pipeline import Pipeline, PipelineConfig, GenerationResult
from codebase_intelligence.feedback import (
    FeedbackEngine,
    ViolationCategory,
    EscalationLevel,
    ViolationDiagnosis,
    RefinementContext,
)
from codebase_intelligence.agent import (
    CodeAgent,
    AgentConfig,
    AgentResult,
    AgentPlan,
    Observation,
)

__all__ = [
    # Core graph
    "SemanticGraph",
    "SemanticNode",
    "ModuleNode",
    "ClassNode",
    "FunctionNode",
    "NodeType",
    "SemanticEdge",
    "EdgeType",
    # Parser
    "ASTParser",
    # Patterns
    "Pattern",
    "PatternExtractor",
    # Constraints
    "Constraint",
    "ConstraintCompiler",
    "ConstraintSet",
    "ConstraintViolation",
    "MustUseConstraint",
    "MustNotCrossConstraint",
    "NamingConstraint",
    "ErrorFormatConstraint",
    # DSL
    "ConstraintDSL",
    "DSLError",
    # Compiler
    "CompiledPrompt",
    "PromptCompiler",
    # LLM
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "StubLLMProvider",
    # Validator
    "CodeValidator",
    "ValidationResult",
    "LintResult",
    "TestResult",
    # Pipeline
    "Pipeline",
    "PipelineConfig",
    "GenerationResult",
    # Feedback
    "FeedbackEngine",
    "ViolationCategory",
    "EscalationLevel",
    "ViolationDiagnosis",
    "RefinementContext",
    # Agent
    "CodeAgent",
    "AgentConfig",
    "AgentResult",
    "AgentPlan",
    "Observation",
]

__version__ = "0.1.0"
