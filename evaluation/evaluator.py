# evaluation/evaluator.py
# Evaluation system for testing prompts and chains

import json
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
import time
from core.chains.base_chain import BaseChain, ChainResult
from utils.logger import Logger

logger = Logger(__name__)


@dataclass
class EvalMetric:
    """Single evaluation metric."""
    name: str
    value: float
    target: float = 0.8


@dataclass
class EvalResult:
    """Evaluation result."""
    chain_name: str
    test_case: str
    input_data: Dict
    output: str
    metrics: List[EvalMetric]
    execution_time_ms: float
    success: bool


class PromptEvaluator:
    """Evaluate prompt quality and output."""
    
    @staticmethod
    def evaluate_json_validity(output: str) -> EvalMetric:
        """Check if output is valid JSON."""
        try:
            json.loads(output)
            return EvalMetric("json_validity", 1.0, 1.0)
        except:
            return EvalMetric("json_validity", 0.0, 1.0)
    
    @staticmethod
    def evaluate_length(output: str, min_len: int = 10, max_len: int = 10000) -> EvalMetric:
        """Check output length is reasonable."""
        length = len(output)
        if min_len <= length <= max_len:
            return EvalMetric("length_validity", 1.0, 1.0)
        else:
            return EvalMetric("length_validity", 0.0, 1.0)
    
    @staticmethod
    def evaluate_contains_keywords(output: str, keywords: List[str]) -> EvalMetric:
        """Check if output contains expected keywords."""
        output_lower = output.lower()
        found = sum(1 for kw in keywords if kw.lower() in output_lower)
        
        score = found / len(keywords) if keywords else 0
        return EvalMetric("keyword_presence", score, 0.8)
    
    @staticmethod
    def evaluate_formatting(output: str, format_type: str = "json") -> EvalMetric:
        """Check output formatting."""
        
        if format_type == "json":
            try:
                json.loads(output.strip())
                return EvalMetric("format_correctness", 1.0, 1.0)
            except:
                return EvalMetric("format_correctness", 0.0, 1.0)
        else:
            return EvalMetric("format_correctness", 0.5, 1.0)


class ChainEvaluator:
    """Evaluate chain performance."""
    
    def __init__(self, chain: BaseChain):
        self.chain = chain
        self.results: List[EvalResult] = []
        logger.info(f"ChainEvaluator initialized for: {chain.name}")
    
    async def evaluate_testcase(
        self,
        test_name: str,
        input_data: Dict[str, Any],
        expected_output_validator: Callable = None,
        **chain_kwargs
    ) -> EvalResult:
        """
        Evaluate single test case.
        
        Args:
            test_name: Test case name
            input_data: Input to chain
            expected_output_validator: Function to validate output
            **chain_kwargs: Additional kwargs for chain.invoke
        
        Returns:
            EvalResult with metrics
        """
        
        # Execute chain
        start_time = time.time()
        try:
            result = await self.chain.invoke(input_data, **chain_kwargs)
            success = result.status == "success"
            output = result.output
        except Exception as e:
            logger.error(f"Chain evaluation failed: {str(e)}")
            success = False
            output = f"Error: {str(e)}"
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Evaluate output
        metrics = []
        
        # Basic metrics
        metrics.append(PromptEvaluator.evaluate_length(output))
        
        # Try JSON validation
        try:
            data = json.loads(output)
            metrics.append(PromptEvaluator.evaluate_json_validity(output))
        except:
            pass
        
        # Custom validator
        if expected_output_validator:
            try:
                is_valid = expected_output_validator(output)
                metrics.append(EvalMetric(
                    "custom_validation",
                    1.0 if is_valid else 0.0,
                    1.0
                ))
            except Exception as e:
                logger.error(f"Custom validator failed: {str(e)}")
        
        # Create result
        eval_result = EvalResult(
            chain_name=self.chain.name,
            test_case=test_name,
            input_data=input_data,
            output=output,
            metrics=metrics,
            execution_time_ms=execution_time_ms,
            success=success
        )
        
        self.results.append(eval_result)
        
        logger.info(
            f"Test case: {test_name} | Success: {success} | Time: {execution_time_ms:.0f}ms"
        )
        
        return eval_result
    
    async def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run multiple test cases and return summary.
        
        Args:
            test_cases: List of {"name": ..., "input": ..., "validator": ...}
        
        Returns:
            Summary dict
        """
        
        for test_case in test_cases:
            await self.evaluate_testcase(
                test_name=test_case.get("name", "unknown"),
                input_data=test_case.get("input", {}),
                expected_output_validator=test_case.get("validator")
            )
        
        return self.get_summary()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get evaluation summary."""
        
        if not self.results:
            return {"status": "no_results"}
        
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        
        total_time_ms = sum(r.execution_time_ms for r in self.results)
        avg_time_ms = total_time_ms / total if total > 0 else 0
        
        # Aggregate metrics
        metrics_agg = {}
        for result in self.results:
            for metric in result.metrics:
                if metric.name not in metrics_agg:
                    metrics_agg[metric.name] = {"values": [], "target": metric.target}
                metrics_agg[metric.name]["values"].append(metric.value)
        
        # Calculate average scores
        metric_scores = {}
        for name, data in metrics_agg.items():
            avg_value = sum(data["values"]) / len(data["values"])
            metric_scores[name] = {
                "average": avg_value,
                "target": data["target"],
                "passed": avg_value >= data["target"]
            }
        
        return {
            "chain": self.chain.name,
            "total_tests": total,
            "successful": successful,
            "success_rate": successful / total if total > 0 else 0,
            "total_time_ms": total_time_ms,
            "avg_time_ms": avg_time_ms,
            "metric_scores": metric_scores
        }
    
    def print_report(self):
        """Print evaluation report."""
        
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print(f"EVALUATION REPORT - {summary.get('chain', 'Unknown Chain')}")
        print("="*80)
        
        print(f"\nTests Run: {summary.get('total_tests', 0)}")
        print(f"Successful: {summary.get('successful', 0)}")
        print(f"Success Rate: {summary.get('success_rate', 0)*100:.1f}%")
        print(f"Total Time: {summary.get('total_time_ms', 0):.0f}ms")
        print(f"Avg Time/Test: {summary.get('avg_time_ms', 0):.1f}ms")
        
        if summary.get("metric_scores"):
            print("\n" + "-"*80)
            print("METRICS:")
            print("-"*80)
            
            for metric_name, scores in summary.get("metric_scores", {}).items():
                status = "✓ PASS" if scores["passed"] else "✗ FAIL"
                print(f"  {metric_name:30} | {scores['average']:.2f} / {scores['target']:.2f} [{status}]")
        
        print("\n" + "="*80 + "\n")
