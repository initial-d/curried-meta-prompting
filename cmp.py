"""
Curried Meta-Prompting (CMP) Framework Implementation
A functional programming approach to context-aware LLM reasoning

This implementation demonstrates the core concepts presented in the research paper:
"Curried Meta-Prompting: A Functional Programming Approach to Context-Aware 
Large Language Model Reasoning"
"""

from typing import Callable, Dict, Any, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import re
import time
import random


@dataclass
class KnowledgeBase:
    """Structured knowledge repository for different domains"""
    domains: Dict[str, Dict[str, Any]]
    
    def get_knowledge(self, domain: str, topic: str = None) -> str:
        """Retrieve knowledge for a specific domain and optional topic"""
        if domain not in self.domains:
            return f"No knowledge available for domain: {domain}"
        
        domain_knowledge = self.domains[domain]
        if topic and topic in domain_knowledge:
            return domain_knowledge[topic]
        
        # Return general domain knowledge
        return domain_knowledge.get('general', '')


@dataclass
class Context:
    """Context information for prompt adaptation"""
    level: str  # e.g., 'elementary', 'high_school', 'university', 'expert'
    constraints: List[str]
    environment: str
    additional_info: Dict[str, Any]


@dataclass
class TaskSpecification:
    """Task-specific reasoning patterns and templates"""
    task_type: str
    reasoning_pattern: str
    output_format: str
    examples: List[str]


class PromptTemplate:
    """Template for constructing prompts with placeholders"""
    
    def __init__(self, template: str):
        self.template = template
    
    def format(self, **kwargs) -> str:
        """Format template with provided arguments"""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required template argument: {e}")


class CurriedMetaPrompt:
    """
    Main CMP framework implementation
    Transforms monolithic prompts into composable, curried functions
    """
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, PromptTemplate]:
        """Initialize prompt templates for different task types"""
        return {
            'mathematical_reasoning': PromptTemplate(
                """You are an expert mathematician with deep knowledge in {domain}.
                
Context: {context}
Knowledge Base: {knowledge}

Task: {task_description}
Reasoning Pattern: {reasoning_pattern}

Problem: {input_data}

Please solve this step-by-step, showing your work clearly.
Output Format: {output_format}"""
            ),
            
            'logical_reasoning': PromptTemplate(
                """You are a logical reasoning expert.
                
Context: {context}
Background Knowledge: {knowledge}

Task: {task_description}
Approach: {reasoning_pattern}

Question: {input_data}

Please reason through this systematically.
Format: {output_format}"""
            ),
            
            'commonsense_reasoning': PromptTemplate(
                """You are applying commonsense reasoning with the following context.
                
Context: {context}
Relevant Knowledge: {knowledge}

Task: {task_description}
Method: {reasoning_pattern}

Scenario: {input_data}

Please provide a thoughtful response.
Format: {output_format}"""
            ),
            
            'general': PromptTemplate(
                """Context: {context}
Knowledge: {knowledge}
Task: {task_description}
Method: {reasoning_pattern}
Input: {input_data}
Format: {output_format}"""
            )
        }
    
    def curry_knowledge(self, domain: str, topic: str = None):
        """
        First level of currying: Apply domain knowledge
        Returns a function that takes context as next argument
        """
        knowledge = self.knowledge_base.get_knowledge(domain, topic)
        
        def knowledge_curried(context: Context):
            """Second level: Apply context"""
            context_str = self._format_context(context)
            
            def context_curried(task_spec: TaskSpecification):
                """Third level: Apply task specification"""
                
                def task_curried(input_data: str) -> str:
                    """Final level: Process input and generate prompt"""
                    return self._construct_final_prompt(
                        domain, knowledge, context_str, task_spec, input_data
                    )
                
                return task_curried
            
            return context_curried
        
        return knowledge_curried
    
    def _format_context(self, context: Context) -> str:
        """Format context information into readable string"""
        context_parts = [
            f"Level: {context.level}",
            f"Environment: {context.environment}"
        ]
        
        if context.constraints:
            context_parts.append(f"Constraints: {', '.join(context.constraints)}")
        
        if context.additional_info:
            for key, value in context.additional_info.items():
                context_parts.append(f"{key}: {value}")
        
        return "\n".join(context_parts)
    
    def _construct_final_prompt(
        self, 
        domain: str, 
        knowledge: str, 
        context: str, 
        task_spec: TaskSpecification, 
        input_data: str
    ) -> str:
        """Construct the final prompt from all components"""
        
        # Select appropriate template
        template_key = self._select_template(task_spec.task_type)
        template = self.templates[template_key]
        
        # Format the final prompt
        return template.format(
            domain=domain,
            knowledge=knowledge,
            context=context,
            task_description=task_spec.task_type,
            reasoning_pattern=task_spec.reasoning_pattern,
            output_format=task_spec.output_format,
            input_data=input_data
        )
    
    def _select_template(self, task_type: str) -> str:
        """Select appropriate template based on task type"""
        task_type_lower = task_type.lower()
        
        if 'math' in task_type_lower or 'equation' in task_type_lower:
            return 'mathematical_reasoning'
        elif 'logic' in task_type_lower or 'reasoning' in task_type_lower:
            return 'logical_reasoning'
        elif 'commonsense' in task_type_lower or 'common sense' in task_type_lower:
            return 'commonsense_reasoning'
        else:
            return 'general'


class CMPApplication:
    """
    High-level application interface for CMP
    Provides convenient methods for common use cases
    """
    
    def __init__(self):
        self.knowledge_base = self._create_sample_knowledge_base()
        self.cmp = CurriedMetaPrompt(self.knowledge_base)
    
    def _create_sample_knowledge_base(self) -> KnowledgeBase:
        """Create a sample knowledge base for demonstration"""
        domains = {
            'mathematics': {
                'general': """
                Mathematics is the study of numbers, shapes, patterns, and logical reasoning.
                Key principles include: proof by contradiction, mathematical induction, 
                algebraic manipulation, geometric reasoning, and calculus concepts.
                """,
                'algebra': """
                Algebra involves solving equations and working with variables.
                Key concepts: linear equations, quadratic equations, systems of equations,
                factoring, polynomials, and function analysis.
                """,
                'geometry': """
                Geometry deals with shapes, sizes, positions, and properties of space.
                Key concepts: angles, triangles, circles, area, volume, proofs,
                coordinate geometry, and transformations.
                """,
                'calculus': """
                Calculus studies continuous change through derivatives and integrals.
                Key concepts: limits, derivatives, integrals, chain rule,
                optimization, and differential equations.
                """
            },
            'science': {
                'general': """
                Science is the systematic study of the natural world through observation
                and experimentation. Key principles: scientific method, hypothesis testing,
                data analysis, peer review, and reproducibility.
                """,
                'physics': """
                Physics studies matter, energy, and their interactions.
                Key concepts: Newton's laws, thermodynamics, electromagnetism,
                quantum mechanics, and relativity theory.
                """,
                'chemistry': """
                Chemistry studies the composition, structure, and properties of matter.
                Key concepts: atomic theory, chemical bonding, stoichiometry,
                reaction mechanisms, and thermochemistry.
                """
            },
            'computer_science': {
                'general': """
                Computer science involves the study of algorithms, data structures,
                and computational systems. Key concepts: complexity analysis,
                software engineering, databases, and artificial intelligence.
                """,
                'algorithms': """
                Algorithms are step-by-step procedures for solving problems.
                Key concepts: time complexity, space complexity, sorting, searching,
                graph algorithms, and dynamic programming.
                """,
                'machine_learning': """
                Machine learning enables computers to learn from data.
                Key concepts: supervised learning, unsupervised learning,
                neural networks, feature engineering, and model evaluation.
                """
            }
        }
        
        return KnowledgeBase(domains)
    
    def create_math_solver(self, level: str = "high_school") -> Callable:
        """Create a specialized mathematics solver"""
        math_context = Context(
            level=level,
            constraints=["show_work", "step_by_step"],
            environment="educational",
            additional_info={"subject": "mathematics"}
        )
        
        task_spec = TaskSpecification(
            task_type="mathematical_reasoning",
            reasoning_pattern="systematic_problem_solving",
            output_format="step_by_step_solution",
            examples=["solve equations", "word problems", "proofs"]
        )
        
        return self.cmp.curry_knowledge("mathematics")(math_context)(task_spec)
    
    def create_science_tutor(self, domain: str, level: str = "university") -> Callable:
        """Create a specialized science tutor"""
        science_context = Context(
            level=level,
            constraints=["conceptual_understanding", "real_world_examples"],
            environment="tutoring",
            additional_info={"domain": domain, "interactive": True}
        )
        
        task_spec = TaskSpecification(
            task_type="educational_explanation",
            reasoning_pattern="scaffolded_learning",
            output_format="detailed_explanation_with_examples",
            examples=["concept explanation", "problem solving", "application"]
        )
        
        return self.cmp.curry_knowledge("science", domain)(science_context)(task_spec)
    
    def create_code_assistant(self, language: str = "python") -> Callable:
        """Create a specialized programming assistant"""
        cs_context = Context(
            level="intermediate",
            constraints=["best_practices", "commented_code", "error_handling"],
            environment="development",
            additional_info={"language": language, "style": "clean_code"}
        )
        
        task_spec = TaskSpecification(
            task_type="code_generation",
            reasoning_pattern="systematic_development",
            output_format="commented_code_with_explanation",
            examples=["algorithm implementation", "debugging", "optimization"]
        )
        
        return self.cmp.curry_knowledge("computer_science")(cs_context)(task_spec)


class AdvancedCMP(CurriedMetaPrompt):
    """
    Advanced CMP with higher-order functions, memoization, and lazy evaluation
    """
    
    def __init__(self, knowledge_base: KnowledgeBase):
        super().__init__(knowledge_base)
        self.cache = {}  # Memoization cache
        self.strategies = self._initialize_strategies()
    
    def _initialize_strategies(self) -> Dict[str, Callable]:
        """Initialize reasoning strategies"""
        return {
            'chain_of_thought': self._chain_of_thought_strategy,
            'step_by_step': self._step_by_step_strategy,
            'analogical': self._analogical_reasoning_strategy,
            'deductive': self._deductive_reasoning_strategy,
            'inductive': self._inductive_reasoning_strategy
        }
    
    def _chain_of_thought_strategy(self, base_prompt: str) -> str:
        """Apply chain-of-thought reasoning pattern"""
        return base_prompt + "\n\nLet's think through this step by step:"
    
    def _step_by_step_strategy(self, base_prompt: str) -> str:
        """Apply systematic step-by-step approach"""
        return base_prompt + "\n\nPlease solve this systematically:\n1. First, identify what we know\n2. Then, determine what we need to find\n3. Next, choose the appropriate method\n4. Finally, solve and verify"
    
    def _analogical_reasoning_strategy(self, base_prompt: str) -> str:
        """Apply analogical reasoning"""
        return base_prompt + "\n\nConsider similar problems and use analogical reasoning to solve this:"
    
    def _deductive_reasoning_strategy(self, base_prompt: str) -> str:
        """Apply deductive reasoning pattern"""
        return base_prompt + "\n\nUsing deductive reasoning, start with general principles and work toward the specific solution:"
    
    def _inductive_reasoning_strategy(self, base_prompt: str) -> str:
        """Apply inductive reasoning pattern"""
        return base_prompt + "\n\nUsing inductive reasoning, examine specific cases to identify patterns and general principles:"
    
    def curry_with_strategy(self, strategy_name: str):
        """Higher-order currying with reasoning strategy"""
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strategy = self.strategies[strategy_name]
        
        def strategy_curried(domain: str, topic: str = None):
            """Apply strategy to domain knowledge currying"""
            base_curry = self.curry_knowledge(domain, topic)
            
            def enhanced_curry(context: Context):
                def enhanced_context_curry(task_spec: TaskSpecification):
                    def enhanced_task_curry(input_data: str) -> str:
                        base_prompt = base_curry(context)(task_spec)(input_data)
                        return strategy(base_prompt)
                    return enhanced_task_curry
                return enhanced_context_curry
            return enhanced_curry
        
        return strategy_curried
    
    def memoized_curry(self, domain: str, topic: str = None):
        """Memoized version of curry_knowledge for performance"""
        cache_key = f"{domain}:{topic}"
        
        if cache_key not in self.cache:
            self.cache[cache_key] = self.curry_knowledge(domain, topic)
        
        return self.cache[cache_key]
    
    def lazy_curry(self, domain: str, topic: str = None):
        """Lazy evaluation version that delays computation"""
        def lazy_evaluator():
            return self.curry_knowledge(domain, topic)
        
        def lazy_curry_wrapper(context: Context):
            def lazy_context_wrapper(task_spec: TaskSpecification):
                def lazy_task_wrapper(input_data: str) -> str:
                    # Only evaluate when actually needed
                    curried_func = lazy_evaluator()
                    return curried_func(context)(task_spec)(input_data)
                return lazy_task_wrapper
            return lazy_context_wrapper
        
        return lazy_curry_wrapper
    
    def compose_strategies(self, *strategy_names):
        """Compose multiple reasoning strategies"""
        strategies = [self.strategies[name] for name in strategy_names if name in self.strategies]
        
        def composed_strategy(base_prompt: str) -> str:
            result = base_prompt
            for strategy in strategies:
                result = strategy(result)
            return result
        
        return composed_strategy
    
    def parallel_curry(self, domains: List[str], topic: str = None):
        """Create multiple curried functions in parallel"""
        return {domain: self.curry_knowledge(domain, topic) for domain in domains}
    
    def conditional_curry(self, condition_func: Callable, 
                         true_domain: str, false_domain: str, topic: str = None):
        """Conditional currying based on runtime conditions"""
        def conditional_wrapper(context: Context):
            def conditional_context_wrapper(task_spec: TaskSpecification):
                def conditional_task_wrapper(input_data: str) -> str:
                    if condition_func(input_data, context, task_spec):
                        curry_func = self.curry_knowledge(true_domain, topic)
                    else:
                        curry_func = self.curry_knowledge(false_domain, topic)
                    
                    return curry_func(context)(task_spec)(input_data)
                return conditional_task_wrapper
            return conditional_context_wrapper
        
        return conditional_wrapper


class CMPAnalyzer:
    """
    Analysis and evaluation tools for CMP performance
    """
    
    def __init__(self, cmp_instance: CurriedMetaPrompt):
        self.cmp = cmp_instance
        self.metrics = {}
    
    def evaluate_prompt_quality(self, curried_prompt: Callable, 
                              test_inputs: List[str], 
                              expected_patterns: List[str]) -> Dict[str, float]:
        """Evaluate the quality of a curried prompt"""
        results = []
        
        for input_data, expected_pattern in zip(test_inputs, expected_patterns):
            try:
                generated_prompt = curried_prompt(input_data)
                quality_score = self._calculate_quality_score(generated_prompt, expected_pattern)
                results.append(quality_score)
            except Exception as e:
                print(f"Error evaluating input '{input_data}': {e}")
                results.append(0.0)
        
        return {
            'average_quality': sum(results) / len(results) if results else 0.0,
            'consistency': 1.0 - (max(results) - min(results)) if results else 0.0,
            'success_rate': sum(1 for r in results if r > 0.5) / len(results) if results else 0.0
        }
    
    def _calculate_quality_score(self, generated_prompt: str, expected_pattern: str) -> float:
        """Calculate quality score based on pattern matching and completeness"""
        # Simple heuristic: check for key components
        components = ['knowledge', 'context', 'task', 'reasoning']
        score = 0.0
        
        for component in components:
            if component.lower() in generated_prompt.lower():
                score += 0.25
        
        # Check for expected pattern
        if expected_pattern.lower() in generated_prompt.lower():
            score += 0.5
        
        return min(score, 1.0)
    
    def benchmark_performance(self, test_suite: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive performance benchmarks"""
        results = {}
        
        for test_name, test_config in test_suite.items():
            print(f"Running benchmark: {test_name}")
            
            start_time = time.time()
            test_results = self._run_single_benchmark(test_config)
            end_time = time.time()
            
            results[test_name] = {
                'results': test_results,
                'execution_time': end_time - start_time,
                'timestamp': time.time()
            }
        
        return results
    
    def _run_single_benchmark(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single benchmark test"""
        # Simulate benchmark execution with random results
        return {
            'accuracy': round(random.uniform(0.7, 0.95), 2),
            'consistency': round(random.uniform(0.8, 0.98), 2),
            'transfer_rate': round(random.uniform(0.6, 0.85), 2)
        }


# Demonstration and Usage Examples
def main():
    """
    Demonstrate CMP capabilities with various examples
    """
    print("=== Curried Meta-Prompting (CMP) Demonstration ===\n")
    
    # Initialize CMP application
    app = CMPApplication()
    
    # Example 1: Mathematics Problem Solving
    print("1. Mathematics Problem Solving:")
    math_solver = app.create_math_solver("high_school")
    
    algebra_problems = [
        "Solve for x: 2x + 5 = 13",
        "Factor the quadratic: x² - 5x + 6",
        "Find the slope of the line passing through (2,3) and (4,7)"
    ]
    
    for problem in algebra_problems:
        print(f"\nProblem: {problem}")
        prompt = math_solver(problem)
        print("Generated Prompt:")
        print("-" * 50)
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
        print("-" * 50)
    
    # Example 2: Science Tutoring
    print("\n\n2. Science Tutoring (Physics):")
    physics_tutor = app.create_science_tutor("physics", "university")
    
    physics_questions = [
        "Explain Newton's second law and its applications",
        "What is the relationship between electric field and potential?",
        "How does quantum tunneling work?"
    ]
    
    for question in physics_questions:
        print(f"\nQuestion: {question}")
        prompt = physics_tutor(question)
        print("Generated Prompt:")
        print("-" * 50)
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
        print("-" * 50)
    
    # Example 3: Code Assistant
    print("\n\n3. Programming Assistant:")
    code_assistant = app.create_code_assistant("python")
    
    coding_tasks = [
        "Implement a binary search algorithm",
        "Create a function to validate email addresses",
        "Write a decorator for function caching"
    ]
    
    for task in coding_tasks:
        print(f"\nTask: {task}")
        prompt = code_assistant(task)
        print("Generated Prompt:")
        print("-" * 50)
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
        print("-" * 50)
    
    # Example 4: Advanced CMP Features
    print("\n\n4. Advanced CMP Features:")
    advanced_cmp = AdvancedCMP(app.knowledge_base)
    
    # Chain of thought reasoning
    print("\nChain of Thought Reasoning:")
    cot_solver = advanced_cmp.curry_with_strategy("chain_of_thought")(
        "mathematics", "algebra"
    )
    
    math_context = Context(
        level="high_school",
        constraints=["detailed_reasoning"],
        environment="educational",
        additional_info={"method": "chain_of_thought"}
    )
    
    task_spec = TaskSpecification(
        task_type="mathematical_reasoning",
        reasoning_pattern="chain_of_thought",
        output_format="detailed_solution",
        examples=["algebraic solutions"]
    )
    
    cot_prompt = cot_solver(math_context)(task_spec)("Solve: 3x² - 12x + 9 = 0")
    print("Chain of Thought Prompt:")
    print("-" * 50)
    print(cot_prompt[:300] + "..." if len(cot_prompt) > 300 else cot_prompt)
    print("-" * 50)
    
    # Example 5: Performance Analysis
    print("\n\n5. Performance Analysis:")
    analyzer = CMPAnalyzer(app.cmp)
    
    test_inputs = ["Solve x + 2 = 5", "Factor x² - 4", "Find derivative of x²"]
    expected_patterns = ["step", "factor", "derivative"]
    
    quality_metrics = analyzer.evaluate_prompt_quality(
        math_solver, test_inputs, expected_patterns
    )
    
    print("Quality Metrics:")
    for metric, value in quality_metrics.items():
        print(f"  {metric}: {value:.2f}")
    
    # Benchmark suite
    test_suite = {
        "algebra_solving": {"domain": "mathematics", "tasks": ["linear", "quadratic"]},
        "physics_concepts": {"domain": "science", "tasks": ["mechanics", "electricity"]},
        "code_generation": {"domain": "computer_science", "tasks": ["algorithms", "data_structures"]}
    }
    
    print("\nRunning Benchmarks:")
    benchmark_results = analyzer.benchmark_performance(test_suite)
    
    for test_name, results in benchmark_results.items():
        print(f"\n{test_name}:")
        for metric, value in results['results'].items():
            print(f"  {metric}: {value}")
        print(f"  execution_time: {results['execution_time']:.3f}s")
    
    # Example 6: Compositional Strategies
    print("\n\n6. Compositional Reasoning Strategies:")
    composed_strategy = advanced_cmp.compose_strategies(
        "chain_of_thought", "step_by_step", "deductive"
    )
    
    base_prompt = "Solve this complex problem involving multiple concepts."
    enhanced_prompt = composed_strategy(base_prompt)
    
    print("Composed Strategy Prompt:")
    print("-" * 50)
    print(enhanced_prompt)
    print("-" * 50)
    
    print("\n=== CMP Demonstration Complete ===")


def demonstrate_real_world_scenarios():
    """
    Demonstrate CMP in realistic educational and professional scenarios
    """
    print("\n=== Real-World CMP Scenarios ===\n")
    
    app = CMPApplication()
    advanced_cmp = AdvancedCMP(app.knowledge_base)
    
    # Scenario 1: Adaptive Learning System
    print("1. Adaptive Learning System:")
    
    # Different difficulty levels for the same concept
    levels = ["elementary", "high_school", "university"]
    
    for level in levels:
        print(f"\n{level.title()} Level:")
        adaptive_solver = app.create_math_solver(level)
        prompt = adaptive_solver("What is the area of a circle with radius 5?")
        print(f"Prompt length: {len(prompt)} characters")
        print(f"Contains 'step-by-step': {'step' in prompt.lower()}")
        print(f"Complexity level: {level}")
    
    # Scenario 2: Multi-Domain Problem Solving
    print("\n\n2. Multi-Domain Problem Solving:")
    
    def is_math_problem(input_data, context, task_spec):
        return any(word in input_data.lower() for word in ['equation', 'solve', 'calculate', 'x'])
    
    adaptive_solver = advanced_cmp.conditional_curry(
        is_math_problem, "mathematics", "science"
    )
    
    mixed_problems = [
        "Solve for x: 2x + 3 = 7",
        "Explain the process of photosynthesis",
        "Calculate the velocity of an object in free fall"
    ]
    
    context = Context(
        level="high_school",
        constraints=["clear_explanation"],
        environment="classroom",
        additional_info={"adaptive": True}
    )
    
    task_spec = TaskSpecification(
        task_type="problem_solving",
        reasoning_pattern="adaptive",
        output_format="educational",
        examples=["mixed problems"]
    )
    
    for problem in mixed_problems:
        print(f"\nProblem: {problem}")
        prompt = adaptive_solver(context)(task_spec)(problem)
        domain = "Mathematics" if is_math_problem(problem, context, task_spec) else "Science"
        print(f"Detected Domain: {domain}")
        print(f"Prompt includes domain knowledge: {domain.lower() in prompt.lower()}")
    
    # Scenario 3: Performance Optimization
    print("\n\n3. Performance Optimization with Memoization:")
    
    # Compare regular vs memoized performance
    import time
    
    print("Testing memoization performance...")
    
    # Regular currying
    start_time = time.time()
    for _ in range(100):
        solver = advanced_cmp.curry_knowledge("mathematics", "algebra")
    regular_time = time.time() - start_time
    
    # Memoized currying  
    start_time = time.time()
    for _ in range(100):
        solver = advanced_cmp.memoized_curry("mathematics", "algebra")
    memoized_time = time.time() - start_time
    
    print(f"Regular currying: {regular_time:.4f}s")
    print(f"Memoized currying: {memoized_time:.4f}s")
    print(f"Performance improvement: {((regular_time - memoized_time) / regular_time * 100):.1f}%")
    
    print("\n=== Real-World Scenarios Complete ===")


if __name__ == "__main__":
    # Run main demonstration
    main()
    
    # Run real-world scenarios
    demonstrate_real_world_scenarios()
    
    print("\n" + "="*60)
    print("CMP Framework Implementation Complete!")
    print("This demonstrates functional programming principles")
    print("applied to LLM prompt engineering and reasoning.")
    print("="*60)
