# Curried Meta-Prompting (CMP) Framework

A functional programming approach to context-aware LLM prompt generation using function currying.

## Overview

CMP transforms monolithic prompts into composable, curried functions that adapt to different contexts, user levels, and domains. It enables dynamic prompt generation for mathematics, science, and programming tasks.

## Key Features

- **Function Currying**: Break prompts into reusable, composable functions
- **Context-Aware**: Automatically adapt to user level (elementary, high school, university, expert)
- **Multi-Domain**: Built-in knowledge bases for math, science, and computer science
- **Reasoning Strategies**: Chain-of-thought, step-by-step, analogical, deductive reasoning
- **Performance Optimized**: Memoization and lazy evaluation

## Quick Start

```python
from cmp_framework import CMPApplication

# Initialize
app = CMPApplication()

# Create specialized solvers
math_solver = app.create_math_solver("high_school")
physics_tutor = app.create_science_tutor("physics")
code_assistant = app.create_code_assistant("python")

# Generate prompts
math_prompt = math_solver("Solve for x: 2x + 5 = 13")
physics_prompt = physics_tutor("Explain Newton's second law")
code_prompt = code_assistant("Implement binary search")
```

## Advanced Usage

```python
from cmp_framework import AdvancedCMP, Context, TaskSpecification

# Chain-of-thought reasoning
advanced_cmp = AdvancedCMP(knowledge_base)
cot_solver = advanced_cmp.curry_with_strategy("chain_of_thought")

# Custom context
context = Context(
    level="university",
    constraints=["show_work", "detailed_reasoning"],
    environment="educational"
)

# Generate enhanced prompts
enhanced_prompt = cot_solver("mathematics")(context)(task_spec)(problem)
```

## Core Architecture

```
Knowledge Base → Context → Task Specification → Final Prompt
     ↓             ↓            ↓                   ↓
   Domain       User Level   Reasoning         Generated
 Information   Environment    Pattern           Prompt
```

## Running Examples

```bash
python cmp_framework.py
```

This will demonstrate:
- Mathematics problem solving
- Science tutoring
- Code generation
- Advanced reasoning strategies
- Performance benchmarking

## Applications

- **Educational Systems**: Adaptive learning with level-appropriate explanations
- **AI Tutoring**: Domain-specific problem solving assistance
- **Code Generation**: Context-aware programming help
- **Research Tools**: Systematic reasoning for complex problems
