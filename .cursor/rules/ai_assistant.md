# AI Assistant Guidelines for llm_hub

## Code Generation

- Follow PEP 8 and project code style
- Include proper type annotations
- Generate comprehensive docstrings
- Include appropriate error handling
- Add tests for generated code
- Respect the existing project structure

## Documentation Generation

- Use Google-style docstrings
- Include:
  - Brief description
  - Args section
  - Returns section 
  - Raises section
  - Examples section when helpful

## Comments

- Add comments for complex algorithms
- Explain "why" not "what" in comments
- Keep comments up-to-date with code changes
- Don't add redundant comments

## Design Patterns

- Prefer composition over inheritance
- Use factory pattern for object creation when appropriate
- Implement dependency injection for better testability
- Use context managers for resource management

## Best Practices

- Follow SOLID principles
- Keep functions focused on a single responsibility
- Limit function length (aim for < 50 lines)
- Use descriptive variable and function names
- Avoid global variables
- Prefer immutable data structures
- Use generators for large data sets

## LLM Package Specifics

- Use appropriate ML design patterns
- Separate model definition from training code
- Provide clear logging for model training
- Include proper validation steps
- Follow best practices for ML parameter management
- Create helper functions for common preprocessing tasks 