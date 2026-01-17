# Hello AI - Learning Project

A beginner project exploring AI integrations: local models with Ollama and cloud AI with Claude API.

## Project Context

- **Skill level**: Python beginner (understands code structure)
- **Purpose**: Learning and experimentation with AI APIs
- **Stack**: Python, Ollama (local), Claude API (cloud)

## Guidelines for Claude

When helping with this project:

- Prefer simple, readable code over clever abstractions
- Add brief comments explaining *why* code works, not just *what* it does
- Use type hints to make code self-documenting
- Suggest small, incremental changes rather than large refactors
- Explain Python concepts when introducing new patterns

## Project Structure

```
hello-ai/
├── CLAUDE.md          # This file
├── requirements.txt   # Python dependencies (to be created)
├── ollama_demo.py     # Local AI examples (to be created)
└── claude_demo.py     # Claude API examples (to be created)
```

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- `ollama` - Python client for local Ollama models
- `anthropic` - Official Claude API client

## Environment Variables

```bash
ANTHROPIC_API_KEY=your_key_here  # Required for Claude API
```

## Common Commands

```bash
# Run Ollama demo (requires Ollama running locally)
python ollama_demo.py

# Run Claude demo (requires API key)
python claude_demo.py
```

## Learning Resources

- [Ollama Documentation](https://ollama.ai)
- [Claude API Documentation](https://docs.anthropic.com)
- [Python Official Tutorial](https://docs.python.org/3/tutorial/)
