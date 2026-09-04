---
name: Bug report
about: Something is broken or behaves incorrectly
title: "[bug] "
labels: bug
body:
  - type: textarea
    id: description
    attributes:
      label: Description
      description: What happened, and what did you expect instead?
    validations:
      required: true
  - type: textarea
    id: reproduction
    attributes:
      label: Minimal reproduction
      description: A short self-contained code snippet (no API keys).
      placeholder: |
        from llm_cache_router import LLMRouter
        ...
    validations:
      required: true
  - type: textarea
    id: environment
    attributes:
      label: Environment
      description: llm-cache-router version, Python version, OS, cache backend, providers involved.
      placeholder: "llm-cache-router 0.2.4, Python 3.12, Linux, redis backend"
  - type: textarea
    id: logs
    attributes:
      label: Logs / traceback
      description: Full traceback or warning output, if any.
