---
name: Feature request
about: Suggest a new capability or improvement
title: "[feature] "
labels: enhancement
body:
  - type: textarea
    id: problem
    attributes:
      label: Problem
      description: What are you trying to do that the library doesn't support today?
    validations:
      required: true
  - type: textarea
    id: solution
    attributes:
      label: Proposed solution
      description: How should it work? API sketch welcome (code block).
    validations:
      required: true
  - type: textarea
    id: alternatives
    attributes:
      label: Alternatives considered
      description: Workarounds, other libraries, or designs you rejected.
  - type: dropdown
    id: area
    attributes:
      label: Area
      options:
        - cache
        - routing / strategies
        - providers
        - cost tracking / budgets
        - observability
        - middleware
        - docs
        - other
    validations:
      required: true
