# BMAD Memory Templates Plan

Last updated: 2025-10-03

Purpose
-------
This document describes the plan and checklist to add built-in Active Memory templates for agents following the BMAD method. The plan is deliberately design-only: it defines placement, naming, content conventions, validation rules, and a checklist for implementers. Implementation (creating files) is out of scope for this document.

Scope and key requirements
--------------------------
- Location: `prebuilt-memory-tmpl/bmad/<agent_name>/`
- Per-agent content: multiple YAML template files named `<template_name>.yaml` (no per-agent README files)
- Template count per agent: minimum 5, maximum 15
- Template file structure contract (required fields):
  - `template.id` (string): naming pattern `bmad.<agent>.<template_name>.v1`
  - `template.name` (string): human-friendly title
  - `sections` (array): ordered list of section objects; each section must include:
    - `id` (string): short snake_case identifier
    - `title` (string): human-friendly section title
    - `description` (string): detailed guidance for what to store in the section and examples
  - `metadata` (optional object): tags, intended workflows, priority, etc.

Why multiple templates per agent
------------------------------
Different templates capture different conversation modes, workflows, or life-cycle stages. Examples: "initial intake", "problem deep-dive", "decision log", "progress checkpoint", "retrospective". Having multiple focused templates reduces ambiguity and improves consolidation quality.

Template filename and ID conventions
-----------------------------------
- File name: `<template_name>.yaml` (lowercase, hyphen-separated; e.g. `initial-intake.yaml`)
- Template ID: `bmad.<agent>.<template_name>.v1` (e.g. `bmad.analyst.initial-intake.v1`)
- Section IDs: short snake_case identifiers (e.g., `context`, `problem_statement`, `assumptions`)

Example template skeleton (canonical example for implementers)
----------------------------------------------------------------
template:
  id: "bmad.<agent>.<template_name>.v1"
  name: "<Agent> — <Template Name>"
sections:
  - id: "context"
    title: "Context"
    description: "Brief context and goals. Example: 'Audit of Q4 paid acquisition campaign for product X.'"
  - id: "problem_statement"
    title: "Problem Statement"
    description: "Concise statement of the core problem the agent should solve. Example: 'CTR is below benchmark.'"
  - id: "assumptions"
    title: "Assumptions"
    description: "Known constraints and assumptions to consider when making recommendations."
  - id: "observations"
    title: "Observations / Evidence"
    description: "Collected evidence, metrics, or findings used to inform decisions."
  - id: "recommendations"
    title: "Recommendations / Actions"
    description: "Actionable recommendations and next steps with owners and priorities."
metadata:
  usage: "intake"
  priority: "medium"

Validation rules
----------------
- Each `.yaml` must contain `template.id`, `template.name`, and at least 3 `sections`.
- Each section must include `id`, `title`, and `description`.
- Template filenames must be unique within the agent folder.

Goals & Acceptance Criteria (for implementers)
---------------------------------------------
1. One directory per chatmode agent under `prebuilt-memory-tmpl/bmad/`.
2. Each agent directory contains between 5 and 15 `.yaml` template files.
3. Each `.yaml` file validates against the template contract above.
4. No per-agent `README.md` files are present (project-level README allowed).

Checklist (design-level, to be completed before implementation)
-------------------------------------------------------------
- [x] Draft this plan and checklist in `docs/BMAD_MEMORY_TEMPLATES_PLAN.md` (this file)
- [ ] Confirm final list of agent names from `.github/chatmodes/` and map to agent folder names
- [ ] For each agent, define 5–15 template names and short descriptions (purpose/usage)
        [ ] analyst
        [ ] architect
        [ ] bmad-master
        [ ] bmad-orchestrator
        [ ] dev
        [ ] pm
        [ ] po
        [ ] qa
        [ ] sm
        [ ] ux-expert
- [ ] Produce canonical YAML example with detailed `description` fields (above)
- [ ] Decide whether `initial_sections` examples should live inside templates or only in companion docs
- [ ] Agree on versioning policy for templates (e.g., `.v1`, bump convention)
- [ ] Implementation handoff (create files) or schedule me to implement when you say "implement"

Risk & Questions
----------------
- Do you want `initial_sections` examples embedded in the YAML files? Embedding them increases the template file size but makes seeding easier.
- Should some templates be shared between agents? If yes, specify which ones and how to reference them.

Next steps
----------
Please confirm naming conventions and whether to include `initial_sections` in the YAML files. After your confirmation I can either (A) stop here and hand off this plan, or (B) implement the templates per the plan.

