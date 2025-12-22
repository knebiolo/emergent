```markdown
# Long-term: Programming Guide (principles)

- author: assistant
- timestamp: 2025-12-07
- tags: long_term, programming-guide, coding-standards

## Principle 1 — Fix errors, don't bandaid

- Preference: Avoid `try/except` fallbacks, nested exceptions, and "best-effort" guards that silently hide root causes.
- Rationale: Silent fallbacks mask defects and create long-term technical debt. When an error occurs, surface it clearly and fix it at the source.
- Practice:
  - Use explicit input validation and clear, early checks instead of broad exception swallowing.
  - If an operation can legitimately fail in expected ways, catch only the specific exception(s) and handle them with a documented, tested remediation.
  - Prefer raising a descriptive exception (with context) over returning None or partial results.

## Principle 2 — Minimal, explicit compatibility shims

- Preference: Where backward compatibility is required, add small, well-documented shim modules that re-export canonical APIs rather than littering production code with conditionals.
- Rationale: Shims isolate compatibility code and allow the main code paths to be clean.

## Principle 3 — Visibility and telemetry

- Preference: When an error is allowed to propagate, add logging with actionable context (file, function, key parameter values). Do not drop stack traces.

## Principle 4 — Tests over try/except

- Preference: Add unit/integration tests for expected failure modes rather than wrapping code in broad `try/except` blocks.

## How I (assistant) will apply this

- When making code edits, I will prefer corrective changes that fix the root cause.
- I will only add narrow, documented `try/except` blocks when an external dependency is known to be flaky and no immediate fix exists; such blocks will include logging and a TODO comment linking to the journal entry that justifies the temporary guard.
- When the user requests compatibility, I will create small shim modules (re-exports) rather than widespread fallbacks.

## Next steps

- Treat this as a living document — we can add concrete examples and a checklist for PR reviewers next.

---

## Canonical Programming Policy (consolidated)

- **Readable, Robust Code:** Favor small, well-named modules and functions, explicit APIs, immutability where appropriate, and unit tests that cover behavior. Avoid global mutable core state.
- **Minimal Defensive Code:** Limit defensive handling to at most one deliberate try/except at public boundaries. Validate inputs early and fail fast with clear exceptions. Avoid broad or silent catches.
- **NASA-Style Constraints (Aspirational):** The strict NASA constraints (no recursion, statically provable loop bounds, no dynamic allocation after init, function size limits, assertion density, strict pointer rules, zero-warning builds) are desirable goals for critical components. They are *aspirational* for this project: we will aim to follow them where practical, but any deviation must be discussed and approved since some third-party libraries or platform constraints may require exceptions.

### Enforcement and Practical Steps

- Maintain a canonical `CONTRIBUTING.md` that summarizes the above and links to examples and checklists.
- Add CI checks incrementally (linters, max-function-length, typed annotations, static analyzers). Start with non-blocking warnings, then raise to blocking as the codebase is cleaned.
- Provide a small scaffold/demonstration module that illustrates the preferred patterns (bounded loops, startup-only allocations, assertion use, minimal defensive wrappers).
- For any code that crosses NASA-style strict lines, create a short design note in `.ai_journal/session/` explaining the rationale, trade-offs, and mitigation plan.

### Governance & Preferences

- **UI changes:** require explicit author approval for changes that affect user-visible behavior; tag the journal `ui-change-request` when proposing.
- **Repository tools:** prefer `GitHub Desktop` workflows by default; include equivalent CLI commands only on request.
- **Assistant behavior constraints:** the assistant must not create directories or new scripts without prior proposal and approval.

### Acceptance Criteria (summary)

- PRs include unit tests for behavior and explicit checks for failure modes.
- No `except: pass` in production code; all catches specify exception types and include logging/context.
- CI shows decreasing warning counts from linters and static analyzers; critical modules aim for zero-warning status.
- NASA-style constraints are documented as aspirational and require explicit approval for exceptions.

## Recent lessons (2025-12-09)

- Centralized, robust logging is critical for large, legacy modules. Replace silent `except Exception: pass` sites with a single standard logging helper (`_safe_log_exception`) that preserves the original traceback and includes contextual key/value pairs.
- Use micro-batch automated edits when updating a large file: small batches (4–8 changes) followed by smoke tests reduce risk and make bisecting regressions easier.
- Prefer failing-fast in tests/CI and only use fallbacks in live production if there is a documented mitigation plan and explicit tests for the fallback.

## Concrete rules for exception handling

1. Never use `except Exception: pass` in production code. If a catch is required, catch explicit exceptions (e.g., `except (OSError, ValueError):`) and log details.
2. For cross-cutting diagnostic reliability, create and use a single `_safe_log_exception(msg, exc, **ctx)` helper which attempts `logger.exception()` and falls back to `print()` if logging fails.
3. When changing many handlers across a file, automate the mechanical replacements and run headless smoke tests after each micro-batch.
4. Add small unit tests for the function(s) where exceptions were previously swallowed; write tests that reproduce the failure mode before fixing.
5. Document temporary guards with a `TODO: remove when <issue>` comment and add a link to the session report explaining the reason.

These rules will be enforced by the assistant when making automated or manual edits to legacy code.

```
