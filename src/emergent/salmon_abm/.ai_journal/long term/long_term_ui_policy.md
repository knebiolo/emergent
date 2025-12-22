```markdown
# UI Change Policy (long-term note)

Metadata:
- author: assistant
- timestamp: 2025-12-06
- tags: policy, ui, governance

Policy:
- UI design, layout, and interactive behavior are owned by the project author and must not be changed by automated edits or contributors without the author's explicit permission.
- Any code edits that affect UI appearance or behavior require a written approval comment in the ticket/PR describing the change, the reason, and the files affected.
- Automated fixes that do not change user-visible UI (e.g., internal refactors, bug fixes, logging reductions) are allowed, but the author should be notified via the AI journal.

Rationale:
- Prevent accidental regressions to user workflows and preserve reproducibility of experiments that rely on specific UI behavior.

Action:
- When a proposed change modifies UI files (files under `src/emergent/*` that import Qt or modify layouts), create a `session` journal entry and tag the change `ui-change-request` and wait for approval.

End of policy.
```
