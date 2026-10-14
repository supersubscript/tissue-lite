# Collaboration Preferences

- Stay narrowly scoped to the user's request.
- When there is a clear, agreed implementation plan, execute it autonomously in coherent steps without requesting
  approval before each edit. Ask questions only when an ambiguity, blocker, dependency change, or required approval
  arises.
- Without a clear implementation plan, work interactively in small, explicit steps and confirm substantive changes.
- Do not create worktrees, delegate implementation, or start autonomous workflows unless the user explicitly requests it.
- If nearby code, APIs, or functions look outdated but are not necessary to complete the request, stop and confirm with the user before changing them.
- Do not make adjacent "cleanup", "consistency", or "backward compatibility" updates unless the user asked for them or approved them.

# Testing Preference

- Do not use test-driven development by default.
- Only use TDD when the user explicitly asks for it, or when they clearly request a test-first workflow.
