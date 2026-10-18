# Explicit Controller routes

This is a small routing correction, not another scientific approval layer.
The same Governor remains the only scheduler. Existing immutable events and
historical single-Controller records are preserved.

`agent_routes` binds each role (`director`, `oracle`, `selector`) to its exact
Herdr `agent_name`, `pane_id`, optional `terminal_id`, and explicit boolean
`held`. Configure using:

```text
py -3 scripts/arc2_governor.py --state <state.json> --configure-agent-routes <routes.json>
```

The input file contains `{"agent_routes": {...}}`. Configuration never prompts
an agent or changes the workflow disposition. A future pane/session replacement
requires an explicit binding update, not an implicit guess.

New Controller event payloads contain `controller_role: "oracle"` or
`controller_role: "selector"`. Review payloads carry that field so Director's
reply returns to the originating Controller. Remote event producers likewise
retain this role; an unbound legacy Controller event stays queued with one local
routing alert, never a guessed delivery. A task cannot use stage text or known
failure IDs as routing rules.

- Verify name + pane + recorded terminal identity before delivery.
- No historical-pane, focused-pane, or remaining-agent fallback.
- A held recipient receives no prompt, including remote-terminal/resume events.
- A held queue entry does not block another eligible role.
- Busy ordinary recipients wait. Blocked/unknown recipients are not prompted.
- Persist delivery intent before sending; ambiguous sends are never replayed.
- Repeated explicit resume commands cannot resend the same event.
- Operator hold changes are rechecked before reservation. Held agents are not
  unheld by review decisions or import of a new event.
- Watchdog process-host repair is sent only to the exact unheld Director when
  named routes are enabled, never to the UI-focused pane.

The routing tests use isolated state and fake Herdr agents, not live scientific
tasks. Reconfiguring/restarting the service does not grant GPU or research scope.
