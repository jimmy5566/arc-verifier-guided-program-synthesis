# Governor idempotency and dispatch reliability fix

## Scope

This isolated branch repairs deterministic Governor dispatch behavior.  It
does not modify the live Governor process, watchdog, durable production state,
RUN_003 remote checkout, remote PID, binding, or scientific configuration.

## Confirmed root causes

1. `controller()` replaced a structured `controller_dispatch` object with the
   string `ACTIVE_CONTROLLER_TURN_NO_REPROMPT` whenever the Controller pane was
   busy.  That erased the action identity needed for deduplication.
2. A `DISPATCHED` action became eligible for another prompt after a 300-second
   cooldown even when the Controller had not published a new semantic stage.
3. `prompt()` returned only a Boolean.  A timeout or nonzero Herdr result could
   occur after message acceptance but was treated like ordinary retryable
   failure.
4. Atomic replace protected readers from partial JSON but did not reject a
   stale Governor snapshot that could overwrite a newer Controller transition.
5. The historical E04-E launcher sent bracketed-paste bytes.  RunPod's verified
   interactive PTY path executes prompt-ready ordinary newline input instead.

## Implementation

- Workflow snapshots now receive a monotonic `state_revision`.  Writes holding
  an older revision raise `StateConflict`; the Governor reloads rather than
  replacing the newer transition.
- Controller dispatch persists `INTENT_RECORDED` before Herdr is invoked.  A
  stable action SHA, built from semantic stage/action/bound-review fields,
  deduplicates forever until the Controller publishes a new semantic action.
- Busy Controller observations retain the structured dispatch record.  A
  delivered action becomes `RUNNING`; an action with no prior record receives a
  separate `controller_activity` observation.
- Herdr delivery is classified as `DELIVERED`, `NOT_DELIVERED`, or
  `UNKNOWN_DELIVERY`.  Timeouts, OS errors, and unclassified nonzero exits are
  `UNKNOWN_DELIVERY` and are never automatically re-prompted.
- A new semantic action archives the prior dispatch with
  `SUPERSEDED_BY_NEW_SEMANTIC_ACTION`, then receives exactly one new intent.
- Remote receipt consumption records a durable receipt key and will not create
  a second continuation for the same receipt.
- `WAIT_REMOTE` remains a deterministic SSH/receipt poll and never invokes a
  Controller or Director prompt.
- The future-only E04-E launcher uses `Popen`, a prompt-ready delay and ordinary
  newline frames.  This changes no SHA-bound RUN_003 file on RunPod.

## Remaining risk

At-most-once delivery intentionally prefers an operator-visible
`UNKNOWN_DELIVERY` over an automatic retry.  If Herdr accepts a prompt but
cannot return an acknowledgement, the Controller may need to publish its next
state transition or an operator may explicitly resolve the durable intent.
This is safer than silently duplicating a scientific execution request.

## Scientific autonomy

Actual stage advance changes the semantic action SHA, so Controller work still
receives one new dispatch.  Owner pauses, Director reviews, remote-job guards,
sealed-data boundaries, frozen identities, and one-shot launch constraints are
unchanged.
