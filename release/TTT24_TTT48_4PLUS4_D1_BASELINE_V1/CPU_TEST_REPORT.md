# CPU validation record

Run on 2026-09-24 from the release-review working tree:

```text
429 passed, 3 xfailed in 53.38s
```

The three xfails are deliberate evidence of unresolved production-release
contracts, not passing checks:

1. runtime task identity is pre-frozen rather than derived from the mounted
   challenge;
2. runtime multi-test structure is pre-assumed rather than derived;
3. checkpoint identity does not bind challenge content and test-index shape.

Focused D1 and finalizer tests passed.  This CPU result does not establish
live 4+4 evidence parity, TTT model-state parity, hidden coverage, or a
competition submission result.
