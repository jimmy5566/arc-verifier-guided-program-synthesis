# Project-wide used-task blacklist audit

- Status: `PASS`
- Unique task IDs: `1120`
- Distinct tracked text blobs scanned: `3851`
- Unreadable text blobs: `0`

Every reachable textual Git blob was read in a single batch. A task reference is excluded from training even when its target-access status is not recoverable from the file path.
