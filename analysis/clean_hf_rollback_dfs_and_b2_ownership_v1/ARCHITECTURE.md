# Clean-HF rollback DFS and Dynamic B2 ownership

Each logical view owns one stable `CacheOwner(DynamicCache)`. DFS recursion records a sequence-length checkpoint and restores it with `DynamicCache.crop()` before siblings and on return. The diagnostic snapshot decoder is separate and is not used by production.

B2 creates only temporary packed/split representations. It copies each split lane's tensors into the existing owner object, preserving the owner identity held by every suspended recursive frame. No zero-copy lane sharing and no allocator-clearing call are used.
