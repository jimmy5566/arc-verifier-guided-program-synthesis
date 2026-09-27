# Same-adapter multi-pair surface audit

Train-pair teacher-forced surfaces are scored at five depths under one shared all-train adapter state per task. Target train pairs are omitted only from their scoring prompt, exactly as the historical LOO scorer does. No test target, generation, candidate, selector, or router is used.
