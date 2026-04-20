# Model Card — Music Discovery Agent

## Model Overview

The Music Discovery Agent is an agentic AI system that extends a content-based music recommender. It uses a large language model (GPT-4o-mini) combined with a weighted scoring engine to convert natural language music requests into personalized song recommendations. The system includes a plan → act → evaluate → refine loop with confidence scoring and self-correction.

---

## AI Collaboration During Development

### How I Used AI

I used AI (Claude) extensively during development as a design partner and coding assistant. My workflow involved describing the overall system goals and architecture, then iterating on individual modules with AI guidance. AI helped me:

- **Architecture design**: I described my Module 3 recommender and the agent extension idea. AI helped me break the system into clean, modular components (planner, recommender, explainer, evaluator, agent orchestrator) and suggested the refinement loop as a reliability feature.
- **Code generation**: AI generated initial drafts of each module, which I then reviewed, tested, and modified to fit my specific needs.
- **Test design**: AI suggested both unit tests and the batch test harness approach, including edge cases I hadn't considered (like contradictory user profiles).

### Helpful AI Suggestion

**The confidence scoring formula.** When I described wanting to "measure how good the recommendations are," AI suggested a weighted multi-factor confidence score combining score coverage (how close to theoretical max), genre availability (catalog coverage), score spread (differentiation between results), and absolute score thresholds. This was more nuanced than the simple "top score / max score" approach I originally had in mind, and it gave the evaluator much better signal for deciding when to trigger refinement.

### Flawed AI Suggestion

**Overly complex initial keyword parser.** AI's first version of the fallback keyword parser used a TF-IDF vectorizer from scikit-learn to match user queries against a pre-built vocabulary. While technically more sophisticated, it added a heavy dependency, was harder to debug, and actually performed worse on short inputs like "chill lofi" because TF-IDF penalizes common words. I scrapped it and replaced it with a simple dictionary-lookup approach that was more transparent, had zero dependencies, and gave me full control over exactly which keywords mapped to which genres and moods. The lesson was that simpler is often better — especially when explainability matters more than theoretical accuracy.

---

## Reflection on System Design

### Limitations

1. **Tiny catalog**: The system only has 20 songs. In a real product, the scoring algorithm would need optimization for catalogs with millions of tracks — the current O(n) scan of every song would not scale.

2. **Binary genre matching**: "Rock" and "metal" get zero partial credit despite being closely related genres. A production system would use genre embeddings or a similarity matrix to capture these relationships.

3. **Keyword fallback parser is naive**: Without an API key, the fallback parser uses simple keyword matching. It cannot understand nuanced requests like "something that sounds like Radiohead" or "music for a dinner party with friends."

4. **No learning**: The system cannot learn from user feedback. Every request is independent — it has no memory of past interactions or preferences.

5. **Dataset bias**: Pop has the most songs (5) while metal has only 1. Users requesting underrepresented genres get worse recommendations through no fault of their own.

### Potential Misuse and Prevention

- **Filter bubble reinforcement**: The system always recommends songs matching stated preferences, which could narrow a user's musical exposure over time. A future version could include a "discovery mode" that intentionally introduces variety.
- **Bias amplification**: If the catalog overrepresents certain genres or cultural styles, the system systematically disadvantages users with different tastes. Prevention: regular audits of catalog diversity and recommendation distribution across user segments.
- **Prompt injection**: A malicious user could craft inputs attempting to manipulate the LLM planner. The system mitigates this by validating all LLM outputs against known-valid genres and moods before use.

### Testing Surprises

The biggest surprise was how well the system handled contradictory requests. I expected the "intense acoustic EDM" edge case to crash or return empty results, but it actually still produced a ranked list — the confidence score just dropped to 0.73, correctly signaling that the results were not a great fit. The evaluator caught the contradiction and flagged it, which is exactly what a guardrail should do. Another surprise was that the keyword fallback parser handled 10 out of 10 test harness queries correctly without needing the LLM at all. This made me realize that for a small, well-defined domain, a rule-based approach can be just as effective as an LLM — the LLM adds value mainly for ambiguous or creative requests. Finally, I was surprised that the refinement loop did not trigger during the standard test harness run because all confidence scores stayed above the 0.5 threshold with the full 20-song catalog. It only activated when I manually removed genres from the catalog during edge-case testing, which taught me that the threshold might need to be tuned higher for stricter quality control.

---

## Ethical Considerations

- The system makes no claims about music quality — it ranks by feature similarity, not artistic merit.
- Recommendations are transparent: every score breakdown is visible to the user.
- The confidence score explicitly tells users when the system is uncertain, rather than presenting low-quality results as definitive.
- No user data is collected, stored, or shared.

---

## Future Improvements

- Add collaborative filtering using listening history
- Implement genre embeddings for partial-match scoring
- Expand the catalog with a real music database API (e.g., Spotify)
- Add a feedback loop where users can rate recommendations to improve future results
- Support multi-language requests
