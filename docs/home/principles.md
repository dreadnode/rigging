# Principles

LLMs are extremely capable machine learning systems, but they operate purely in textual spaces as a byproduct of
their training data. We have access to the compression of a huge repository of human knowledge, but are limited to quering
that information via natural language. Our first inclination is to let these language interfaces drive 
our design decisions. We build chat bots and text search, and when it comes time to align them with closely
with the rest of our fixed software stack, we quickly get frustrated by their inconsistencies and limited
control over their products.

In software we operate on the principle of known interfaces as the basis for composability. In the functional paradigm, we want our
software functions to operate like mathmatical ones, where the same input always produces the same output with no side effects.
Funny enough LLMs (like all models) also operate in that way (minus things like floating point errors), but we intentionally
inject randomness to our sampling process to give them the freedom to explore and produce novel outputs. Therefore we shouldn't
aim for "purity" in the strict sense, but we should aim for consistency in their interface.

Once you start to think of a "prompt", "completion", or "chat interaction" as being the temporary textual interface by which we pass in
structured inputs and produce structured outputs, we can begin to link them with traditional software. Many libraries get close to this
idea, but they rarely hold the opinion that programing types and structures, and not text, are the best way to make LLM-based
systems composible.

Reframing these language models as tools which use tokens of text in context windows to navigate latest space and produce
probabilities of output tokens, but do not need to have the data they consume or produce be holistically constrained to
textual spaces in our use of them is a core opinion of Rigging.