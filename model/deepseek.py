# Multi Query Attention (MQA):

# Each head still has its own Query projection (Q_h = W_q^h * X).
# But all heads share the same K and V projections:
# Different heads still attend to different things, but from the same context representation - the same K/V vectors

# The diversity now comes only from the Q-side.
# Why Doesn’t This Completely Kill Performance?
# Because:

# Q Projections Still Differ → Each head asks a different question.
# Same K/V ≠ Same Attention → Even with shared context, different Qs can produce very different attention maps.


# Multiple heads still help by:
# Producing multiple different Q projections = different attention scores.
# Having different softmax distributions → different attention outputs even over the same K/V.

# Think of it like this:
# You’re asking 8 experts (heads) the same set of facts (K/V), but each expert asks their own question (Q). You'll get 8 different perspectives on the same knowledge base.