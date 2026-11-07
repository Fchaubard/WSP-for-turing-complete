# Independent, post-freeze evaluation

This plan precedes the revision collection's global freeze. This evaluator
must find the final FROZEN.json marker and verify its contents before opening
any held-out arrays or issuing any scores. Do not send these data, metrics,
or comparisons to the model or use them to request another round.

1. Independently reconstruct all 174 initial weights from the 60 original
objective-matched raw first replies, using the original index and validity
checks. Verify the 193 hashes in one_shot_snapshot.json against the original
archived files, including each prompt/reply and the dated paper. Check hashing
of the final trajectories and the pre-dispatch feedback rule.
2. Recreate training and feedback inputs from that rule. Reconstruct the
incumbent, validation/training loss ordering, rejection streak and stop for
each attempted revision. Independently score each valid proposed weight
vector and compare the accepted final weights to the recorded selections.
Audit prompts/response provenance, round limits and sources of displayed
examples. A retained incumbent that already met the stopping condition is a
valid zero-revision trajectory.
3. Only after the global freeze, score every valid round-zero, proposed and
selected endpoint on its original outer observations. A rejected proposal's
test score is retrospective, never a substitute selection rule. For continuous
and vector tasks the original central test remains untouched. For the two
canonical sequences, dev is now the disjoint block 17–32. Also reproduce
the original early score on 9–32 as an explicitly overlapping diagnostic.
Keep the observed training loss, whole exact-output counts and raw scalar
RMSE in the same schemas as the primary study.
4. After the global freeze, draw one fresh random 128-bit seed using the host
operating system. Save that seed to reproduce an independent confirmation
without relying on the already inspected original test draws. For each of the
eight stochastic tasks use an independent numbered substream and 4,096 fresh
central examples. Scalar signed functions have 4,096 examples per outer sign;
positive self-power has 4,096 upper-shell inputs. For addition/subtraction,
sums and sorting, draw 4,096 outer vectors with independent signs and magnitudes
for each coordinate. Draw dimensions and ranges match the primary test.
Use the same confirmation observations for every size, seed and method of a
task. An integer trajectory with a fixed initial condition has no fresh sample
of that kind; the score on 33–64 remains a deterministic transfer test.
5. Report independent scores by original task, size, starting seed, prompt
condition and method. Compare iterative within condition to round zero,
and to Adam plus full BFGS, trust-region least squares, plain GD, GD plus
limited-memory BFGS, Adam plus limited-memory BFGS and direct limited-memory
BFGS. For train improvements report strict and 1e-10 additive tolerance. Keep
target-unit RMSE distinct across tasks; report matched direction counts
and per-task medians. No hypothesis of independent draws across widths or
seeds is required for the descriptive paired counts.

The evaluator writes only to focus/iterative_eval. The collector does not
read it. All post-freeze conclusions refer only to the fixed protocol in
focus/iterative/PROTOCOL.md with SHA-256
75c8a10bc407ddd2808e3a7101a82fcda21fe45b42c8d28ecd48cad31cc88638.
