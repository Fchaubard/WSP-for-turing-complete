# Bounded, central-feedback weight revision

The prespecified rules in PROTOCOL.md determine when a revision is requested, how its candidate weights are selected, and when each size stops. protocol.sha256 gives the digest pinned by run.py before a request can be sent. This folder contains only the training and newly generated development inputs or diagnostics. Its driver never generates test inputs or reads test results.

Run from the repository root in the published Python environment. The following command replays archived prompts, scores every received or failed proposal from the recorded training and development data, and verifies the final selected trajectory files byte-for-byte against the frozen hashes. It makes no new model calls.

    python focus/iterative/run.py --replay --workers 4

The live collection used twelve workers and a 300-second per-call timeout. It can resume a genuinely interrupted collection from the preserved reply/metadata or terminal-error files. Re-running without replay mode can initiate new model requests if required archival files have been removed.

    WSP_REQUEST_TIMEOUT=300 python focus/iterative/run.py --workers 12

Each original request has a separate ID, one task, one seed, one information condition and multiple sizes. The source field identifies its original prompt and first response in focus/objective/trials or focus/arithmetic/trials. The original prompt contains the full Xavier initialization and labeled observations. The initial field in a trajectory is the actual vector returned by Astra in that original prompt, before any feedback. A new call reveals the incumbent vector, central diagnostics and a bounded number of labeled examples; the other information condition's reply and all outer outcomes are withheld.

- trials/ holds the revision prompt and displayed feedback, followed by an unmodified final response and metadata or a terminal request error. For example, ID-r1 means the first revision request for original ID. No external or training tools are permitted in a reply.
- trajectories/ holds one document per original request. For each size, history gives each candidate, its scores before selection, whether it was accepted and the incumbent's training and development scores after the decision. A JSON null proposal represents an invalid or absent array. A finite parameter vector with nonfinite predictions is archived as a candidate without finite scores. The stop field distinguishes exactness, the non-improvement rule, and the maximum number of rounds. A case can finish with zero new requests.
- FROZEN.json is created only after all 60 requests' trajectories have a final decision for every size. It pins their SHA-256 hashes. Original central, fresh central and outer evaluation belongs to the independent evaluator, not this collection driver.

After freeze, run the Python driver focus/iterative/summarize.py to write selection.csv and selection_summary.json from central data. archive_sha256.json covers the revised request artifacts and the original prompts, answers, and metadata referenced by each case.

## Frozen collection

All 161 new requests returned a final model message, with no tools recorded. Of 332 attempted size revisions, 217 met the selection criterion. One width-sixteen square vector contained 52 coefficients where 50 were required and was rejected. The 60 initially exact size settings needed no feedback. Of the remaining 114, 33 stopped after two unsuccessful proposals and 81 reached the third round. Comparing the selected endpoint to the original response, development loss strictly decreased in 52 of 87 settings for each information condition. Training loss strictly decreased, tied or increased in 6, 35 and 46 examples-only settings and in 5, 35 and 47 formula-given settings. These developments reflect the explicit validation-first selection rule.
