# Secondary experiment metrics

The 17,197 rows in secondary.csv give **raw root mean squared error (RMSE)** in the target's units, alongside exact matches. Each row names its unit of an exact match, number of examples, scalar output count, split, method family and initialization where applicable. A complete stream or sorted list must match in every coordinate. Signed integer recurrences score each index as one exact match. For RMSE every scalar output is included, including every position in a stream. The controls retain their own architecture, numerical oracle and data generator, so do not pool heterogeneous tasks or interpret the rows as independent replications.

- **Sparse WSP versus BP.** sparse.py evaluates all 12 fixed writer replies and a matched gradient control in sparse_paired.csv. Both start at exactly the original Xavier arrays, and BP is given the support of that specific WSP reply. Its free parameters are only the selected coordinates. The observed targets are the first H impulse coefficients for each short condition (H=2,3,4) or d+1 for each all-horizon order-one condition (width d=4,5,6). The scalar loss is half the mean squared impulse residual. Analytic reverse-mode gradients are directionally checked at initialization. BP uses 800 Adam updates and up to 1,000 BFGS iterations, using the adapter in sequence/run.py. The all-horizon WSP writer also received the recurrence rule. A held-out impulse and 512 independent stream draws under each original dev/shell protocol test transfer. Train/development-only data and no outer queries enter optimization. The same 512 stream arrays are paired between methods, and distinct test examples are not pooled across starts. WSP and fitted parameters are saved in sparse_states.npz.
- **Population tests and legacy controls.** frontier short and trajectory data keep their 64 initialization seeds, impulse lengths, and stream lengths. The one-state adaptive-gradient test keeps its 256 starts. Complete shape and state information for the signed, sorting and recurrent controls is present in the originating sequence/ JSON files. Each targeted robust-sort pair suite has a raw RMSE field. The larger matched recurrent study in extension/recurrent uses separate inputs and is reported separately. The gain-deletion and algebraic-compilation controls are labeled explicitly in the family column.
- **Signed self-power.** The fixed 21 cardinal-hat features take the values of the identity matrix on the 21 integer sites. The WSP head stores all 21 once-rounded rational function values. The minimum-norm readout solver and decoder-only BP see the seven central labels. BP starts with the original Xavier output vector and uses two unit gradient steps on half-summed square loss. The central features are orthonormal; unobserved readout coordinates retain their initial values. Its controls have identical structural features and zero fixed output bias. Raw predictions and labels are stored as hexadecimal floats in scalar/signed_selfpower.csv.

To regenerate in a fresh extraction with the repository's Python dependencies, run the following from the project root:

    python writer/probe.py score
    python writer/short_probe.py score
    python frontier/run.py
    python frontier/gradient.py
    python scalar/signed_selfpower.py
    python sequence/robust_sort.py
    python revision/other_metrics/sparse.py
    python revision/other_metrics/collect.py

The remaining legacy JSON files are read deterministically, with recorded integer-reference/teacher-forcing predictions where available. Existing fields and machine-output arithmetic are preserved. Raw output residuals are recomputed for the new sparse study, and decimal-record truncation is avoided. The previous publication's RMSE fields in legacy controls are reported as archived.
