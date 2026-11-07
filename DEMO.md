# Run the written weights

The ten circuits are already compiled into **circuits.npz** (8,470 bytes). **circuits.json** lists their architectures, tensor names and shapes, and SHA-256 checksum. Run the commands below from this directory with Python 3.10+ and NumPy. The normal inference path loads numeric weights and executes the circuit; it performs no fitting.

```sh
python demo.py verify
python demo.py rolling_sum 7 -9 4        # [7.0, -2.0, 2.0]
python demo.py sort 9 -3 2 10 2          # [-3.0, 2.0, 2.0, 9.0, 10.0]
python demo.py fibonacci -10 -3 0 10     # [-55.0, 2.0, 0.0, 55.0]
python demo.py triangular -10 -3 0 10    # [45.0, 3.0, 0.0, 55.0]
python demo.py add 4.75 -8               # -3.25
python demo.py subtract 4.75 -8          # 12.75
python demo.py square -7.5 6.25          # [56.25, 39.0625]
python demo.py sin 8                     # approximately 0.98949
python demo.py tanh -7 4
python demo.py xpowx -4 0 4.5            # [0.00390625, 1.0, 875.7951799529037]
```

Inputs use the declared envelope [−10,10]. The signed Fibonacci and triangular clocks accept integer indices from −64 to 64, taking |n| forward or backward ticks. Sorting accepts arbitrary list lengths within available memory. It pads to the next power of two with the maximum allowed value, 10, and drops the padding after a fixed Batcher schedule routes the values through the saved 13-ReLU comparator. The sort guarantee identifies signed zeros and assumes binary64 round-to-nearest, gradual underflow, and the documented operation order.

The sine circuit folds phase and applies its trained 63-hinge spline. Positive self-power uses the balanced relative-error 63-hinge spline, which received formula access throughout [0,10] at build time. The nonpositive integer branch uses the separately reported full-specification lookup with the 0⁰=1 convention. Negative noninteger binary64 inputs have no conventional real value here. The tanh circuit uses NumPy's native tanh neuron with identity input and readout.

To see a failure of the compensated 48-stage square circuit, run `python demo.py square 0x1.0bb639c98c0b5p+2`. The recorded output is 17.497465374439237. The correctly rounded square is 17.497465374439233, one representable step lower. The `verify` command checks this discrepancy as well as positive cases for all ten circuits, including non-power-of-two sorting lengths. It runs independent oracle functions only for those checks.

For use from Python:

```python
from demo import Circuits

circuits = Circuits()
predictions = circuits.predict("sort", [8, -6.5, 5])
```

To reconstruct the archive and manifest, install SciPy alongside NumPy and run `python build_demo.py`, then `python demo.py verify`. The recorded build used NumPy 2.5.4 and SciPy 1.18.1. Two consecutive builds in that environment produced identical archive bytes. Fitting the sine readout uses the 1,024 original central labels, so a different least-squares implementation may change its last few bits.

The original shallow demo is frozen. The new high-precision programs can be run with python extension/scalar/infer.py sin 3.14 5 -7 or python extension/scalar/infer.py selfpower 0.125 4 9.5. The tests and one-ULP failure inputs are in extension/scalar/README.md. Matched optimizer comparisons are in extension/writer/table.csv and extension/optimizer/.
