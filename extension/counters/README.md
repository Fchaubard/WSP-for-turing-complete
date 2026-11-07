# Recurrent ReLU register programs

run.py compiles instructions into Xavier-initialized D,W,E matrices and grades every state against an independent integer interpreter. Five program texts are in PROGRAMS; state and trace counts are in results.json. Natural-number subtraction saturates at zero. The all-state certificate uses exact counters; binary64 cannot distinguish every integer beyond 2^53.

writer.py regenerates six fresh model prompts with permuted state positions, or scores the archived literal matrices with python writer.py score. All six pass the held-out one-step and rollout tests and match the certified weights up to equivalent gate order, using the last ReLU to clip a negative register. BP.md documents the dense trained and discrete inference controls.
