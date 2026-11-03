# Methods critic review (development evidence, before sealed results)

Verdicts recommended: edit claim **contradicted_exploratory**; "latent but unexpressed" diagnosis **inconclusive**.

Main threats: shared-model-family judges calibrated only on easy AI-authored contrasts (conversation judge uncalibrated);
six families, one episode per cell, mostly one seed; edit directions built from teacher-forced reference replies (may
encode wording/length/topic rather than CEFR) applied to free generation; gate threshold not calibrated on realistic
states; candidate selection reused development families and no strength showed a dose-response.

Supported (exploratory): weak metadata-only control (gap 0.5, positive in 3/6 families); requested level linearly
decodable at the pre-decision position (decodes metadata, not a generation mechanism); small likelihood conditioning
(+0.116 nats/token, positive in 8/9 families). Not supported: the edit effect (agent 0.17 vs raw 0.5 vs random 1.0 vs
strong prompt 2.33 vs best-of-2 2.5).

Next experiments: (1) frozen edit on new unsealed families with multiple seeds, paired intervals and native-speaker
ratings; (2) directions from generated trajectories with length/topic-matched references, cross-fitted by family, with
label-permutation and sign controls; (3) factorial direction x gate x alpha with a calibrated gate and a preregistered
monotonic dose-response.
