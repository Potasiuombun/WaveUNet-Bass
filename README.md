This is an implementation of Wave U Net trained for increasing loudness while preserving the audio quality.
Input - Mono audio file [-1.0,1.0] Pa
Output - Mono audio file [-0.5,0.5] Pa

The output can then be reamplified using the equation  (1 / (0.5 + 1e-3)) * Output
