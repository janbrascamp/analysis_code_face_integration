# mattarAdapt

This model is being developed to analyze data as part of a collaboration with Katy Thakkar and Jan Brascamp of Michigan State University (MSU).

The model simultaneously fits the shape of the HRF (using the FLOBS components), the gain parameters of a several covariates, and the two parameters that define a carry-over effect that is proportion to the difference between a current stimulus, and a drifting prior that integrates the history of recent stimuli.

The model accepts several key-values, which are used to create a nuanced metric value. These key-values include:

  stimLabels    - A cell array of char vectors, one for each row of the
                  stimulus matrix.
