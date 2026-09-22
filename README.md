# Machine Vision Projects

Selected implementations developed during COMP0137 (2025–26), extracted from the author's working notebooks. The author confirmed ownership of both account identities present in the original working copies.

- `tracking.py`: patch likelihood and particle-based corner tracking. The original routine expects corner-named `.mat` templates and a `Pattern01/` frame directory; supply your own compatible images and templates. Paths and display behaviour follow the original implementation.
- `homography.py`: homogeneous least squares, direct linear transform, planar pose estimation and 3D point projection. Point arrays use coordinates by columns.
- `pose_regression.py`: feature dataset, train/validation split, bidirectional LSTM regressor, and regression training/evaluation loops. Inputs are your own `.npy` arrays of shape `(time, 51)`, with an integer target at the start of each filename followed by `_`. No pose extractor, external model download or trained checkpoint is bundled.

## Use

Use Python 3.10+ and install `requirements.txt`; install `requirements-neural.txt` only for pose regression. Import the required functions into your own script. Imports do not start training or upload files.

```python
import numpy as np
from homography import calcBestHomography
points = np.array([[0., 1., 1., 0.], [0., 0., 1., 1.]])
H = calcBestHomography(points, points + np.array([[2.], [3.]]))
print(H)
```

This curated export omits assessment questions, teaching notebooks, reports, identifiers, outputs, footage, datasets, model weights, cloud storage access, login and publishing code. Algorithms retain the limitations of the original work; no accuracy or complete-reproduction claims are made. No blanket licence is added to course-derived interfaces or third-party dependencies.
