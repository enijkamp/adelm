#!/bin/bash

cd matconvnet-1.0-beta16-gpu/matlab

matlab -nodisplay -nosplash -nojvm -r "vl_compilenn('enableGpu', true, 'cudaRoot', '/usr/local/cuda-8.0')";
