# Advanaced Lane Finding

In this project, the **radius of curvature** of the road ahead and the **vehicle offset** with respect to lane-centre position are infered, using a histogram based sliding window approach, applied to infer probability of lane marking pixels, followed by a second degree polynomial fit of the found left and right lane markings in a perspectively corrected top-down (bird's eye) view.

**Author:** Lyuboslav Petrov

Pipeline Summary

* Step 1: Lens Distortion Correction
* Step 2: Perspective Distortion
* Step 3: Lane Detection through Color and Edge based binary masking
* Step 4: Lane pixel inference
* Step 5: Radius of Curvature computation
* Step 6: Vehicle offset computation

Please find a detailed description of the algorithm within the standalone [WriteUp.html](./WriteUp.html) page, or execute step by step the pipeline with the notebook [WriteUp.ipynb](./WriteUp.ipynb).

Usage:

    # Execute on a single image
    python lane_detect_pipe.py -i test_images/test1.jpg

    # Execute on a video
    python lane_detect_pipe.py -v project_video.mp4

    # Save the output
    python lane_detect_pipe.py -v project_video.mp4 -o output/my-test-video.mp4
