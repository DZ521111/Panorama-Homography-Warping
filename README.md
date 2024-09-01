# Beyond the Horizon: Enhanced Panoramas

**Course Project | Jul. 2023 - Jan. 2024**  
**Supervisor**: Dr. Debanga Raj Neog, Assistant Professor, Dept. of MFSDS&AI, IIT Guwahati

## Project Overview

This project focuses on developing a robust image stitching pipeline to create high-quality panoramic images. The pipeline leverages ORB (Oriented FAST and Rotated BRIEF) for feature detection and RANSAC (Random Sample Consensus) for outlier removal. The goal is to seamlessly stitch overlapping images into panoramic visuals by aligning overlapping regions and using homography transformation and warping techniques.

## Problem Statement

Image stitching involves merging multiple photographic images with overlapping fields of view to produce a single panoramic or high-resolution image. This process requires precise overlaps and consistent exposures to ensure seamless integration. The project aims to study and implement various algorithms for image stitching and to explore the challenges associated with creating panoramic images.

## Dataset

- **Source**: Images were collected from various locations within the IIT Guwahati campus using smartphone cameras.
- **Content**: Both horizontal and vertical panorama orientations were considered, with multiple images taken per scene to test algorithm robustness.
- **Preprocessing**: Images were cropped to standardized sizes and resolution to speed up processing time without compromising stitching quality.
- **Lighting Conditions**: Images were captured under different lighting conditions, including daylight and nighttime, to test the robustness of the stitching algorithm.

## Related Work

The project builds upon existing research in panoramic image stitching:
- **Feature-Based Techniques**: Utilized algorithms like SIFT (Scale-Invariant Feature Transform), SURF (Speeded Up Robust Features), and ORB for initial image alignment.
- **Blending Methods**: Techniques such as Discrete Wavelet Transform (DWT) for seamless blending and adaptive brightness/contrast adjustments were explored.

## Methodology

### 1. Feature Detection and Matching
- **Keypoint Detection**: ORB was used, leveraging the FAST algorithm for corner detection and applying a pyramid approach for scale invariance.
- **Descriptor Generation**: Rotated BRIEF was employed for binary descriptors, ensuring rotation invariance and efficient computation.
- **Feature Matching**: Brute Force and K-Nearest Neighbors (KNN) algorithms were utilized with Hamming distance as the metric for binary descriptors.

### 2. Homography and RANSAC
- **Homography Estimation**: Used to map corresponding points between images.
- **RANSAC**: Implemented to eliminate outliers and refine the homography matrix.

### 3. Warping and Image Blending
- **Warping**: Performed using affine and projective transformations to align images.
- **Blending**: Applied interpolation methods such as nearest-neighbor, bilinear, and bicubic to merge the images smoothly.

## Experiments and Results

- **Panorama of the Nanotech Department**: Successfully created a wide-angle view by stitching multiple images.
- **Panorama of Brahmaputra River**: Demonstrated the ability to handle natural landscapes and varying lighting conditions.
- **Vertical Panorama of Buildings**: Highlighted the algorithm's effectiveness in stitching vertical sequences.

## Challenges and Limitations

1. **Motion-Induced Inconsistencies**: Minor movements during image capture caused misalignments.
2. **Brightness and Contrast Disparities**: Differences in lighting led to visible seams.
3. **Optimal Threshold Estimation**: Difficulties in setting the thresholds for feature detection.
4. **Blank Spaces**: Occurred due to insufficient overlap or alignment issues.
5. **Image Blending**: Need for more advanced algorithms to achieve seamless blending.

## Future Work

- **Enhancing Motion Stabilization**: Developing better stabilization techniques.
- **Adaptive Brightness and Contrast Adjustments**: Automating these adjustments during the stitching process.
- **Advanced Blending Algorithms**: Researching algorithms that can more effectively handle color and texture differences.
- **Real-Time Stitching**: Implementing real-time stitching capabilities to enable immediate panorama generation, which could be beneficial for live applications such as surveillance and live event coverage.

## Conclusion

The project successfully demonstrated the use of feature-based techniques for creating high-quality panoramas. By addressing challenges related to motion, lighting, and blending, the project provides insights for future research and practical applications in fields such as Virtual Reality, Real Estate, Autonomous Vehicles, and more.

## References

1. [A survey on image and video stitching](https://www.sciencedirect.com/science/article/pii/S2096579619300063)
2. [A Review Over Panoramic Image Stitching Techniques](https://iopscience.iop.org/article/10.1088/1742-6596/1999/1/012115)
3. [Image Stitching System Based on ORB Feature Based Technique](https://pdfs.semanticscholar.org/cf0d/3838b87c0f14f5941f78252c444126ae36bc.pdf)

## Project Repository
Find the project on GitHub: [Beyond-Horizon](https://github.com/DZ521111/Beyond-Horizon)

