
# MotionSwap

## Project Overview

This project investigates state-of-the-art deep learning techniques for both the generation and detection of deepfakes. It includes implementations of key models from literature, a comprehensive evaluation of their performance, and several proposed enhancements to improve realism and detection accuracy.

## Objectives

- Implement and evaluate leading deepfake generation models.
- Develop and test effective deepfake detection methods.
- Propose architectural innovations to enhance generation and detection.
- Analyze model performance using quantitative metrics and qualitative assessments.

## Implemented Techniques

### Generation Models

- **StyleGAN**: A style-based generative adversarial network enabling fine-grained control over generated face attributes.
- **Neural Voice Puppetry**: Audio-driven facial animation model capable of realistic lip synchronization.
- **Face2Face**: Real-time expression transfer using parametric face models.
- **SimSwap**: Identity-preserving face swapping framework.
- **First Order Motion Model**: Self-supervised motion-based animation using keypoint estimation.

### Detection Method

- **Prediction Error Inconsistency**: A ConvLSTM-based method analyzing temporal inconsistencies in video frames to detect deepfakes.

## Datasets

- **MEAD**: Multi-view audiovisual dataset used for training facial animation models (First Order Motion Model).
- **VGGFace2-HQ**: High-quality facial image dataset used for face swapping (SimSwap).

## Proposed Innovations

- **Enhanced Style Mixing**: Improves facial feature control in StyleGAN outputs.
- **Hybrid Audio-Visual Features**: Combines audio and visual cues for improved expression synchronization.
- **Region-Specific Error Analysis**: Focuses detection on manipulated facial regions to improve accuracy.
- **Cross-Model Identity Preservation**: Enhances identity retention during face swaps, especially under extreme poses.
- **Motion Transfer Refinement**: Reduces artifacts in motion animation through attention-based refinement.

## Experimental Results

- StyleGAN achieved an FID score of 8.43 with high-resolution image generation.
- SimSwap preserved identity with a 91% accuracy as measured by FaceNet similarity.
- Deepfake detection accuracy exceeded 93% on publicly available datasets, with consistent performance across models.
- Proposed enhancements yielded up to 23% improvements in specific evaluation metrics.

## Installation

Clone the repository and install the required dependencies:

```
git clone https://github.com/your-repo/deepfake-detection.git
cd deepfake-detection
pip install -r requirements.txt
```

**Note**: A CUDA-enabled GPU is strongly recommended for training and inference.

## Usage

To train or run models, navigate to the appropriate directory:

```
# Train StyleGAN
cd models/stylegan
python train.py --config configs/stylegan.yaml

# Run detection
cd detection/prediction_error
python detect.py --input path_to_video
```

## Repository Structure

```
deepfake-detection/
│
├── datasets/                  # MEAD, VGGFace2-HQ
├── models/
│   ├── stylegan/
│   ├── neural_voice_puppetry/
│   ├── face2face/
│   ├── simswap/
│   └── first_order_motion/
├── detection/
│   └── prediction_error/
├── utils/                     # Preprocessing, metrics
├── results/                   # Generated outputs
├── README.md
└── requirements.txt
```

## Future Work

- Optimize models for real-time deployment.
- Investigate cross-modal detection using both audio and visual signals.
- Explore adversarial co-training between generation and detection modules.
- Create a unified framework that integrates multiple generation techniques.
- Establish ethical usage guidelines for synthetic media.

## References

Key models and techniques referenced in this project include:

- T. Karras et al., "A Style-Based Generator Architecture for GANs", 2019.
- J. Thies et al., "Neural Voice Puppetry", 2020.
- J. Thies et al., "Face2Face", 2016.
- R. Chen et al., "SimSwap", 2020.
- A. Siarohin et al., "First Order Motion Model", 2019.
- I. Amerini et al., "Prediction Error Inconsistencies for Deepfake Detection", 2019.

A full list of references is available in the project report.