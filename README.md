# Face Morphing Detection Benchmark



## Project Overview
This repository is dedicated to the development and benchmarking of various deep learning models for detecting face morphing attacks. Models include ResNet, EfficientNet, Vision Transformer, and their ensemble versions. They are tested on datasets such as TWIN, FERET, FRGC, and FRLL under different morphing attack types including OpenCV, FaceMorpher, and StyleGAN. This project also includes the weights of trained models for evaluation purposes.

<p align="center">
<img src="docs/process.jpg" width="900px"/>
<br>
Facial Recognition at the Frontline: Detecting Face Morphing Attacks at Passport Control 
</p>


## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.8+
- torchvision
- torchsampler
- sklearn
- tqdm
- matplotlib
- CUDA (for GPU support)

### Setup
Clone this repository to your local machine:

```bash
git clone https://github.com/your-github-repository/face-morphing-attack-detection-benchmark.git
cd face-morphing-attack-detection-benchmark
```

Install the required Python packages:
```
pip install -r requirements.txt
```

## Usage
### Training a Model
To train a model on your dataset, use the 'train_single_morph_attack_detection.py' script. Here's an example of how to run the training process for a ResNet model:
```
python train_single_morph_attack_detection.py --model_name model_vit_B16 \
    --train_dir ./dataset/frgc_train/train --val_dir ./dataset/test_sets/VSAPP \
    --epochs 80 --batch_size 8 --learning_rate 0.0001 \
    --scheduler_type StepLR --scheduler_step_size 15 --scheduler_gamma 0.1 \
    --results_name resnet_training_results
```

###  Evaluating a Model
To evaluate the trained resnet model using the 'evaluation_single_detector_resnet.py' script, run:

```
python evaluation_single_detector_resnet.py --model_name model_resnet_vanilla \
    --data_paths "FRLL_FaceMorpher:./dataset/FRLL/facemorpher,FRLL_OpenCV:./dataset/FRLL/opencv,FRLL_StyleGAN:./dataset/FRLL/stylegan" \
    --pretrained_weights ./pretrained_weights/resnet_vanilla.pth \
    --results_path ./results --results_name resnet_evaluation
```
This command evaluates the ResNet model across three different datasets and saves the results in the specified results directory.


To evaluate the trained vit_L32 model using the 'evaluation_single_detector_vit_L32.py' script, run:
```
python evaluation_single_detector_vit_L32.py --model_name model_vit_L32 \
    --data_paths "FRLL_FaceMorpher:./dataset/FRLL/facemorpher,FRLL_OpenCV:./dataset/FRLL/opencv,FRLL_StyleGAN:./dataset/FRLL/stylegan" \
    --pretrained_weights ./pretrained_weights/ViT_L_32.pth \
    --results_path ./results --results_name vit_L32_evaluation
```
This command evaluates the vit_L32 model across three different datasets and saves the results in the specified results directory.


To evaluate the trained vit_B16 model using the 'evaluation_single_detector_vit_B16.py' script, run:
```
python evaluation_single_detector_vit_B16.py --model_name model_vit_B16 \
    --data_paths "FRLL_FaceMorpher:./dataset/FRLL/facemorpher,FRLL_OpenCV:./dataset/FRLL/opencv,FRLL_StyleGAN:./dataset/FRLL/stylegan" \
    --pretrained_weights ./pretrained_weights/ViT_B_16.pth \
    --results_path ./results --results_name vit_B16_evaluation
```
This command evaluates the vit_B16 model across three different datasets and saves the results in the specified results directory.


### Plotting Results
The evaluation script automatically plots ROC and DET curves if the `--plot` flag is set to true. The plots are saved in the specified results directory.

## Project Structure
- `models/`: Contains the Python modules for different deep learning models.
- `utils/`: Utility functions such as image processing and accuracy calculations.
- `dataset/`: Dataset directory where training and validation data should be placed.
- `pretrained_weights/`: Store your pre-trained model weights here.
- `results/`: Output directory for evaluation results and plots.

## Contributing
Contributions to this project are welcome. Please fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
"# Face-Morphing-Attack-Detection-Benchmark" 
"# Face-Morphing-Attack-Detection-Benchmark" 
