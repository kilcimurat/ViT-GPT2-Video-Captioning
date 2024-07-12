# ViT-GPT2 Video Captioning

This is the official codebase for paper: A New Video Captioning Framework with Visual Transformer Encoder and GPT-2 Based Decoder [https://www.amerikakongresi.org/_files/ugd/797a84_6c6dfe33b3334093907214838e7d9c1d.pdf].


```
@article{kilci2023fusion,
  title={A New Video Captioning Framework with Visual Transformer Encoder and GPT-2 Based Decoder},
  author={Kılcı, Murat and Ö. Atılım, Koca and Cayli, Özkan and Kılıç, Volkan},
  journal={Latin America 8th International Conference on Scientific Researches},
  pages={322--329},
  year={2024}
}
```
## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Result](#result)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Video captioning is the task of generating a textual description for a given video. This model combines Vision Transformer (ViT) for video frame feature extraction and GPT-2 for natural language generation.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/kilcimurat/vit_gpt2_video_captioning.git
cd vit_gpt2_video_captioning
pip install -r environment.txt
 ```

## Usage

You can start generating captions for images by running the main script. For example:

```bash
python main.py --video_path path_to_your_video.mp4
```
This will process the image at `path_to_your_video.jpg` and generate a descriptive caption based on the trained model.

## Result
Our model was evaluated on the VizWiz and MSCOCO Captions datasets. Below are some of the results achieved by our model.

### Examples
![resim](https://github.com/kilcimurat/vit_gpt2_video_captioning/blob/main/result.png)



## Project Structure

- `data/`: Contains datasets and pre-trained models.
- `scripts/`: Contains various scripts for training, testing, and evaluation.
- `models/`: Contains model definitions and architectures.
- `utils/`: Contains utility functions.
- `main.py`: The main script to run the captioning process.
- `environment.txt`: A list of required Python libraries.
- `README.md`: This readme file.


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


