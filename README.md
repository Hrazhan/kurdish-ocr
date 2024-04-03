# Kurdish OCR
<p align="center">
<img src="https://www.razhan.ai/_next/image?url=/static/images/projects/kurdish-ocr.webp&w=1200&q=75" alt="Banner Image" height="240" width="1200">

<br>
    <p align="center">
    <a href="https://huggingface.co/spaces/razhan/kurdish-ocr">🌐 Live Demo</a> |
    <a href="https://huggingface.co/razhan/trocr-base-ckb">📦 Base Model</a> |
    <a href="https://huggingface.co/razhan/trocr-handwritten-ckb">✍️ Handwritten Model</a> |
    <a href="https://huggingface.co/datasets/razhan/ktr">🗃️ Data</a>
    </p>
</p>



This project is an implementation of an Optical Character Recognition (OCR) system for the Central Kurdish language. But it can be easily extended to other languages since its data is generated synthetically. The architecture is a Vision Encoder-Decoder where the encoder can be any transformer vision model and the decoder can be any pretrained language model. The cross attention layers of the final model will be added to the decoder. The model supports single line by default to extend it to multi-line you can either train your own text dection or use something like CRAFT (which in my experience is not so good with perso-arabic scripts)


## Usage

You can install the requirements using pip:
```sh
pip install -r requirements.txt
```

You will need a single line text corpus and fonts for your language of choice
- [`gen_vocab.py`](utils/gen_vocab.py) Generate the vocab from the OSCAR corpus and wikipedia. Modify `--chars` to the characters you want to keep in the final corpus
- [`gen_ocr_data.py`](utils/gen_ocr_data.py) Generates the final dataset with various filters and distortions. Modify the number of lines in this script according to your corpus and have the fonts in `data/fonts` directory.
- ['init_model.py`](init_model.py) Initialize the model
- [`accelerate_train.py`](accelerate_train.py) or [`train.py](train.py) is used to train the model.
- [`inference.py`](inference.py) runs the model through command line.
- [`app.py`](app.py) Is a UI where can run the model with CRAFT for multi-line text recognition.



> Note: If you wanna train on handwritten Kurdish data, download the the dataset from [here](https://data.krd/kurdish-handwritten-words) and delete the .DS_Store file. Pass `--handwritten_dataset` to `train.py` a class for that dataset is implement in [`dataset.py`](dataset.py).

## License

This project is open-source and available under the GNU General Public License v3.0
