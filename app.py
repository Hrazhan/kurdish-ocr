import gradio as gr
import cv2
from transformers import pipeline
from PIL import Image
from craft_text_detector import Craft
import os

model_ckpt = "razhan/trocr-base-ckb"
ocr = pipeline("image-to-text", model=model_ckpt, device=0)


craft = Craft(
    output_dir=None,
    crop_type="poly",
    export_extra=False,
    text_threshold=0.7,
    link_threshold=0.4,
    low_text=0.4,
    long_size=1280,
    cuda=False,
)


def recoginition(img, prediction_result, ocr):
    text = []
    for i, j in enumerate(prediction_result["boxes"]):
        roi = img[
            int(prediction_result["boxes"][i][0][1]) : int(
                prediction_result["boxes"][i][2][1]
            ),
            int(prediction_result["boxes"][i][0][0]) : int(
                prediction_result["boxes"][i][2][0]
            ),
        ]
        image = Image.fromarray(roi).convert("RGB")
        generated_text = ocr(image)[0]["generated_text"]
        text.append(generated_text)
    return "\n".join(text)


def visualize(img, prediction_result):
    for i, j in enumerate(prediction_result["boxes"]):
        y1 = int(prediction_result["boxes"][i][0][1])
        y2 = int(prediction_result["boxes"][i][2][1])

        x1 = int(prediction_result["boxes"][i][0][0])
        x2 = int(prediction_result["boxes"][i][2][0])

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return Image.fromarray(img)


def multi_line(img):
    detection = craft.detect_text(img)
    viz = visualize(img, detection)
    text = recoginition(img, detection, ocr)

    return viz, text


def single_line(image):
    generated_text = ocr(image)[0]["generated_text"]
    return generated_text


txt_output = gr.Textbox()
image_output = gr.Image(type="filepath")
# mode_input = gr.Radio(["single-line", "multi-line"], label="Mode", info="Wether to use the OCR model alone or with a text detection model (CRAFT)"),


article = "<p style='text-align: center'> Made with ❤️ by <a href='https://hrazhan.github.io'>Razhan Hameed</a></p>"
# examples =[["1.jpg"], ["2.jpg"]]
examples = []

# get the path of all the files inside the folder data/examples put them in the format [["1.jpg"], ["2.jpg"]]
for file in os.listdir("data/examples"):
    examples.append([os.path.join("data/examples", file)])
print(examples)


with gr.Blocks() as demo:
    gr.HTML(
        """
    <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
        <h1 style="font-weight: 900; font-size: 3rem; margin: 0rem"> 🚀 Kurdish OCR </h1>

        <p style="font-weight: 450; font-size: 1rem; margin: 0rem"> Demo for Kurdish OCR encoder-decoder vision model on single-text line images.</p>
        <h2 style="text-align: left; font-weight: 450; font-size: 1rem; margin-top: 0.5rem; margin-bottom: 0.5rem">
            <ul style="list-style-type:disc;">
                <li>The model's original training focuses on recognizing text in single lines. Once you upload the image, use the pen icon to crop the image into a single line format</li>
                <li>For images containing multiple lines of text, you can utilize the multi-line tab. Please be aware that the CRAFT text detection used in the pipeline may encounter difficulties with Arabic letters, resulting in potential inaccuracies in detecting the boundaries and angles of the text. The OCR model will receive the identified regions, but it might not provide accurate results if certain parts of the letters are excluded in the captured regions. </li>
            </ul>
        </h2>
    </div>
    """
    )

    with gr.Tab("Signle line"):
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="Image")
                button = gr.Button("Submit")

            with gr.Column(scale=1):
                txt_output = gr.Textbox(label="Extracted text")

        gr.Markdown("## Single Line Examples")
        gr.Examples(
            examples=examples,
            inputs=image,
            outputs=txt_output,
            fn=single_line,
            examples_per_page=20,
            cache_examples=False,
            run_on_click=True,
        )
        button.click(single_line, inputs=[image], outputs=[txt_output])

    with gr.Tab("Multi line"):
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(label="Image")
                button = gr.Button("Submit")

            with gr.Column(scale=1):
                txt_output = gr.Textbox(label="Extracted text")
                image_output = gr.Image(type="filepath")

        button.click(multi_line, inputs=[image], outputs=[image_output, txt_output])
    # at the bottom write its made by Razhan
    gr.Markdown(article)
    demo.launch()
