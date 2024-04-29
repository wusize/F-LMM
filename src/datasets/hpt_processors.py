from src.datasets.llava_processors import CustomLlavaImageProcessor
CustomHPTImageProcessor = CustomLlavaImageProcessor


if __name__ == "__main__":
    from PIL import Image
    image_size = 588
    image_processor = CustomHPTImageProcessor.from_pretrained(
        pretrained_model_name_or_path='HyperGAI/HPT',
        subfolder='visual_encoder',
        size={"shortest_edge": image_size},
        crop_size={"height": image_size, "width": image_size}
    )
    image = Image.open('src/datasets/000000000139.jpg')
    image_data = image_processor.preprocess(image)
    print(image_data.keys())
