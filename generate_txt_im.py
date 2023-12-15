import os
import torch
import argparse
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from my_utils import create_directory
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images from a prompt.")
    parser.add_argument(
        "--num-images",
        type=int,
        default=5000,
        help="number of images to generate",
    )
    parser.add_argument("--input", type=str, help="folder to save images to")
    parser.add_argument("--folder", type=str, help="folder to save images to")
    parser.add_argument(
        "--part",
        type=int,
        help="prompt to generate images from",
    )
    args, _ = parser.parse_known_args()

    return args


def read_sentences_from_file(file_path):
    with open(file_path, "r") as file:
        sentences = file.readlines()
    # Removing newline characters from each sentence
    sentences = [sentence.strip() for sentence in sentences]
    return sentences


def main(args):

    prompt_list = read_sentences_from_file(f"./{args.input}.txt")
    output_folder = os.path.join("datasets/stable_diffusion", f"{args.folder}/images")
    create_directory(output_folder)
    model_id = "stabilityai/stable-diffusion-2"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to("cuda")
    # Writing the strings to a file
    for j in range(args.num_images):
        prompt = prompt_list[j % len(prompt_list)]
        image = pipe(
            prompt,
            height=768,
            width=768,
            num_inference_steps=50,
            guidance_scale=7.5,
        ).images[0]
        image.save(
            os.path.join(
                output_folder,
                f"im_{j+(args.part*args.num_images)}_p{j%len(prompt_list)}.png",
            )
        )
        print(f"image {j} saved")


if __name__ == "__main__":
    args = parse_args()
    main(args)
