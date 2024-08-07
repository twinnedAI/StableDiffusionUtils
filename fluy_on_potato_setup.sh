#!/bin/bash
python -m venv venv
source venv/bin/activate
pip install Flask huggingface-hub diffusers[torch] huggingface-cli torch torchvision transformers sentencepiece protobuf accelerate optimum-quanto
huggingface-cli login --token $HF_TOKEN
python_file="flux_on_potato.py"

cat << EOF > $python_file
import logging
from flask import Flask, request, jsonify
import torch
from optimum.quanto import freeze, qfloat8, quantize
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from io import BytesIO
import base64
from PIL import Image
import random  # Import the random module

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Define global variables for model components
pipe = None
generator = None

def initialize_model():
    global pipe, generator
    
    logging.info("Initializing model components...")

    dtype = torch.bfloat16
    
    # Set up the model components
    bfl_repo = "black-forest-labs/FLUX.1-dev"
    revision = "refs/pr/3"

    try:
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler", revision=revision)
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
        text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype, revision=revision)
        tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype, revision=revision)
        vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype, revision=revision)
        transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype, revision=revision)

        logging.info("Model components loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model components: {e}")
        raise e

    # Quantize and freeze model components
    logging.info("Quantizing and freezing model components...")
    try:
        quantize(transformer, weights=qfloat8)
        freeze(transformer)

        quantize(text_encoder_2, weights=qfloat8)
        freeze(text_encoder_2)
        logging.info("Quantization and freezing completed.")
    except Exception as e:
        logging.error(f"Error during quantization and freezing: {e}")
        raise e

    # Initialize the pipeline
    logging.info("Initializing the pipeline...")
    try:
        pipe = FluxPipeline(
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=None,
            tokenizer_2=tokenizer_2,
            vae=vae,
            transformer=None,
        )
        pipe.text_encoder_2 = text_encoder_2
        pipe.transformer = transformer
        pipe.enable_model_cpu_offload()

        generator = torch.Generator().manual_seed(12345)
        logging.info("Pipeline initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing pipeline: {e}")
        raise e

@app.route('/generate', methods=['POST'])
def generate_image():
    logging.info("Received request for image generation.")
    try:
        # Parse the input data
        data = request.json
        prompt = data.get('prompt', '')
        width = data.get('width', 1024)
        height = data.get('height', 1024)
        num_inference_steps = data.get('num_inference_steps', 20)
        guidance_scale = data.get('guidance_scale', 3.5)

        logging.info(f"Generating image with prompt: '{prompt}'")
        logging.info(f"Image dimensions: {width}x{height}, Steps: {num_inference_steps}, Guidance Scale: {guidance_scale}")
        
         # Generate a random seed for each image generation
        random_seed = random.randint(0, 2**32 - 1)  # 32-bit random seed
        logging.info(f"Using random seed: {random_seed}")
        generator = torch.Generator().manual_seed(random_seed)

        # Generate the image
        image = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=guidance_scale,
        ).images[0]

        logging.info("Image generated successfully.")

        # Convert the image to Base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        logging.info("Image converted to Base64.")
        return jsonify({'image': img_str})

    except Exception as e:
        logging.error(f"Error during image generation: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logging.info("Starting API server...")
    initialize_model()
    logging.info("Model initialized. API server is running.")
    app.run(host='0.0.0.0', port=5000)

EOF
chmod +x $python_file
python $python_file