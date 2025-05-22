import os

import numpy as np
import onnx
import tensorrt as trt
import torch
from diffusers import AutoencoderKL, StableDiffusionXLInpaintPipeline  # For loading VAE

# --- Configuration ---
# Replace with your model ID (e.g., "stabilityai/stable-diffusion-xl-base-1.0" or your inpainting model ID)
ORIGINAL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
OUTPUT_DIR = "./sdxl_vae_decoder_onnx_trt"  # Directory to save exported files
# Path for the initial ONNX export (might be >2GB)
ONNX_VAE_DECODER_INITIAL_PATH = os.path.join(OUTPUT_DIR, "vae_decoder", "model_initial.onnx")
# Path for the ONNX model with weights stored externally (this one is used for TRT)
FINAL_ONNX_VAE_DECODER_PATH = os.path.join(OUTPUT_DIR, "vae_decoder", "model_external.onnx")
# Path for the final TensorRT engine
TRT_VAE_DECODER_PATH = os.path.join(OUTPUT_DIR, "vae_decoder", "model.plan")

# VAE Decoder specific parameters (typical for SDXL)
BATCH_SIZE = 1  # Default/Optimal batch size for TRT profile
LATENT_CHANNELS = 4  # Number of channels in the latent space
IMAGE_CHANNELS = 3  # Number of channels in the output image (RGB)
IMAGE_HEIGHT = 512  # Target image height
IMAGE_WIDTH = 512  # Target image width
VAE_SCALE_FACTOR = 8  # VAE downscaling/upscaling factor (std for SDXL)
TORCH_DTYPE = torch.float16  # Operate in FP16 for efficiency

# Create output directories if they don't exist
os.makedirs(os.path.join(OUTPUT_DIR, "vae_decoder"), exist_ok=True)

# Initialize TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# --- VAE Decoder Wrapper for ONNX Export ---
class VAEDecoderONNXWrapper(torch.nn.Module):
    """
    Wrapper for AutoencoderKL's decode method to ensure it returns a single tensor
    suitable for ONNX export.
    """

    def __init__(self, vae_model: AutoencoderKL):
        super().__init__()
        self.vae_model = vae_model

    def forward(self, latents: torch.FloatTensor) -> torch.FloatTensor:
        # The decode method of AutoencoderKL might return a class or a tuple.
        # We ensure it returns only the sample tensor.
        # return_dict=False should make it return a tuple where the first element is the sample.
        decoded_output = self.vae_model.decode(latents, return_dict=False)
        if isinstance(decoded_output, tuple):
            return decoded_output[0]  # The actual decoded image tensor
        return decoded_output  # Should be the tensor if not a tuple


# --- Function to Export VAE Decoder to ONNX ---
def export_vae_decoder_to_onnx(
    vae_model: AutoencoderKL,
    initial_onnx_export_path: str,
    final_onnx_path_for_trt: str,
    batch_size: int,
    latent_channels: int,
    latent_height: int,
    latent_width: int,
    torch_dtype: torch.dtype = torch.float16,
) -> str | None:
    """
    Exports the VAE decoder part of the AutoencoderKL model to ONNX format.
    Handles large model sizes by saving with external data.

    Args:
        vae_model: The loaded AutoencoderKL model.
        initial_onnx_export_path: Path to save the initial (potentially large) ONNX model.
        final_onnx_path_for_trt: Path to save the ONNX model with external data (used for TRT).
        batch_size: Batch size for dummy input.
        latent_channels: Number of channels for latent input.
        latent_height: Height of the latent input.
        latent_width: Width of the latent input.
        torch_dtype: PyTorch dtype for export (e.g., torch.float16).

    Returns:
        Path to the final ONNX model with external data, or None on failure.
    """
    print(f"Starting VAE Decoder ONNX export to: {initial_onnx_export_path}")
    vae_model.eval().to("cuda", dtype=torch_dtype)  # Ensure eval mode and correct device/dtype

    # Wrap the VAE model's decode method
    wrapped_decoder = VAEDecoderONNXWrapper(vae_model).cuda().eval()

    # Create dummy input for the VAE Decoder
    dummy_latents = torch.randn(batch_size, latent_channels, latent_height, latent_width, dtype=torch_dtype).cuda()
    print(f"  Dummy latents shape for VAE Decoder export: {dummy_latents.shape}")

    # Define input and output names for the ONNX graph
    input_names = ["latent_sample"]  # Name for the latent input tensor
    output_names = ["sample"]  # Name for the decoded image output tensor

    # Define dynamic axes for inputs and outputs
    dynamic_axes = {
        input_names[0]: {0: "batch_size", 2: "latent_height", 3: "latent_width"},
        output_names[0]: {0: "batch_size", 2: "image_height", 3: "image_width"},
    }

    try:
        # Step 1: Initial ONNX export
        print(f"  Exporting to ONNX (initial pass) at {initial_onnx_export_path}...")
        torch.onnx.export(
            wrapped_decoder,
            (dummy_latents,),  # Arguments to the forward method of wrapped_decoder
            initial_onnx_export_path,
            export_params=True,
            opset_version=17,  # Recommended for SDXL components
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        print(f"  VAE Decoder ONNX initially exported to: {initial_onnx_export_path}")

        # Step 2: Verify the initially exported model (using file path for large models)
        print(f"  Checking ONNX VAE Decoder model at path: {initial_onnx_export_path}")
        onnx.checker.check_model(initial_onnx_export_path)
        print("  ONNX VAE Decoder model (from path) checked successfully.")

        # Step 3: Reload and save with external data to handle potential >2GB protobuf limit
        print(f"  Reloading ONNX model from {initial_onnx_export_path} to save with external data...")
        # Load the model structure without immediately loading all tensor data if it's already external
        onnx_model_loaded = onnx.load(initial_onnx_export_path, load_external_data=False)

        # Define the name for the external weights file (will be saved alongside final_onnx_path_for_trt)
        external_weights_filename = os.path.splitext(os.path.basename(final_onnx_path_for_trt))[0] + "_weights.dat"

        print(f"  Saving VAE Decoder ONNX model with external data.")
        print(f"    Final ONNX graph: {final_onnx_path_for_trt}")
        print(f"    External weights: {external_weights_filename}")

        onnx.save_model(
            onnx_model_loaded,
            final_onnx_path_for_trt,
            save_as_external_data=True,
            all_tensors_to_one_file=True,  # Store all external tensors in one separate file
            location=external_weights_filename,  # Name of the file where weights will be stored
            size_threshold=1024,  # Tensors larger than 1KB will be stored externally
        )
        print("  VAE Decoder ONNX model saved with external data successfully.")
        return final_onnx_path_for_trt

    except Exception as e:
        print(f"ERROR during VAE Decoder ONNX export or external data conversion: {e}")
        import traceback

        traceback.print_exc()
        return None


# --- Function to Convert ONNX VAE Decoder to TensorRT Engine ---
def convert_vae_decoder_onnx_to_trt(
    onnx_path: str,
    trt_engine_path: str,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
    min_latent_h: int,
    opt_latent_h: int,
    max_latent_h: int,
    min_latent_w: int,
    opt_latent_w: int,
    max_latent_w: int,
    latent_channels: int,
    fp16_mode: bool = True,
    workspace_size_gb: int = 2,  # VAE Decoder is generally smaller than UNet
) -> bool:
    """
    Converts an ONNX VAE Decoder model to a TensorRT engine.

    Args:
        onnx_path: Path to the ONNX model (preferably with external data).
        trt_engine_path: Path to save the built TensorRT engine.
        min_batch, opt_batch, max_batch: Batch size range for the TRT profile.
        min_latent_h, opt_latent_h, max_latent_h: Latent height range for TRT profile.
        min_latent_w, opt_latent_w, max_latent_w: Latent width range for TRT profile.
        latent_channels: Number of channels for the latent input.
        fp16_mode: Whether to enable FP16 precision for the TRT engine.
        workspace_size_gb: GPU memory (in GB) for TRT builder workspace.

    Returns:
        True if conversion was successful, False otherwise.
    """
    print(f"Starting ONNX VAE Decoder to TensorRT conversion: {trt_engine_path}")
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()

    # Set workspace size
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size_gb * (1024**3))

    # Enable FP16 mode if specified
    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  FP16 mode enabled for TensorRT.")

    # Create network definition from ONNX parser
    # EXPLICIT_BATCH allows for dynamic batch sizes
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    if not os.path.exists(onnx_path):
        print(f"ERROR: ONNX file not found at {onnx_path}")
        return False

    print(f"  Parsing ONNX model from: {onnx_path}")
    os.chdir(os.path.dirname(onnx_path))
    onnx_path = os.path.basename(onnx_path)
    with open(onnx_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            print("ERROR: Failed to parse the ONNX VAE Decoder file.")
            for error_idx in range(parser.num_errors):
                print(f"  Parser error ({error_idx}): {parser.get_error(error_idx)}")
            return False
    print("  ONNX VAE Decoder file parsed successfully.")

    # Define an optimization profile for dynamic input shapes
    profile = builder.create_optimization_profile()
    # The input name must match the name defined during ONNX export (e.g., "latent_sample")
    latent_input_name = "latent_sample"  # Should match input_names[0] from export

    profile.set_shape(
        latent_input_name,
        min=(min_batch, latent_channels, min_latent_h, min_latent_w),
        opt=(opt_batch, latent_channels, opt_latent_h, opt_latent_w),
        max=(max_batch, latent_channels, max_latent_h, max_latent_w),
    )
    config.add_optimization_profile(profile)
    print(f"  Optimization profile added for '{latent_input_name}':")
    print(f"    Min Shape: ({min_batch}, {latent_channels}, {min_latent_h}, {min_latent_w})")
    print(f"    Opt Shape: ({opt_batch}, {latent_channels}, {opt_latent_h}, {opt_latent_w})")
    print(f"    Max Shape: ({max_batch}, {latent_channels}, {max_latent_h}, {max_latent_w})")

    # Build the TensorRT engine
    print("  Building TensorRT VAE Decoder engine... This may take a few minutes.")
    # serialized_engine = builder.build_serialized_network(network, config) # Old API
    plan = builder.build_serialized_network(network, config)  # New API for TRT 8+

    if plan is None:
        print("ERROR: Failed to build the TensorRT VAE Decoder engine.")
        return False
    print("  TensorRT VAE Decoder engine built successfully.")

    # Save the engine to a file
    trt_engine_path = os.path.basename(trt_engine_path)
    with open(trt_engine_path, "wb") as engine_file:
        engine_file.write(plan)
    print(f"TensorRT VAE Decoder engine saved to: {trt_engine_path}")
    return True


# --- Main Execution Script ---
if __name__ == "__main__":
    print("Starting VAE Decoder to ONNX and TensorRT conversion process...")

    # 1. Load the VAE model from the Diffusers pipeline
    print(f"Loading VAE model from Hugging Face model ID: {ORIGINAL_MODEL_ID}")
    vae = None
    try:
        # SDXL VAE is typically in a subfolder named "vae"
        vae = AutoencoderKL.from_pretrained(
            ORIGINAL_MODEL_ID,
            subfolder="vae",
            torch_dtype=TORCH_DTYPE,  # Load in target dtype for consistency
            use_safetensors=True,
        )
        print("VAE model loaded successfully directly.")
    except Exception as e:
        print(f"Warning: Could not load VAE directly from subfolder: {e}")
        print("Attempting to load VAE by loading the full pipeline (this might be slower and use more memory)...")
        try:
            # Fallback: Load the full pipeline and extract the VAE
            # This is less efficient but can be a fallback if direct loading fails
            # or if the VAE isn't in a standard "vae" subfolder for a custom model.
            temp_pipeline = StableDiffusionXLPipeline.from_pretrained(
                ORIGINAL_MODEL_ID,
                torch_dtype=TORCH_DTYPE,
                use_safetensors=True,
                # Load only essential components if possible, though VAE is usually core
            )
            vae = temp_pipeline.vae
            del temp_pipeline  # Free up memory from the rest of the pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("VAE extracted from full pipeline successfully.")
        except Exception as e_pipe:
            print(f"ERROR: Failed to load VAE model even from full pipeline: {e_pipe}")
            exit(1)

    if vae is None:
        print("ERROR: VAE model could not be loaded. Exiting.")
        exit(1)

    # 2. Export VAE Decoder to ONNX (if the final ONNX file doesn't already exist)
    actual_final_onnx_path = None
    if not os.path.exists(FINAL_ONNX_VAE_DECODER_PATH):
        print(f"Final ONNX VAE Decoder ({FINAL_ONNX_VAE_DECODER_PATH}) not found. Starting export process...")
        actual_final_onnx_path = export_vae_decoder_to_onnx(
            vae_model=vae,
            initial_onnx_export_path=ONNX_VAE_DECODER_INITIAL_PATH,
            final_onnx_path_for_trt=FINAL_ONNX_VAE_DECODER_PATH,
            batch_size=BATCH_SIZE,  # Using the global BATCH_SIZE for dummy input
            latent_channels=LATENT_CHANNELS,
            latent_height=IMAGE_HEIGHT // VAE_SCALE_FACTOR,
            latent_width=IMAGE_WIDTH // VAE_SCALE_FACTOR,
            torch_dtype=TORCH_DTYPE,
        )
        if not actual_final_onnx_path:
            print("ERROR: ONNX VAE Decoder export failed. Exiting.")
            exit(1)
    else:
        actual_final_onnx_path = FINAL_ONNX_VAE_DECODER_PATH
        print(f"Final ONNX VAE Decoder already exists, skipping export: {actual_final_onnx_path}")

    # 3. Convert ONNX VAE Decoder to TensorRT Engine (if TRT engine doesn't already exist)
    if actual_final_onnx_path and os.path.exists(actual_final_onnx_path):
        if not os.path.exists(TRT_VAE_DECODER_PATH):
            print(f"TensorRT VAE Decoder engine ({TRT_VAE_DECODER_PATH}) not found. Starting conversion...")
            # Define dynamic shape profiles for TensorRT engine
            # These can be adjusted based on expected usage patterns
            success = convert_vae_decoder_onnx_to_trt(
                onnx_path=actual_final_onnx_path,
                trt_engine_path=TRT_VAE_DECODER_PATH,
                # Min/Opt/Max for Batch Size
                min_batch=1,
                opt_batch=BATCH_SIZE,  # Optimal batch size from config
                max_batch=max(1, BATCH_SIZE * 2),  # Example: allow up to 2x optimal batch
                # Min/Opt/Max for Latent Height (derived from image height)
                min_latent_h=(IMAGE_HEIGHT // VAE_SCALE_FACTOR) // 2,  # Example: allow half optimal height
                opt_latent_h=IMAGE_HEIGHT // VAE_SCALE_FACTOR,
                max_latent_h=IMAGE_HEIGHT // VAE_SCALE_FACTOR,
                # Min/Opt/Max for Latent Width (derived from image width)
                min_latent_w=(IMAGE_WIDTH // VAE_SCALE_FACTOR) // 2,  # Example: allow half optimal width
                opt_latent_w=IMAGE_WIDTH // VAE_SCALE_FACTOR,
                max_latent_w=IMAGE_WIDTH // VAE_SCALE_FACTOR,
                latent_channels=LATENT_CHANNELS,
                fp16_mode=True,  # Enable FP16 for speed
                workspace_size_gb=2,  # VAE Decoder is typically smaller than UNet
            )
            if not success:
                print("ERROR: TensorRT VAE Decoder conversion failed. Exiting.")
                exit(1)
        else:
            print(f"TensorRT VAE Decoder engine already exists, skipping conversion: {TRT_VAE_DECODER_PATH}")
    else:
        print(
            f"ERROR: Cannot convert to TensorRT. Final ONNX VAE Decoder file not found at {actual_final_onnx_path or FINAL_ONNX_VAE_DECODER_PATH}."
        )
        exit(1)

    print("VAE Decoder conversion script finished successfully.")
