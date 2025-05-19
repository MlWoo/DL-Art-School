import torch
from diffusers import StableDiffusionXLInpaintPipeline, EulerDiscreteScheduler
from PIL import Image
import tensorrt as trt
import os

os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/home/wumenglin/repo-dev/DL-Art-School-dev/codes/scripts/image/torchinductor_cache'
# --- Configuration ---
# Model ID for SDXL Inpainting
# You can also use "stabilityai/stable-diffusion-xl-refiner-1.0" with appropriate inpainting setup,
# but a dedicated inpainting pipeline is often more straightforward.
# "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" is a community model fine-tuned for inpainting.
# For official Stability AI models, you might need to construct the inpainting pipeline slightly differently
# or use specific components. Let's use a known inpainting-focused model.
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0" # Base model
# For inpainting, often a specific inpainting UNet is used or the pipeline handles it.
# The StableDiffusionXLInpaintPipeline is designed to work with SDXL base models.

# Paths for your images
INIT_IMAGE_PATH = "/home/wumenglin/repo-dev/DL-Art-School-dev/codes/scripts/image/overture-creations-5sI6fQgYIuo.png"  # Replace with your initial image file
MASK_IMAGE_PATH = "/home/wumenglin/repo-dev/DL-Art-School-dev/codes/scripts/image/overture-creations-5sI6fQgYIuo_mask.png"    # Replace with your mask image file (white for inpaint area, black for keep)
OUTPUT_IMAGE_PATH = "sdxl_inpainted_image_demo.png"

# Inference Parameters
PROMPT = "A futuristic robot standing in the room, photorealistic, 4k"
NEGATIVE_PROMPT = "blurry, low quality, unrealistic, watermark, signature, text, ugly, deformed"
NUM_INFERENCE_STEPS = 500  # Number of denoising steps
GUIDANCE_SCALE = 8.0       # How much to adhere to the prompt
STRENGTH = 0.85            # How much to transform the init_image (1.0 means ignore init_image in masked area)
SEED = 42                  # For reproducibility

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32 # Use float16 on GPU for speed


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_image(image_path, size=None):
    """Loads an image and optionally resizes it."""
    try:
        img = Image.open(image_path).convert("RGB")
        if size:
            img = img.resize(size)
        return img
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        print("Please ensure you have an image at this location or update the path.")
        print("For this demo, a placeholder image will be created if the file is missing.")
        if "mask" in image_path.lower():
            # Create a black image with a white square in the middle for the mask
            img = Image.new("L", size if size else (1024,1024), "black")
            draw_img = Image.new("L", (size[0]//2, size[1]//2) if size else (1024,1024), "white")
            img.paste(draw_img, (size[0]//4 if size else 256, size[1]//4 if size else 256))
            img = img.convert("RGB") # Inpainting pipeline expects RGB for mask too
        else:
            # Create a simple placeholder for the initial image
            img = Image.new("RGB", size if size else (1024,1024), "lightgray")
        print(f"Using a placeholder for: {image_path}")
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None



# --- Helper function to map TRT DataType to Torch DType ---
def trt_dtype_to_torch_dtype(trt_dtype):
    if trt_dtype == trt.DataType.FLOAT:
        return torch.float32
    elif trt_dtype == trt.DataType.HALF:
        return torch.float16
    elif trt_dtype == trt.DataType.BF16:
        return torch.float16
    elif trt_dtype == trt.DataType.INT8:
        return torch.int8
    elif trt_dtype == trt.DataType.INT32: 
        return torch.int32
    elif trt_dtype == trt.DataType.INT64:
        return torch.int64
    elif trt_dtype == trt.DataType.BOOL:
        return torch.bool
    elif trt_dtype == trt.DataType.UINT8:
        return torch.uint8
    else:
        raise TypeError(f"Unsupported TRT DataType: {trt_dtype}")



class TRTEngine:
    def __init__(self, engine_path, vae_scale_factor_from_config=8): 
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.stream = None
        self.output_shapes = {}
        self.input_shapes = {}
        self.input_names = []
        self.output_names = []
        self.input_dtypes_torch = {} 
        self.output_dtypes_torch = {}
        self.tensor_name_to_idx_map = {} 
        self.vae_scale_factor = vae_scale_factor_from_config 

        self._load_engine()
        self._prepare_binding_info() 

    def _load_engine(self):
        print(f"Loading TensorRT engine from: {self.engine_path}")
        print(f"DEBUG: TensorRT version: {trt.__version__}")
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {self.engine_path}")
        
        print(f"DEBUG: Type of self.engine: {type(self.engine)}")
        print(f"DEBUG: Is self.engine an ICudaEngine? {isinstance(self.engine, trt.ICudaEngine)}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")
        print("TensorRT engine loaded successfully.")

    def _prepare_binding_info(self):
        if not self.engine:
            raise RuntimeError("TRTEngine._prepare_binding_info called but self.engine is not initialized.")
        
        print("Preparing TRT engine binding information using I/O tensor API...")
        
        if not hasattr(self.engine, 'num_io_tensors'):
            raise AttributeError(f"'ICudaEngine' object (version {trt.__version__}) has no attribute 'num_io_tensors'. This is critical. Please check TensorRT installation.")

        num_io_tensors = self.engine.num_io_tensors
        print(f"DEBUG: self.engine.num_io_tensors = {num_io_tensors}")

        for i in range(num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            self.tensor_name_to_idx_map[tensor_name] = i 

            tensor_shape = tuple(self.engine.get_tensor_shape(tensor_name))
            tensor_dtype_trt = self.engine.get_tensor_dtype(tensor_name)
            
            try:
                tensor_dtype_torch = trt_dtype_to_torch_dtype(tensor_dtype_trt)
            except TypeError as e:
                print(f"Error for tensor '{tensor_name}': {e}")
                print(f"  Problematic TRT DataType was: {tensor_dtype_trt} (enum value: {int(tensor_dtype_trt)})")
                raise

            tensor_mode = self.engine.get_tensor_mode(tensor_name)

            if tensor_mode == trt.TensorIOMode.INPUT:
                self.input_shapes[tensor_name] = tensor_shape
                self.input_names.append(tensor_name)
                self.input_dtypes_torch[tensor_name] = tensor_dtype_torch
                print(f"  Input: {tensor_name} (Index: {i}), Shape: {tensor_shape}, TorchDtype: {tensor_dtype_torch}")
            elif tensor_mode == trt.TensorIOMode.OUTPUT:
                self.output_shapes[tensor_name] = tensor_shape
                self.output_names.append(tensor_name)
                self.output_dtypes_torch[tensor_name] = tensor_dtype_torch
                print(f"  Output: {tensor_name} (Index: {i}), Shape: {tensor_shape}, TorchDtype: {tensor_dtype_torch}")
            else:
                print(f"Warning: Tensor '{tensor_name}' at I/O index {i} has an unexpected mode: {tensor_mode}")
        
        self.stream = torch.cuda.Stream() # Using PyTorch CUDA stream
        print("TRT binding information prepared.")

    def __call__(self, inputs_dict: dict):
        binding_idx_to_ptr = {}

        for name, tensor in inputs_dict.items():
            if name not in self.input_names:
                print(f"Warning: Input tensor '{name}' provided but not an input for the TRT engine. Skipping.")
                continue
            
            if name not in self.tensor_name_to_idx_map: 
                raise ValueError(f"Tensor name '{name}' not found in prepared binding map. This should not happen if _prepare_binding_info was successful.")
            idx = self.tensor_name_to_idx_map[name]

            if not tensor.is_cuda:
                raise ValueError(f"Input tensor '{name}' must be on CUDA device.")
            
            expected_torch_dtype = self.input_dtypes_torch[name]
            if tensor.dtype != expected_torch_dtype:
                print(f"Warning: Input tensor '{name}' dtype mismatch. Expected {expected_torch_dtype}, got {tensor.dtype}. Casting.")
                tensor = tensor.to(expected_torch_dtype)

            if -1 in self.engine.get_tensor_shape(name): 
                if not self.context.set_input_shape(name, tuple(tensor.shape)):
                    raise ValueError(f"Failed to set binding shape for dynamic input: {name} to {tensor.shape}")
            
            binding_idx_to_ptr[idx] = tensor.data_ptr()

        outputs_torch = {}
        for name in self.output_names:
            if name not in self.tensor_name_to_idx_map: 
                raise ValueError(f"Tensor name '{name}' not found in prepared binding map for outputs. This should not happen if _prepare_binding_info was successful.")
            idx = self.tensor_name_to_idx_map[name]

            dtype_torch = self.output_dtypes_torch[name]
            
            final_alloc_shape = None
            try:
                output_shape_from_context = tuple(self.context.get_tensor_shape(name))
                current_batch_size = list(inputs_dict.values())[0].shape[0]
                if current_batch_size != output_shape_from_context[0]:
                    output_shape_from_context = (current_batch_size, ) + output_shape_from_context[1:]
                
                if -1 not in output_shape_from_context:
                    final_alloc_shape = output_shape_from_context
                else:
                    print(f"Warning: Output tensor '{name}' shape from context.get_binding_shape is still dynamic: {output_shape_from_context}. Will use fallback.")
            except AttributeError: 
                print(f"Warning: context.get_binding_shape not available on this TRT version for output '{name}'. Using fallback.")
            except Exception as e:
                print(f"Error calling context.get_binding_shape for output '{name}': {e}. Using fallback.")

            if final_alloc_shape is None:
                build_time_opt_shape = list(self.output_shapes[name]) 
                current_batch_size = list(inputs_dict.values())[0].shape[0] 
                
                if build_time_opt_shape[0] == -1 or build_time_opt_shape[0] != current_batch_size :
                    build_time_opt_shape[0] = current_batch_size
                
                if name == "sample": 
                    if len(build_time_opt_shape) == 4: 
                        if build_time_opt_shape[1] == -1 : 
                            print(f"Info: Output '{name}', dim 1 (Channels) was -1. Setting to global IMAGE_CHANNELS: {IMAGE_CHANNELS}")
                            build_time_opt_shape[1] = IMAGE_CHANNELS
                        if build_time_opt_shape[2] == -1: 
                            print(f"Info: Output '{name}', dim 2 (Height) was -1. Setting to global IMAGE_HEIGHT: {IMAGE_HEIGHT}")
                            build_time_opt_shape[2] = IMAGE_HEIGHT 
                        if build_time_opt_shape[3] == -1: 
                            print(f"Info: Output '{name}', dim 3 (Width) was -1. Setting to global IMAGE_WIDTH: {IMAGE_WIDTH}")
                            build_time_opt_shape[3] = IMAGE_WIDTH 
                elif name == "out_sample": 
                    if len(build_time_opt_shape) == 4:
                        if build_time_opt_shape[1] == -1: 
                            print(f"Info: Output '{name}', dim 1 (Channels) was -1. Setting to global UNET_OUTPUT_CHANNELS: {UNET_OUTPUT_CHANNELS}")
                            build_time_opt_shape[1] = UNET_OUTPUT_CHANNELS
                        if build_time_opt_shape[2] == -1: 
                            latent_height = IMAGE_HEIGHT // self.vae_scale_factor
                            print(f"Info: Output '{name}', dim 2 (Latent Height) was -1. Setting to derived latent_height: {latent_height}")
                            build_time_opt_shape[2] = latent_height
                        if build_time_opt_shape[3] == -1: 
                            latent_width = IMAGE_WIDTH // self.vae_scale_factor
                            print(f"Info: Output '{name}', dim 3 (Latent Width) was -1. Setting to derived latent_width: {latent_width}")
                            build_time_opt_shape[3] = latent_width
                else: 
                    for i_dim in range(1, len(build_time_opt_shape)): 
                        if build_time_opt_shape[i_dim] == -1:
                            print(f"CRITICAL WARNING: Dimension {i_dim} for output '{name}' is -1 in build_time_opt_shape and unresolved. Using 1 as placeholder. THIS IS RISKY.")
                            build_time_opt_shape[i_dim] = 1
                
                final_alloc_shape = tuple(build_time_opt_shape)
                print(f"DEBUG: Using fallback allocation shape for '{name}': {final_alloc_shape}")
            
            outputs_torch[name] = torch.empty(size=final_alloc_shape, dtype=dtype_torch, device="cuda")
            binding_idx_to_ptr[idx] = outputs_torch[name].data_ptr()

        final_bindings_list = [0] * self.engine.num_io_tensors 
        for original_io_index, ptr in binding_idx_to_ptr.items(): 
            if original_io_index >= len(final_bindings_list): 
                raise IndexError(f"Binding index {original_io_index} (derived from I/O index) is out of range for bindings list of size {len(final_bindings_list)}.")
            final_bindings_list[original_io_index] = ptr
        
        for i in range(len(final_bindings_list)):
            if final_bindings_list[i] == 0: 
                missing_tensor_name = None
                for name, mapped_idx in self.tensor_name_to_idx_map.items(): 
                    if mapped_idx == i:
                        missing_tensor_name = name
                        break
                raise RuntimeError(f"Binding for index {i} (tensor: {missing_tensor_name or 'Unknown'}) was not set. Ensure all engine inputs/outputs are handled.")

        stream_handle = self.stream.cuda_stream # Using PyTorch CUDA stream handle
        
        executed_successfully = False
        self.context.execute_v2(bindings=final_bindings_list)
        
        # if hasattr(self.context, "execute_async_v3"):
        #     try:
        #         for i, ptr in enumerate(final_bindings_list):
        #             if hasattr(self.context, "set_binding_address"):
        #                 if not self.context.set_binding_address(i, ptr):
        #                     raise RuntimeError(f"Failed to set binding address for binding index {i} using set_binding_address.")
        #             elif hasattr(self.context, "set_tensor_address"): 
        #                 tensor_name_for_idx_i = None
        #                 for t_name, t_idx in self.tensor_name_to_idx_map.items():
        #                     if t_idx == i:
        #                         tensor_name_for_idx_i = t_name
        #                         break
        #                 if tensor_name_for_idx_i is None:
        #                     raise RuntimeError(f"Could not find tensor name for binding index {i} to use with set_tensor_address.")
        #                 if not self.context.set_tensor_address(tensor_name_for_idx_i, ptr):
        #                      raise RuntimeError(f"Failed to set tensor address for tensor '{tensor_name_for_idx_i}' (binding index {i}) using set_tensor_address.")
        #             else:
        #                 raise AttributeError("IExecutionContext has execute_async_v3 but is missing set_binding_address and set_tensor_address. Cannot set bindings for v3.")

        #         if not self.context.execute_async_v3(stream_handle=stream_handle): 
        #             raise RuntimeError("execute_async_v3 call failed to enqueue.")
        #         executed_successfully = True
        #     except TypeError as e_v3_type: 
        #          if "incompatible function arguments" in str(e_v3_type) and "bindings" in str(e_v3_type):
        #             print(f"DEBUG: execute_async_v3 was called with 'bindings' kwarg due to previous logic, but signature is (self, stream_handle). Error: {e_v3_type}")
        #          else:
        #             print(f"DEBUG: execute_async_v3 path failed with TypeError: {e_v3_type}")
        #     except Exception as e_v3:
        #         print(f"DEBUG: execute_async_v3 path failed: {e_v3}")

        # if not executed_successfully and hasattr(self.context, "execute_async_v2"):
        #     try:
        #         self.context.execute_async_v2(bindings=final_bindings_list, stream_handle=stream_handle)
        #         executed_successfully = True
        #     except AttributeError as e_v2: 
        #         if "execute_async_v3" in str(e_v2): 
        #             print(f"DEBUG: execute_async_v2 failed suggesting v3 ({e_v2}), but v3 path was already tried or failed. This is a problematic state.")
        #         else: 
        #             print(f"DEBUG: execute_async_v2 failed for a different reason: {e_v2}")
        #     except Exception as e_v2_other:
        #          print(f"DEBUG: execute_async_v2 failed with a non-AttributeError: {e_v2_other}")


        # if not executed_successfully and hasattr(self.context, "execute_async"):
        #     try:
        #         current_batch_size = list(inputs_dict.values())[0].shape[0]
        #         self.context.execute_async(batch_size=current_batch_size,
        #                                    bindings=final_bindings_list,
        #                                    stream_handle=stream_handle)
        #         executed_successfully = True
        #     except Exception as e_base:
        #         print(f"DEBUG: execute_async (base) failed: {e_base}")

        # if not executed_successfully:
        #     raise AttributeError("No suitable execute_async method (v3, v2, or base) was found or executed successfully on IExecutionContext.")

        # self.stream.synchronize()
        return outputs_torch


UNET_TRT_PATH = "./sdxl_unet_onnx_trt_cfg_fp8_1024_f/unet/model.plan" 


class UNetWrapper():
    def __init__(self):
        self.unet = TRTEngine(UNET_TRT_PATH)

    def __call__(self,
                latent_model_input,
                t,
                encoder_hidden_states,
                timestep_cond,
                cross_attention_kwargs,
                added_cond_kwargs,
                return_dict=False):
        current_batch_size_for_unet = latent_model_input.shape[0]
        timestep_pt = torch.tensor([t.item(), ] * current_batch_size_for_unet, device="cuda", dtype=torch.int64)

        unet_inputs_trt = {
            "sample": latent_model_input.to(DTYPE),
            "timestep": timestep_pt, 
            "encoder_hidden_states": encoder_hidden_states.to(DTYPE),
            "text_embeds": added_cond_kwargs["text_embeds"].to(DTYPE),
            "time_ids": added_cond_kwargs["time_ids"].to(DTYPE)
        }
        unet_outputs_trt = self.unet(unet_inputs_trt)
        noise_pred = unet_outputs_trt["out_sample"]
        return [noise_pred]


class UNetWrapper2(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = torch.compile(unet)

    def forward(self,
                latent_model_input,
                t,
                encoder_hidden_states,
                timestep_cond,
                cross_attention_kwargs,
                added_cond_kwargs,
                return_dict=False):
        
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states,
            timestep_cond=timestep_cond,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=return_dict
        )
        return noise_pred

def main():
    print(f"Using device: {DEVICE}")
    print(f"Using dtype: {DTYPE}")

    # 1. Load the SDXL Inpainting Pipeline
    print(f"Loading SDXL Inpainting pipeline from: {MODEL_ID}...")
    try:
        # For SDXL, the inpainting pipeline often uses the base model's components.
        # The pipeline itself is specialized for the inpainting task.
        pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            variant="fp16" if DTYPE == torch.float16 else None, # Use fp16 variant if available and using float16
            use_safetensors=True
        )
        # Using EulerDiscreteScheduler as it's common for SDXL
        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(DEVICE)
        print("Pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        print("This might be due to an incorrect model ID, network issues, or missing model components.")
        print("Ensure the model ID is correct and you have accepted any terms if it's a gated model.")
        return

    unet = UNetWrapper()
    # unet2 = UNetWrapper2(pipeline.unet)

    # Enable model offloading if memory is an issue (slower)
    # pipeline.enable_model_cpu_offload()
    # Enable memory-efficient attention if available (for users with xformers)
    try:
        pipeline.enable_xformers_memory_efficient_attention()
        print("xFormers memory efficient attention enabled.")
    except ImportError:
        print("xFormers not available. Running without memory efficient attention.")
    except Exception as e:
        print(f"Could not enable xFormers: {e}")


    # 2. Load initial image and mask
    # SDXL typically works well with 1024x1024 images
    image_size = (1024, 1024)
    print(f"Loading initial image from: {INIT_IMAGE_PATH}")
    init_image = load_image(INIT_IMAGE_PATH, size=image_size)
    if init_image is None:
        return

    print(f"Loading mask image from: {MASK_IMAGE_PATH}")
    # The mask should be RGB, where white pixels indicate the area to inpaint.
    mask_image = load_image(MASK_IMAGE_PATH, size=image_size)
    if mask_image is None:
        return

    # For some pipelines, the mask might be expected as a single channel (L) and binarized.
    # However, StableDiffusionXLInpaintPipeline often expects an RGB image for the mask as well.
    # Let's ensure it's in the format the pipeline expects.
    # Typically, white areas (255) are inpainted, black areas (0) are preserved.

    # 3. Set up generator for reproducibility
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)

    # 4. Perform inpainting
    print("Starting inpainting process...")
    print(f"  Prompt: {PROMPT}")
    print(f"  Negative Prompt: {NEGATIVE_PROMPT}")
    print(f"  Steps: {NUM_INFERENCE_STEPS}, Guidance Scale: {GUIDANCE_SCALE}, Strength: {STRENGTH}")

    try:
        with torch.no_grad(): # Ensure no gradients are computed
            pipeline.lunet = unet
            # pipeline.lunet = unet2
            inpainted_image = pipeline(
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                image=init_image,
                mask_image=mask_image,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                strength=STRENGTH, # Strength of inpainting. 1.0 means full inpainting in masked area.
                generator=generator,
                # For SDXL, you might need to specify target image size if not resizing inputs
                height=image_size[1],
                width=image_size[0],
                # aesthetic_score=6.0, # Common for SDXL base
                # negative_aesthetic_score=2.5 # Common for SDXL base
            ).images[0]
        print("Inpainting completed.")
    except Exception as e:
        print(f"Error during inpainting: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Save the output image
    try:
        inpainted_image.save(OUTPUT_IMAGE_PATH)
        print(f"Inpainted image saved to: {OUTPUT_IMAGE_PATH}")
        # Display the image if in an environment that supports it (e.g., Jupyter)
        # inpainted_image.show()
    except Exception as e:
        print(f"Error saving image: {e}")


if __name__ == "__main__":
    # Create dummy images if they don't exist for a quick test
    if not os.path.exists(INIT_IMAGE_PATH):
        print(f"Creating dummy init image at {INIT_IMAGE_PATH}")
        dummy_init = Image.new("RGB", (1024, 1024), "lightcoral")
        # Add some features to the dummy image
        from PIL import ImageDraw
        draw = ImageDraw.Draw(dummy_init)
        draw.ellipse((200, 200, 500, 500), fill="lightblue")
        draw.rectangle((600, 600, 800, 800), fill="lightgreen")
        dummy_init.save(INIT_IMAGE_PATH)

    if not os.path.exists(MASK_IMAGE_PATH):
        print(f"Creating dummy mask image at {MASK_IMAGE_PATH}")
        dummy_mask = Image.new("RGB", (1024, 1024), "black") # Mask is also RGB
        # White square in the middle to inpaint
        draw_mask = ImageDraw.Draw(dummy_mask)
        draw_mask.rectangle((300, 300, 700, 700), fill="white")
        dummy_mask.save(MASK_IMAGE_PATH)
        
    main()
