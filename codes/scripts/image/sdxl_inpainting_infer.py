# import os
# import torch
# import tensorrt as trt
# import numpy as np
# from PIL import Image
# from diffusers import (
#     StableDiffusionXLInpaintPipeline,
#     AutoencoderKL,
#     EulerDiscreteScheduler # Or your chosen scheduler
# )
# from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPTextModel
# from transformers.modeling_outputs import BaseModelOutputWithPooling # For type checking
# # import pycuda.driver as cuda
# # import pycuda.autoinit # Important for PyCUDA context

# # --- Configuration (ensure these are defined in your script) ---
# ORIGINAL_MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1" # Or your local path
# UNET_TRT_PATH = "./sdxl_inpainting_onnx_trt/unet/model.plan" 
# VAE_DECODER_TRT_PATH = "./sdxl_vae_decoder_onnx_trt/vae_decoder/model.plan" 

# # Inference parameters
# PROMPT = "A majestic dragon soaring through a nebula, epic, cinematic lighting"
# NEGATIVE_PROMPT = "low quality, blurry, ugly, deformed, watermark, text, signature"
# INIT_IMAGE_PATH = "init_image.png" # Replace with your image path
# MASK_IMAGE_PATH = "mask_image.png" # Replace with your mask path
# OUTPUT_IMAGE_PATH = "sdxl_inpaint_trt_output.png"

# IMAGE_HEIGHT = 512 # Expected output height for VAE optimal case
# IMAGE_WIDTH = 512  # Expected output width for VAE optimal case
# NUM_INFERENCE_STEPS = 30
# GUIDANCE_SCALE = 7.5
# SEED = 42
# INPAINT_STRENGTH = 0.85 # Typical strength for inpainting with noise

# # For SDXL, UNet input channels for inpainting (latents + mask + masked_image_latents)
# INPAINT_UNET_INPUT_CHANNELS = 9
# UNET_OUTPUT_CHANNELS = 4 # Typically 4 for latent noise prediction
# IMAGE_CHANNELS = 3 # RGB output from VAE
# VAE_SCALE_FACTOR = 8 # Defined later from vae_pt.config, but useful to know it's ~8

# DTYPE = torch.float16 # Global dtype

# TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# # --- Helper function to map TRT DataType to Torch DType ---
# def trt_dtype_to_torch_dtype(trt_dtype):
#     if trt_dtype == trt.DataType.FLOAT:
#         return torch.float32
#     elif trt_dtype == trt.DataType.HALF:
#         return torch.float16
#     elif trt_dtype == trt.DataType.INT8:
#         return torch.int8
#     elif trt_dtype == trt.DataType.INT32: # Make sure this is mapped if engine expects int32
#         return torch.int32
#     # Add trt.DataType.INT64 if it exists and your engine uses it, map to torch.int64
#     elif trt_dtype == trt.DataType.INT64:
#         return torch.int64
#     elif trt_dtype == trt.DataType.BOOL:
#         return torch.bool
#     elif trt_dtype == trt.DataType.UINT8:
#         return torch.uint8
#     else:
#         raise TypeError(f"Unsupported TRT DataType: {trt_dtype}")

# class TRTEngine:
#     def __init__(self, engine_path, vae_scale_factor_from_config=8): # Pass vae_scale_factor
#         self.engine_path = engine_path
#         self.engine = None
#         self.context = None
#         self.stream = None
#         self.output_shapes = {}
#         self.input_shapes = {}
#         self.input_names = []
#         self.output_names = []
#         self.input_dtypes_torch = {} 
#         self.output_dtypes_torch = {}
#         self.tensor_name_to_idx_map = {} 
#         self.vae_scale_factor = vae_scale_factor_from_config # Store for fallback shape calculation


#         self._load_engine()
#         self._prepare_binding_info() 

#     def _load_engine(self):
#         print(f"Loading TensorRT engine from: {self.engine_path}")
#         print(f"DEBUG: TensorRT version: {trt.__version__}")
#         with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
#             self.engine = runtime.deserialize_cuda_engine(f.read())
#         if self.engine is None:
#             raise RuntimeError(f"Failed to load TensorRT engine from {self.engine_path}")
        
#         print(f"DEBUG: Type of self.engine: {type(self.engine)}")
#         print(f"DEBUG: Is self.engine an ICudaEngine? {isinstance(self.engine, trt.ICudaEngine)}")

#         self.context = self.engine.create_execution_context()
#         if self.context is None:
#             raise RuntimeError("Failed to create TensorRT execution context.")
#         print("TensorRT engine loaded successfully.")

#     def _prepare_binding_info(self):
#         if not self.engine:
#             raise RuntimeError("TRTEngine._prepare_binding_info called but self.engine is not initialized.")
        
#         print("Preparing TRT engine binding information using I/O tensor API...")
        
#         if not hasattr(self.engine, 'num_io_tensors'):
#             raise AttributeError(f"'ICudaEngine' object (version {trt.__version__}) has no attribute 'num_io_tensors'. This is critical. Please check TensorRT installation.")

#         num_io_tensors = self.engine.num_io_tensors
#         print(f"DEBUG: self.engine.num_io_tensors = {num_io_tensors}")

#         for i in range(num_io_tensors):
#             tensor_name = self.engine.get_tensor_name(i)
#             self.tensor_name_to_idx_map[tensor_name] = i 

#             tensor_shape = tuple(self.engine.get_tensor_shape(tensor_name))
#             tensor_dtype_trt = self.engine.get_tensor_dtype(tensor_name)
            
#             try:
#                 tensor_dtype_torch = trt_dtype_to_torch_dtype(tensor_dtype_trt)
#             except TypeError as e:
#                 print(f"Error for tensor '{tensor_name}': {e}")
#                 print(f"  Problematic TRT DataType was: {tensor_dtype_trt} (enum value: {int(tensor_dtype_trt)})")
#                 raise

#             tensor_mode = self.engine.get_tensor_mode(tensor_name)

#             if tensor_mode == trt.TensorIOMode.INPUT:
#                 self.input_shapes[tensor_name] = tensor_shape
#                 self.input_names.append(tensor_name)
#                 self.input_dtypes_torch[tensor_name] = tensor_dtype_torch
#                 print(f"  Input: {tensor_name} (Index: {i}), Shape: {tensor_shape}, TorchDtype: {tensor_dtype_torch}")
#             elif tensor_mode == trt.TensorIOMode.OUTPUT:
#                 self.output_shapes[tensor_name] = tensor_shape
#                 self.output_names.append(tensor_name)
#                 self.output_dtypes_torch[tensor_name] = tensor_dtype_torch
#                 print(f"  Output: {tensor_name} (Index: {i}), Shape: {tensor_shape}, TorchDtype: {tensor_dtype_torch}")
#             else:
#                 print(f"Warning: Tensor '{tensor_name}' at I/O index {i} has an unexpected mode: {tensor_mode}")
        
#         self.stream = torch.cuda.Stream() 
#         print("TRT binding information prepared.")

#     def __call__(self, inputs_dict: dict):
#         binding_idx_to_ptr = {}

#         for name, tensor in inputs_dict.items():
#             if name not in self.input_names:
#                 print(f"Warning: Input tensor '{name}' provided but not an input for the TRT engine. Skipping.")
#                 continue
            
#             if name not in self.tensor_name_to_idx_map: 
#                 raise ValueError(f"Tensor name '{name}' not found in prepared binding map. This should not happen if _prepare_binding_info was successful.")
#             idx = self.tensor_name_to_idx_map[name]

#             if not tensor.is_cuda:
#                 raise ValueError(f"Input tensor '{name}' must be on CUDA device.")
            
#             expected_torch_dtype = self.input_dtypes_torch[name]
#             if tensor.dtype != expected_torch_dtype:
#                 print(f"Warning: Input tensor '{name}' dtype mismatch. Expected {expected_torch_dtype}, got {tensor.dtype}. Casting.")
#                 tensor = tensor.to(expected_torch_dtype)

#             if -1 in self.engine.get_tensor_shape(name): 
#                 if not self.context.set_input_shape(name, tuple(tensor.shape)):
#                     raise ValueError(f"Failed to set binding shape for dynamic input: {name} to {tensor.shape}")
            
#             binding_idx_to_ptr[idx] = tensor.data_ptr()

#         outputs_torch = {}
#         for name in self.output_names:
#             if name not in self.tensor_name_to_idx_map: 
#                 raise ValueError(f"Tensor name '{name}' not found in prepared binding map for outputs. This should not happen if _prepare_binding_info was successful.")
#             idx = self.tensor_name_to_idx_map[name]

#             dtype_torch = self.output_dtypes_torch[name]
            
#             final_alloc_shape = None
#             try:
#                 output_shape_from_context = tuple(self.context.get_tensor_shape(name))
#                 current_batch_size = list(inputs_dict.values())[0].shape[0]
#                 if current_batch_size != output_shape_from_context[0]:
#                     output_shape_from_context = (current_batch_size, ) + output_shape_from_context[1:]
                
#                 if -1 not in output_shape_from_context:
#                     final_alloc_shape = output_shape_from_context
#                 else:
#                     print(f"Warning: Output tensor '{name}' shape from context.get_binding_shape is still dynamic: {output_shape_from_context}. Will use fallback.")
#             except AttributeError: 
#                 print(f"Warning: context.get_binding_shape not available on this TRT version for output '{name}'. Using fallback.")
#             except Exception as e:
#                 print(f"Error calling context.get_binding_shape for output '{name}': {e}. Using fallback.")

#             if final_alloc_shape is None:
#                 build_time_opt_shape = list(self.output_shapes[name]) 
#                 current_batch_size = list(inputs_dict.values())[0].shape[0] 
                
#                 if build_time_opt_shape[0] == -1 or build_time_opt_shape[0] != current_batch_size :
#                     build_time_opt_shape[0] = current_batch_size
                
                
#                 # More specific fallback for known output tensor names and their expected structures
#                 if name == "sample": # VAE Decoder output
#                     if len(build_time_opt_shape) == 4: 
#                         if build_time_opt_shape[1] == -1 : # Channels
#                             print(f"Info: Output '{name}', dim 1 (Channels) was -1. Setting to global IMAGE_CHANNELS: {IMAGE_CHANNELS}")
#                             build_time_opt_shape[1] = IMAGE_CHANNELS
#                         if build_time_opt_shape[2] == -1: # Height
#                             print(f"Info: Output '{name}', dim 2 (Height) was -1. Setting to global IMAGE_HEIGHT: {IMAGE_HEIGHT}")
#                             build_time_opt_shape[2] = IMAGE_HEIGHT 
#                         if build_time_opt_shape[3] == -1: # Width
#                             print(f"Info: Output '{name}', dim 3 (Width) was -1. Setting to global IMAGE_WIDTH: {IMAGE_WIDTH}")
#                             build_time_opt_shape[3] = IMAGE_WIDTH 
#                 elif name == "out_sample": # UNet output
#                     if len(build_time_opt_shape) == 4:
#                         if build_time_opt_shape[1] == -1: # Channels
#                             print(f"Info: Output '{name}', dim 1 (Channels) was -1. Setting to global UNET_OUTPUT_CHANNELS: {UNET_OUTPUT_CHANNELS}")
#                             build_time_opt_shape[1] = UNET_OUTPUT_CHANNELS
#                         if build_time_opt_shape[2] == -1: # Latent Height
#                             latent_height = IMAGE_HEIGHT // self.vae_scale_factor
#                             print(f"Info: Output '{name}', dim 2 (Latent Height) was -1. Setting to derived latent_height: {latent_height}")
#                             build_time_opt_shape[2] = latent_height
#                         if build_time_opt_shape[3] == -1: # Latent Width
#                             latent_width = IMAGE_WIDTH // self.vae_scale_factor
#                             print(f"Info: Output '{name}', dim 3 (Latent Width) was -1. Setting to derived latent_width: {latent_width}")
#                             build_time_opt_shape[3] = latent_width
#                 else: # General fallback for other outputs
#                     for i_dim in range(1, len(build_time_opt_shape)): 
#                         if build_time_opt_shape[i_dim] == -1:
#                             print(f"CRITICAL WARNING: Dimension {i_dim} for output '{name}' is -1 in build_time_opt_shape and unresolved. Using 1 as placeholder. THIS IS RISKY.")
#                             build_time_opt_shape[i_dim] = 1
                
#                 final_alloc_shape = tuple(build_time_opt_shape)
#                 print(f"DEBUG: Using fallback allocation shape for '{name}': {final_alloc_shape}")
            
#             outputs_torch[name] = torch.empty(size=final_alloc_shape, dtype=dtype_torch, device="cuda")
#             binding_idx_to_ptr[idx] = outputs_torch[name].data_ptr()

#         final_bindings_list = [0] * self.engine.num_io_tensors 
#         for original_io_index, ptr in binding_idx_to_ptr.items(): 
#             if original_io_index >= len(final_bindings_list): 
#                 raise IndexError(f"Binding index {original_io_index} (derived from I/O index) is out of range for bindings list of size {len(final_bindings_list)}.")
#             final_bindings_list[original_io_index] = ptr
        
#         for i in range(len(final_bindings_list)):
#             if final_bindings_list[i] == 0: 
#                 missing_tensor_name = None
#                 for name, mapped_idx in self.tensor_name_to_idx_map.items(): 
#                     if mapped_idx == i:
#                         missing_tensor_name = name
#                         break
#                 raise RuntimeError(f"Binding for index {i} (tensor: {missing_tensor_name or 'Unknown'}) was not set. Ensure all engine inputs/outputs are handled.")

#         stream_handle = self.stream.cuda_stream 
        
#         executed_successfully = False
        
#         if hasattr(self.context, "execute_async_v3"):
#             try:
#                 for i, ptr in enumerate(final_bindings_list):
#                     if hasattr(self.context, "set_binding_address"):
#                         if not self.context.set_binding_address(i, ptr):
#                             raise RuntimeError(f"Failed to set binding address for binding index {i} using set_binding_address.")
#                     elif hasattr(self.context, "set_tensor_address"): 
#                         tensor_name_for_idx_i = None
#                         for t_name, t_idx in self.tensor_name_to_idx_map.items():
#                             if t_idx == i:
#                                 tensor_name_for_idx_i = t_name
#                                 break
#                         if tensor_name_for_idx_i is None:
#                             raise RuntimeError(f"Could not find tensor name for binding index {i} to use with set_tensor_address.")
#                         if not self.context.set_tensor_address(tensor_name_for_idx_i, ptr):
#                              raise RuntimeError(f"Failed to set tensor address for tensor '{tensor_name_for_idx_i}' (binding index {i}) using set_tensor_address.")
#                     else:
#                         raise AttributeError("IExecutionContext has execute_async_v3 but is missing set_binding_address and set_tensor_address. Cannot set bindings for v3.")

#                 if not self.context.execute_async_v3(stream_handle=stream_handle): 
#                     raise RuntimeError("execute_async_v3 call failed to enqueue.")
#                 executed_successfully = True
#             except TypeError as e_v3_type: 
#                  if "incompatible function arguments" in str(e_v3_type) and "bindings" in str(e_v3_type):
#                     print(f"DEBUG: execute_async_v3 was called with 'bindings' kwarg due to previous logic, but signature is (self, stream_handle). Error: {e_v3_type}")
#                  else:
#                     print(f"DEBUG: execute_async_v3 path failed with TypeError: {e_v3_type}")
#             except Exception as e_v3:
#                 print(f"DEBUG: execute_async_v3 path failed: {e_v3}")

#         if not executed_successfully and hasattr(self.context, "execute_async_v2"):
#             try:
#                 self.context.execute_async_v2(bindings=final_bindings_list, stream_handle=stream_handle)
#                 executed_successfully = True
#             except AttributeError as e_v2: 
#                 if "execute_async_v3" in str(e_v2): 
#                     print(f"DEBUG: execute_async_v2 failed suggesting v3 ({e_v2}), but v3 path was already tried or failed. This is a problematic state.")
#                 else: 
#                     print(f"DEBUG: execute_async_v2 failed for a different reason: {e_v2}")
#             except Exception as e_v2_other:
#                  print(f"DEBUG: execute_async_v2 failed with a non-AttributeError: {e_v2_other}")


#         if not executed_successfully and hasattr(self.context, "execute_async"):
#             try:
#                 current_batch_size = list(inputs_dict.values())[0].shape[0]
#                 self.context.execute_async(batch_size=current_batch_size,
#                                            bindings=final_bindings_list,
#                                            stream_handle=stream_handle)
#                 executed_successfully = True
#             except Exception as e_base:
#                 print(f"DEBUG: execute_async (base) failed: {e_base}")

#         if not executed_successfully:
#             raise AttributeError("No suitable execute_async method (v3, v2, or base) was found or executed successfully on IExecutionContext.")

#         self.stream.synchronize()
#         return outputs_torch

# # --- Main Inference Script's encode_text function (ensure this is part of your script) ---
# def encode_text(prompt_text, tokenizer, text_encoder_model):
#     text_inputs = tokenizer(
#         prompt_text, padding="max_length", max_length=tokenizer.model_max_length,
#         truncation=True, return_tensors="pt"
#     )
#     with torch.no_grad():
#         prompt_embeds = text_encoder_model(text_inputs.input_ids.to(text_encoder_model.device)) 

#     if not hasattr(prompt_embeds, 'hidden_states') or prompt_embeds.hidden_states is None:
#         print("Warning: 'hidden_states' not found in text encoder output. Using 'last_hidden_state' for sequence embeddings.")
#         seq_embeds = prompt_embeds.last_hidden_state
#     else:
#         seq_embeds = prompt_embeds.hidden_states[-2]

#     if isinstance(text_encoder_model, CLIPTextModelWithProjection):
#         if hasattr(prompt_embeds, 'text_embeds') and prompt_embeds.text_embeds is not None:
#             pooled_embeds = prompt_embeds.text_embeds 
#         elif hasattr(prompt_embeds, 'pooler_output') and prompt_embeds.pooler_output is not None:
#             print("Warning: 'text_embeds' (projected) not found on CLIPTextModelWithProjection output, using 'pooler_output'.")
#             pooled_embeds = prompt_embeds.pooler_output
#         else:
#             print("Warning: Neither 'text_embeds' (projected) nor 'pooler_output' found on CLIPTextModelWithProjection output. Using last_hidden_state[:, 0].")
#             pooled_embeds = prompt_embeds.last_hidden_state[:, 0] 
#     elif isinstance(text_encoder_model, CLIPTextModel):
#         if hasattr(prompt_embeds, 'pooler_output') and prompt_embeds.pooler_output is not None:
#             pooled_embeds = prompt_embeds.pooler_output
#         else:
#             print("Warning: 'pooler_output' not found on CLIPTextModel output. Using last_hidden_state[:, 0].")
#             pooled_embeds = prompt_embeds.last_hidden_state[:, 0] 
#     else:
#         print(f"Warning: Unknown text_encoder_model type ({type(text_encoder_model)}). Attempting to use 'pooler_output' or fallback.")
#         if hasattr(prompt_embeds, 'pooler_output') and prompt_embeds.pooler_output is not None:
#             pooled_embeds = prompt_embeds.pooler_output
#         elif hasattr(prompt_embeds, 'last_hidden_state'):
#             pooled_embeds = prompt_embeds.last_hidden_state[:, 0]
#         else:
#             raise AttributeError(f"Cannot determine pooled embeddings for text_encoder_model of type {type(text_encoder_model)}")

#     return seq_embeds, pooled_embeds

# # --- Main Inference Script ---
# if __name__ == "__main__":
#     torch.manual_seed(SEED)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(SEED)
#     np.random.seed(SEED)

#     print("Loading original Diffusers pipeline components...")
#     tokenizer_1 = CLIPTokenizer.from_pretrained(ORIGINAL_MODEL_ID, subfolder="tokenizer", torch_dtype=DTYPE)
#     tokenizer_2 = CLIPTokenizer.from_pretrained(ORIGINAL_MODEL_ID, subfolder="tokenizer_2", torch_dtype=DTYPE)
#     text_encoder_1 = CLIPTextModel.from_pretrained(ORIGINAL_MODEL_ID, subfolder="text_encoder", torch_dtype=DTYPE, use_safetensors=True).cuda().eval()
#     text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(ORIGINAL_MODEL_ID, subfolder="text_encoder_2", torch_dtype=DTYPE, use_safetensors=True).cuda().eval()
#     vae_pt = AutoencoderKL.from_pretrained(ORIGINAL_MODEL_ID, subfolder="vae", torch_dtype=DTYPE, use_safetensors=True).cuda().eval()
#     vae_scale_factor = 2 ** (len(vae_pt.config.block_out_channels) - 1) # Now global VAE_SCALE_FACTOR can be derived
#     scheduler = EulerDiscreteScheduler.from_pretrained(ORIGINAL_MODEL_ID, subfolder="scheduler")
#     print("Original pipeline components loaded.")

#     print("Initializing TensorRT engines...")
#     # Pass vae_scale_factor to TRTEngine instances if they need it for shape calculations
#     unet_trt = TRTEngine(UNET_TRT_PATH, vae_scale_factor_from_config=vae_scale_factor)
#     vae_decoder_trt = TRTEngine(VAE_DECODER_TRT_PATH, vae_scale_factor_from_config=vae_scale_factor)
#     print("TensorRT engines initialized.")

#     print("Preparing inputs...")
#     try:
#         init_image_pil = Image.open(INIT_IMAGE_PATH).convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT))
#         mask_image_pil = Image.open(MASK_IMAGE_PATH).convert("L").resize((IMAGE_WIDTH, IMAGE_HEIGHT))
#     except FileNotFoundError:
#         print(f"Error: Input image or mask not found. Creating dummy images.")
#         init_image_pil = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), "white")
#         mask_array = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
#         mask_array[IMAGE_HEIGHT//4:3*IMAGE_HEIGHT//4, IMAGE_WIDTH//4:3*IMAGE_WIDTH//4] = 255 # Mask a central square
#         mask_image_pil = Image.fromarray(mask_array, mode="L")


#     def _preprocess_image(image: Image.Image):
#         image_np = np.array(image).astype(np.float32) / 255.0
#         image_pt = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
#         return (2.0 * image_pt - 1.0).to(device="cuda", dtype=DTYPE)

#     def _preprocess_mask(mask: Image.Image):
#         mask_np = np.array(mask).astype(np.float32) / 255.0
#         mask_pt = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
#         return (mask_pt > 0.5).to(device="cuda", dtype=DTYPE) # Binarize to True/False then cast to float

#     init_image_pt = _preprocess_image(init_image_pil)
#     mask_pt = _preprocess_mask(mask_image_pil)

#     print("Encoding initial image to latents...")
#     with torch.no_grad():
#         init_latents_dist = vae_pt.encode(init_image_pt).latent_dist
#         init_latents = init_latents_dist.sample(generator=torch.Generator(device='cuda').manual_seed(SEED)) # Ensure generator is on CUDA
#     init_latents = init_latents * vae_pt.config.scaling_factor
#     print(f"Initial latents shape: {init_latents.shape}")

#     mask_latents = torch.nn.functional.interpolate(
#         mask_pt, size=(init_latents.shape[2], init_latents.shape[3])
#     )
#     masked_image_latents = init_latents * (1 - mask_latents)
#     print(f"Mask latents shape: {mask_latents.shape}")

#     print("Encoding text prompts...")
#     prompt_embeds_t1_seq, prompt_embeds_t1_pool = encode_text(PROMPT, tokenizer_1, text_encoder_1)
#     prompt_embeds_t2_seq, prompt_embeds_t2_pool = encode_text(PROMPT, tokenizer_2, text_encoder_2)
    
#     encoder_hidden_states = prompt_embeds_t2_seq 
#     add_text_embeds = prompt_embeds_t2_pool # Typically the projected output from text_encoder_2

#     if GUIDANCE_SCALE > 1.0:
#         uncond_embeds_t1_seq, uncond_embeds_t1_pool = encode_text(NEGATIVE_PROMPT, tokenizer_1, text_encoder_1)
#         uncond_embeds_t2_seq, uncond_embeds_t2_pool = encode_text(NEGATIVE_PROMPT, tokenizer_2, text_encoder_2)
        
#         uncond_encoder_hidden_states = uncond_embeds_t2_seq
#         uncond_add_text_embeds = uncond_embeds_t2_pool

#         encoder_hidden_states = torch.cat([uncond_encoder_hidden_states, encoder_hidden_states])
#         add_text_embeds = torch.cat([uncond_add_text_embeds, add_text_embeds])

#     original_size = (IMAGE_HEIGHT, IMAGE_WIDTH)
#     crops_coords_top_left = (0, 0)
#     target_size = (IMAGE_HEIGHT, IMAGE_WIDTH)
#     add_time_ids_elements = list(original_size + crops_coords_top_left + target_size)
#     add_time_ids = torch.tensor([add_time_ids_elements], dtype=DTYPE, device="cuda")

#     if GUIDANCE_SCALE > 1.0:
#         add_time_ids = torch.cat([add_time_ids] * 2)
    
#     added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
#     print(f"Shape of final encoder_hidden_states for UNet: {encoder_hidden_states.shape}")
#     print(f"Shape of final add_text_embeds for UNet: {add_text_embeds.shape}")
#     print(f"Shape of final add_time_ids for UNet: {add_time_ids.shape}")

#     print("Starting denoising loop...")
#     scheduler.set_timesteps(NUM_INFERENCE_STEPS, device="cuda")
#     timesteps = scheduler.timesteps

#     init_timestep_index = int(NUM_INFERENCE_STEPS * (1 - INPAINT_STRENGTH)) if INPAINT_STRENGTH < 1.0 else 0
#     init_timestep_index = max(0, min(init_timestep_index, NUM_INFERENCE_STEPS -1)) # Clamp index
#     init_timestep = timesteps[init_timestep_index]


#     noise = torch.randn_like(init_latents) #, generator=torch.Generator(device='cuda').manual_seed(SEED)) # Ensure generator is on CUDA
#     latents = scheduler.add_noise(init_latents, noise, init_timestep.unsqueeze(0))
#     print(f"Initial noisy latents shape (after add_noise): {latents.shape}")

#     t_start_idx_numpy = np.where(timesteps.cpu().numpy() == init_timestep.cpu().numpy())[0]
#     if t_start_idx_numpy.size == 0:
#         print(f"Warning: init_timestep {init_timestep.item()} not found in scheduler.timesteps. Starting from the beginning of timesteps.")
#         t_start = 0
#     else:
#         t_start = t_start_idx_numpy[0]
        
#     timesteps_to_iterate = timesteps[t_start:]


#     for i, t in enumerate(timesteps_to_iterate):
#         print(f"  Step {i+1}/{len(timesteps_to_iterate)}, Timestep: {t.item()}")

#         latent_model_input_unscaled = torch.cat([latents, mask_latents, masked_image_latents], dim=1)
        
#         if GUIDANCE_SCALE > 1.0:
#             unet_input_sample_unscaled = torch.cat([latent_model_input_unscaled] * 2)
#         else:
#             unet_input_sample_unscaled = latent_model_input_unscaled

#         unet_input_sample = scheduler.scale_model_input(unet_input_sample_unscaled, t)
        
#         current_batch_size_for_unet = unet_input_sample.shape[0]
#         # Timestep needs to be a tensor for UNet, but its dtype for TRT might be int64 or int32
#         # Check the dtype expected by your TRT UNet for 'timestep' input
#         # Assuming it's int64 based on typical PyTorch UNet and previous export attempts
#         timestep_pt = torch.tensor([t.item()] * current_batch_size_for_unet, device="cuda", dtype=torch.long) 

#         unet_inputs_trt = {
#             "sample": unet_input_sample.to(DTYPE),
#             "timestep": timestep_pt, # Ensure this dtype matches TRT engine's expectation
#             "encoder_hidden_states": encoder_hidden_states.to(DTYPE),
#             "text_embeds": added_cond_kwargs["text_embeds"].to(DTYPE),
#             "time_ids": added_cond_kwargs["time_ids"].to(DTYPE)
#         }
        
#         unet_outputs_trt = unet_trt(unet_inputs_trt)
#         noise_pred = unet_outputs_trt["out_sample"]

#         if GUIDANCE_SCALE > 1.0:
#             noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
#             noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)

#         latents = scheduler.step(noise_pred, t, latents).prev_sample
#         if torch.isnan(latents).any():
#             import pdb; pdb.set_trace()


#     print("Denoising loop finished.")

#     print("Decoding latents using VAE Decoder TRT...")
#     latents_to_decode = latents / vae_pt.config.scaling_factor

#     vae_decoder_inputs_trt = {"latent_sample": latents_to_decode.to(DTYPE)} 
    
#     decoded_image_trt = vae_decoder_trt(vae_decoder_inputs_trt)["sample"]

#     decoded_image_trt = (decoded_image_trt / 2 + 0.5).clamp(0, 1)
#     image_np = decoded_image_trt.squeeze(0).permute(1, 2, 0).cpu().float().numpy() 
#     image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

#     image_pil.save(OUTPUT_IMAGE_PATH)
#     print(f"Inpainted image saved to {OUTPUT_IMAGE_PATH}")

from typing import List, Tuple, Dict, Any, Optional, Union
import os
import torch
import tensorrt as trt
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionXLInpaintPipeline,
    AutoencoderKL,
    LMSDiscreteScheduler, # Or your chosen scheduler
    EulerDiscreteScheduler,
    PNDMScheduler,
)
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPTextModel
from transformers.modeling_outputs import BaseModelOutputWithPooling # For type checking
# import pycuda.driver as cuda # Removed
# import pycuda.autoinit # Removed

# --- Configuration (ensure these are defined in your script) ---
ORIGINAL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0" # Or your local path
UNET_TRT_PATH = "./sdxl_inpainting_onnx_trt/unet/model.plan" 
VAE_DECODER_TRT_PATH = "./sdxl_vae_decoder_onnx_trt/vae_decoder/model.plan" 

# Inference parameters
PROMPT = "A futuristic robot standing in the room, photorealistic, 4k"
NEGATIVE_PROMPT = "blurry, low quality, unrealistic, watermark, signature, text, ugly, deformed"
INIT_IMAGE_PATH = "/workspace/repo-dev/DL-Art-School-dev/codes/scripts/image/overture-creations-5sI6fQgYIuo.png" # Replace with your image path
MASK_IMAGE_PATH = "/workspace/repo-dev/DL-Art-School-dev/codes/scripts/image/overture-creations-5sI6fQgYIuo_mask.png" # Replace with your mask path
OUTPUT_IMAGE_PATH = "sdxl_inpaint_trt_output.png"

IMAGE_HEIGHT = 512 # Expected output height for VAE optimal case
IMAGE_WIDTH = 512  # Expected output width for VAE optimal case
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 1.5
SEED = 42
INPAINT_STRENGTH = 0.85 # Typical strength for inpainting with noise

# For SDXL, UNet input channels for inpainting (latents + mask + masked_image_latents)
INPAINT_UNET_INPUT_CHANNELS = 9
UNET_OUTPUT_CHANNELS = 4 # Typically 4 for latent noise prediction
IMAGE_CHANNELS = 3 # RGB output from VAE
VAE_SCALE_FACTOR = 8 # Defined later from vae_pt.config, but useful to know it's ~8

DTYPE = torch.float16 # Global dtype

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# --- NaN/Inf Checking Function ---
def check_tensor(tensor, name="Tensor", raise_error=False, step_info=""):
    if tensor is None:
        print(f"!!! {name} is None {step_info} !!!")
        if raise_error: raise ValueError(f"{name} is None {step_info}")
        return True # Treat None as an issue
    if not isinstance(tensor, torch.Tensor):
        print(f"!!! {name} is not a Tensor (type: {type(tensor)}) {step_info} !!!")
        if raise_error: raise ValueError(f"{name} is not a Tensor {step_info}")
        return True # Treat non-tensor as an issue

    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    
    if has_nan or has_inf:
        if has_nan:
            print(f"!!! NaN DETECTED IN {name} {step_info} !!!")
        if has_inf:
            print(f"!!! Inf DETECTED IN {name} {step_info} !!!")
        
        print(f"  Details for {name}: Shape: {tensor.shape}, Dtype: {tensor.dtype}, Device: {tensor.device}")
        # print(f"  Min: {tensor.min().item() if tensor.numel() > 0 else 'N/A'}, Max: {tensor.max().item() if tensor.numel() > 0 else 'N/A'}, Mean: {tensor.mean().item() if tensor.numel() > 0 else 'N/A'}")
        if raise_error:
            if has_nan and has_inf: raise ValueError(f"NaN and Inf detected in {name} {step_info}")
            if has_nan: raise ValueError(f"NaN detected in {name} {step_info}")
            if has_inf: raise ValueError(f"Inf detected in {name} {step_info}")
        return True
    # print(f"DEBUG {name}{step_info}: min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}, dtype={tensor.dtype}")
    return False

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
        
        if hasattr(self.context, "execute_async_v3"):
            try:
                for i, ptr in enumerate(final_bindings_list):
                    if hasattr(self.context, "set_binding_address"):
                        if not self.context.set_binding_address(i, ptr):
                            raise RuntimeError(f"Failed to set binding address for binding index {i} using set_binding_address.")
                    elif hasattr(self.context, "set_tensor_address"): 
                        tensor_name_for_idx_i = None
                        for t_name, t_idx in self.tensor_name_to_idx_map.items():
                            if t_idx == i:
                                tensor_name_for_idx_i = t_name
                                break
                        if tensor_name_for_idx_i is None:
                            raise RuntimeError(f"Could not find tensor name for binding index {i} to use with set_tensor_address.")
                        if not self.context.set_tensor_address(tensor_name_for_idx_i, ptr):
                             raise RuntimeError(f"Failed to set tensor address for tensor '{tensor_name_for_idx_i}' (binding index {i}) using set_tensor_address.")
                    else:
                        raise AttributeError("IExecutionContext has execute_async_v3 but is missing set_binding_address and set_tensor_address. Cannot set bindings for v3.")

                if not self.context.execute_async_v3(stream_handle=stream_handle): 
                    raise RuntimeError("execute_async_v3 call failed to enqueue.")
                executed_successfully = True
            except TypeError as e_v3_type: 
                 if "incompatible function arguments" in str(e_v3_type) and "bindings" in str(e_v3_type):
                    print(f"DEBUG: execute_async_v3 was called with 'bindings' kwarg due to previous logic, but signature is (self, stream_handle). Error: {e_v3_type}")
                 else:
                    print(f"DEBUG: execute_async_v3 path failed with TypeError: {e_v3_type}")
            except Exception as e_v3:
                print(f"DEBUG: execute_async_v3 path failed: {e_v3}")

        if not executed_successfully and hasattr(self.context, "execute_async_v2"):
            try:
                self.context.execute_async_v2(bindings=final_bindings_list, stream_handle=stream_handle)
                executed_successfully = True
            except AttributeError as e_v2: 
                if "execute_async_v3" in str(e_v2): 
                    print(f"DEBUG: execute_async_v2 failed suggesting v3 ({e_v2}), but v3 path was already tried or failed. This is a problematic state.")
                else: 
                    print(f"DEBUG: execute_async_v2 failed for a different reason: {e_v2}")
            except Exception as e_v2_other:
                 print(f"DEBUG: execute_async_v2 failed with a non-AttributeError: {e_v2_other}")


        if not executed_successfully and hasattr(self.context, "execute_async"):
            try:
                current_batch_size = list(inputs_dict.values())[0].shape[0]
                self.context.execute_async(batch_size=current_batch_size,
                                           bindings=final_bindings_list,
                                           stream_handle=stream_handle)
                executed_successfully = True
            except Exception as e_base:
                print(f"DEBUG: execute_async (base) failed: {e_base}")

        if not executed_successfully:
            raise AttributeError("No suitable execute_async method (v3, v2, or base) was found or executed successfully on IExecutionContext.")

        self.stream.synchronize()
        return outputs_torch


# --- Main Inference Script's encode_text function (ensure this is part of your script) ---
def encode_text(prompt_text, tokenizer, text_encoder_model):
    text_inputs = tokenizer(
        prompt_text, padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        prompt_embeds = text_encoder_model(text_inputs.input_ids.to(text_encoder_model.device)) 
        check_tensor(prompt_embeds.last_hidden_state, f"TextEncoder Output last_hidden_state for '{prompt_text[:20]}...'")


    if not hasattr(prompt_embeds, 'hidden_states') or prompt_embeds.hidden_states is None:
        print("Warning: 'hidden_states' not found in text encoder output. Using 'last_hidden_state' for sequence embeddings.")
        seq_embeds = prompt_embeds.last_hidden_state
    else:
        seq_embeds = prompt_embeds.hidden_states[-1]
    check_tensor(seq_embeds, f"seq_embeds for '{prompt_text[:20]}...'")


    if isinstance(text_encoder_model, CLIPTextModelWithProjection):
        if hasattr(prompt_embeds, 'text_embeds') and prompt_embeds.text_embeds is not None:
            pooled_embeds = prompt_embeds.text_embeds 
        elif hasattr(prompt_embeds, 'pooler_output') and prompt_embeds.pooler_output is not None:
            print("Warning: 'text_embeds' (projected) not found on CLIPTextModelWithProjection output, using 'pooler_output'.")
            pooled_embeds = prompt_embeds.pooler_output
        else:
            print("Warning: Neither 'text_embeds' (projected) nor 'pooler_output' found on CLIPTextModelWithProjection output. Using last_hidden_state[:, 0].")
            pooled_embeds = prompt_embeds.last_hidden_state[:, 0] 
    elif isinstance(text_encoder_model, CLIPTextModel):
        if hasattr(prompt_embeds, 'pooler_output') and prompt_embeds.pooler_output is not None:
            pooled_embeds = prompt_embeds.pooler_output
        else:
            print("Warning: 'pooler_output' not found on CLIPTextModel output. Using last_hidden_state[:, 0].")
            pooled_embeds = prompt_embeds.last_hidden_state[:, 0] 
    else:
        print(f"Warning: Unknown text_encoder_model type ({type(text_encoder_model)}). Attempting to use 'pooler_output' or fallback.")
        if hasattr(prompt_embeds, 'pooler_output') and prompt_embeds.pooler_output is not None:
            pooled_embeds = prompt_embeds.pooler_output
        elif hasattr(prompt_embeds, 'last_hidden_state'):
            pooled_embeds = prompt_embeds.last_hidden_state[:, 0]
        else:
            raise AttributeError(f"Cannot determine pooled embeddings for text_encoder_model of type {type(text_encoder_model)}")
    
    check_tensor(pooled_embeds, f"pooled_embeds for '{prompt_text[:20]}...'")
    return seq_embeds, pooled_embeds



class UNetWrapper():
    def __init__(self, unet):
        self.unet = TRTEngine(UNET_TRT_PATH, vae_scale_factor_from_config=vae_scale_factor)

    def forward(self,
                latent_model_input,
                t,
                encoder_hidden_states,
                timestep_cond,
                cross_attention_kwargs,
                added_cond_kwargs,
                return_dict=False):
        current_batch_size_for_unet = latent_model_input.shape[0]
        timestep_pt = torch.tensor([t.item()] * current_batch_size_for_unet, device="cuda", dtype=torch.long) 

        unet_inputs_trt = {
            "sample": latent_model_input.to(DTYPE),
            "timestep": timestep_pt, 
            "encoder_hidden_states": encoder_hidden_states.to(DTYPE),
            "text_embeds": added_cond_kwargs["text_embeds"].to(DTYPE),
            "time_ids": added_cond_kwargs["time_ids"].to(DTYPE)
        }
        unet_outputs_trt = self.unet(unet_inputs_trt)
        noise_pred = unet_outputs_trt["out_sample"]

        return noise_pred


# --- Main Inference Script ---
if __name__ == "__main__":
    # Ensure CUDA context is initialized for PyTorch if not using pycuda.autoinit
    if torch.cuda.is_available():
        torch.cuda.init() # Explicitly initialize PyTorch CUDA context if needed

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    print("Loading original Diffusers pipeline components...")
    tokenizer_1 = CLIPTokenizer.from_pretrained(ORIGINAL_MODEL_ID, subfolder="tokenizer", torch_dtype=DTYPE)
    tokenizer_2 = CLIPTokenizer.from_pretrained(ORIGINAL_MODEL_ID, subfolder="tokenizer_2", torch_dtype=DTYPE)
    text_encoder_1 = CLIPTextModel.from_pretrained(ORIGINAL_MODEL_ID, subfolder="text_encoder", torch_dtype=DTYPE, use_safetensors=True).cuda().eval()
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(ORIGINAL_MODEL_ID, subfolder="text_encoder_2", torch_dtype=DTYPE, use_safetensors=True).cuda().eval()
    vae_pt = AutoencoderKL.from_pretrained(ORIGINAL_MODEL_ID, subfolder="vae", torch_dtype=DTYPE, use_safetensors=True).cuda().eval()
    vae_scale_factor = 2 ** (len(vae_pt.config.block_out_channels) - 1) 
    scheduler = EulerDiscreteScheduler.from_pretrained(ORIGINAL_MODEL_ID, subfolder="scheduler")
    print("Original pipeline components loaded.")

    print("Initializing TensorRT engines...")
    unet_trt = TRTEngine(UNET_TRT_PATH, vae_scale_factor_from_config=vae_scale_factor)
    vae_decoder_trt = TRTEngine(VAE_DECODER_TRT_PATH, vae_scale_factor_from_config=vae_scale_factor)
    print("TensorRT engines initialized.")

    print("Preparing inputs...")
    try:
        init_image_pil = Image.open(INIT_IMAGE_PATH).convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        mask_image_pil = Image.open(MASK_IMAGE_PATH).convert("L").resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    except FileNotFoundError:
        print(f"Error: Input image or mask not found. Creating dummy images.")
        init_image_pil = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), "white")
        mask_array = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
        mask_array[IMAGE_HEIGHT//4:3*IMAGE_HEIGHT//4, IMAGE_WIDTH//4:3*IMAGE_WIDTH//4] = 255 
        mask_image_pil = Image.fromarray(mask_array, mode="L")


    def _preprocess_image(image: Image.Image):
        image_np = np.array(image).astype(np.float32) / 255.0
        image_pt = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
        return (2.0 * image_pt - 1.0).to(device="cuda", dtype=DTYPE)

    def _preprocess_mask(mask: Image.Image):
        mask_np = np.array(mask).astype(np.float32) / 255.0
        mask_pt = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
        return (mask_pt > 0.5).to(device="cuda", dtype=DTYPE) 

    init_image_pt = _preprocess_image(init_image_pil)
    mask_pt = _preprocess_mask(mask_image_pil)
    check_tensor(init_image_pt, "init_image_pt")
    check_tensor(mask_pt, "mask_pt")

    print("Encoding initial image to latents...")
    with torch.no_grad():
        generator_vae_encode = torch.Generator(device=init_image_pt.device).manual_seed(SEED)
        init_latents_dist = vae_pt.encode(init_image_pt).latent_dist
        init_latents = init_latents_dist.sample(generator=generator_vae_encode)
    init_latents = init_latents * vae_pt.config.scaling_factor
    check_tensor(init_latents, "init_latents after VAE encode")
    print(f"Initial latents shape: {init_latents.shape}")

    mask_latents = torch.nn.functional.interpolate(
        mask_pt, size=(init_latents.shape[2], init_latents.shape[3])
    )
    masked_image_latents = init_latents * (1 - mask_latents)
    check_tensor(mask_latents, "mask_latents")
    check_tensor(masked_image_latents, "masked_image_latents")
    print(f"Mask latents shape: {mask_latents.shape}")

    print("Encoding text prompts...")
    prompt_embeds_t1_seq, prompt_embeds_t1_pool = encode_text(PROMPT, tokenizer_1, text_encoder_1)
    prompt_embeds_t2_seq, prompt_embeds_t2_pool = encode_text(PROMPT, tokenizer_2, text_encoder_2)
    import pdb; pdb.set_trace()
    
    encoder_hidden_states = prompt_embeds_t2_seq 
    add_text_embeds = prompt_embeds_t2_pool 

    if GUIDANCE_SCALE > 1.0:
        uncond_embeds_t1_seq, uncond_embeds_t1_pool = encode_text(NEGATIVE_PROMPT, tokenizer_1, text_encoder_1)
        uncond_embeds_t2_seq, uncond_embeds_t2_pool = encode_text(NEGATIVE_PROMPT, tokenizer_2, text_encoder_2)
        
        uncond_encoder_hidden_states = uncond_embeds_t2_seq
        uncond_add_text_embeds = uncond_embeds_t2_pool

        encoder_hidden_states = torch.cat([uncond_encoder_hidden_states, encoder_hidden_states])
        add_text_embeds = torch.cat([uncond_add_text_embeds, add_text_embeds])
    
    check_tensor(encoder_hidden_states, "final encoder_hidden_states")
    check_tensor(add_text_embeds, "final add_text_embeds")

    original_size = (IMAGE_HEIGHT, IMAGE_WIDTH)
    crops_coords_top_left = (0, 0)
    target_size = (IMAGE_HEIGHT, IMAGE_WIDTH)
    add_time_ids_elements = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids_elements], dtype=DTYPE, device="cuda")

    if GUIDANCE_SCALE > 1.0:
        add_time_ids = torch.cat([add_time_ids] * 2)
    
    check_tensor(add_time_ids, "final add_time_ids")
    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
    print(f"Shape of final encoder_hidden_states for UNet: {encoder_hidden_states.shape}")
    print(f"Shape of final add_text_embeds for UNet: {add_text_embeds.shape}")
    print(f"Shape of final add_time_ids for UNet: {add_time_ids.shape}")

    print("Starting denoising loop...")
    scheduler.set_timesteps(NUM_INFERENCE_STEPS, device="cuda")
    timesteps = scheduler.timesteps

    init_timestep_index = int(NUM_INFERENCE_STEPS * (1 - INPAINT_STRENGTH)) if INPAINT_STRENGTH < 1.0 else 0
    init_timestep_index = max(0, min(init_timestep_index, NUM_INFERENCE_STEPS -1)) 
    init_timestep = timesteps[init_timestep_index]

    noise_generator = torch.Generator(device=init_latents.device).manual_seed(SEED)
    noise = torch.randn_like(init_latents) #, generator=noise_generator) 
    check_tensor(noise, "initial noise")
    
    latents = scheduler.add_noise(init_latents, noise, init_timestep.unsqueeze(0))
    check_tensor(latents, f"latents after add_noise (t={init_timestep.item()})", raise_error=True)
    print(f"Initial noisy latents shape (after add_noise): {latents.shape}")

    t_start_idx_numpy = np.where(timesteps.cpu().numpy() == init_timestep.cpu().numpy())[0]
    if t_start_idx_numpy.size == 0:
        print(f"Warning: init_timestep {init_timestep.item()} not found in scheduler.timesteps. Starting from the beginning of timesteps.")
        t_start = 0
    else:
        t_start = t_start_idx_numpy[0]
        
    timesteps_to_iterate = timesteps[t_start:]

    for i, t in enumerate(timesteps_to_iterate):
        current_step_info = f"at step {i+1}/{len(timesteps_to_iterate)} (t={t.item()})"
        print(f"  {current_step_info}")
        check_tensor(latents, "latents at start of loop", step_info=current_step_info, raise_error=True)


        latent_model_input_unscaled = torch.cat([latents, mask_latents, masked_image_latents], dim=1)
        check_tensor(latent_model_input_unscaled, "latent_model_input_unscaled", step_info=current_step_info, raise_error=True)

        
        if GUIDANCE_SCALE > 1.0:
            unet_input_sample_unscaled = torch.cat([latent_model_input_unscaled] * 2)
        else:
            unet_input_sample_unscaled = latent_model_input_unscaled
        
        unet_input_sample = scheduler.scale_model_input(unet_input_sample_unscaled, t)
        check_tensor(unet_input_sample, "unet_input_sample (scaled)", step_info=current_step_info, raise_error=True)

        
        current_batch_size_for_unet = unet_input_sample.shape[0]
        timestep_pt = torch.tensor([t.item()] * current_batch_size_for_unet, device="cuda", dtype=torch.long) 

        unet_inputs_trt = {
            "sample": unet_input_sample.to(DTYPE),
            "timestep": timestep_pt, 
            "encoder_hidden_states": encoder_hidden_states.to(DTYPE),
            "text_embeds": added_cond_kwargs["text_embeds"].to(DTYPE),
            "time_ids": added_cond_kwargs["time_ids"].to(DTYPE)
        }
        
        # Check inputs to TRT UNet
        for k, v_tensor in unet_inputs_trt.items():
            check_tensor(v_tensor, f"UNet TRT input '{k}'", step_info=current_step_info, raise_error=True)

        unet_outputs_trt = unet_trt(unet_inputs_trt)
        noise_pred = unet_outputs_trt["out_sample"]
        check_tensor(noise_pred, "noise_pred from UNet TRT", step_info=current_step_info, raise_error=True)


        if GUIDANCE_SCALE > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_uncond_fp32 = noise_pred_uncond #.float()
            noise_pred_text_fp32 = noise_pred_text #.float()
            noise_pred = noise_pred_uncond_fp32 + GUIDANCE_SCALE * (noise_pred_text_fp32 - noise_pred_uncond_fp32)
            check_tensor(noise_pred, "noise_pred after CFG", step_info=current_step_info, raise_error=True)

        latents = scheduler.step(noise_pred, t, latents).prev_sample
        check_tensor(latents, "latents after scheduler.step", step_info=current_step_info, raise_error=True)


    print("Denoising loop finished.")

    print("Decoding latents using VAE Decoder TRT...")
    latents_to_decode = latents / vae_pt.config.scaling_factor
    check_tensor(latents_to_decode, "latents_to_decode for VAE TRT", raise_error=True)


    vae_decoder_inputs_trt = {"latent_sample": latents_to_decode.to(DTYPE)} 
    check_tensor(vae_decoder_inputs_trt["latent_sample"], "VAE TRT input 'latent_sample'", raise_error=True)

    
    decoded_image_trt = vae_decoder_trt(vae_decoder_inputs_trt)["sample"]
    check_tensor(decoded_image_trt, "decoded_image_trt from VAE TRT", raise_error=True)


    decoded_image_trt = (decoded_image_trt / 2 + 0.5).clamp(0, 1)
    check_tensor(decoded_image_trt, "decoded_image_trt after denorm_clamp", raise_error=True)

    image_np = decoded_image_trt.squeeze(0).permute(1, 2, 0).cpu().float().numpy() 
    # No direct NaN check for numpy array here, but previous checks should catch it.
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

    image_pil.save(OUTPUT_IMAGE_PATH)
    print(f"Inpainted image saved to {OUTPUT_IMAGE_PATH}")

