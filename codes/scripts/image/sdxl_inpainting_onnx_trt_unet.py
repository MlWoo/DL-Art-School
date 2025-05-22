# from optimum.nvidia.utils import (  # optimum-nvidia 提供了很多便利工具
#     get_user_agent,
#     HUB_AUTH_SCHEME
# )
# from optimum.exporters.onnx import export_models  # 用于导出模型组件
import os
from collections import OrderedDict

import onnx
import tensorrt as trt
import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionXLInpaintPipeline

# --- 配置参数 ---
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"  # 或者你本地的路径
OUTPUT_DIR = "./sdxl_inpainting_onnx_trt"
ONNX_UNET_PATH = os.path.join(OUTPUT_DIR, "unet", "model.onnx")
TRT_UNET_PATH = os.path.join(OUTPUT_DIR, "unet", "model.plan")  # .plan 或 .engine

# 对于SDXL，典型的尺寸和批处理大小
BATCH_SIZE = 2
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
LATENT_CHANNELS = 4
INPAINT_UNET_INPUT_CHANNELS = 4  # 4 (latents) + 1 (mask) + 4 (masked_image_latents)
TEXT_SEQ_LENGTH = 77  # CLIP max sequence length

# 创建输出目录
os.makedirs(os.path.join(OUTPUT_DIR, "unet"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "text_encoder"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "text_encoder_2"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "vae_encoder"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "vae_decoder"), exist_ok=True)

# --- 1. 加载原始 PyTorch 模型 ---
print(f"Loading PyTorch model: {MODEL_ID}")
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    variant="fp16",  # Use fp16 variant if available and using float16
    use_safetensors=True,
)
# Using EulerDiscreteScheduler as it's common for SDXL
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
print("PyTorch model loaded.")

# --- 2. 导出 UNet 为 ONNX ---
# 这是最复杂的部分，因为需要正确处理 Inpainting UNet 的输入
# `optimum.exporters.onnx.export_models` 是一个更高级的抽象，
# 但对于SDXL Inpainting的UNet，我们可能需要更细致地手动处理或确保Optimum支持。

# 关键：SDXL Inpainting UNet的输入张量 `sample` 会有9个通道
# (latents, mask, masked_image_latents 拼接而成)
# 其他输入：timestep, encoder_hidden_states, added_cond_kwargs (text_embeds, time_ids)


def export_unet_to_onnx(pipeline, onnx_path):
    print(f"Exporting UNet to ONNX: {onnx_path}")
    unet = pipeline.unet
    unet.eval()  # 确保是评估模式

    print("Generating dummy inputs for UNet export using pipeline methods...")
    prompt = "A photo of an astronaut riding a horse on the moon"  # 示例提示词

    # 1. 获取 encoder_hidden_states 和 pooled_prompt_embeds (用于 text_embeds)
    # 当 do_classifier_free_guidance=False 时，返回的元组中，负向提示相关的项会是 None
    print("Calling pipeline.encode_prompt to get text embeddings...")
    encode_prompt_outputs = pipeline.encode_prompt(
        prompt=prompt,
        device="cuda",
        num_images_per_prompt=BATCH_SIZE,  # 确保与 BATCH_SIZE 一致
        do_classifier_free_guidance=False,
        negative_prompt=None,
    )
    # 对于SDXL, encode_prompt 返回 (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
    # prompt_embeds 是来自 text_encoder_2 的 token embeddings -> 这是我们的 encoder_hidden_states
    # pooled_prompt_embeds 是来自 text_encoder (TE1) 的 pooled output -> 这是我们的 text_embeds for added_cond_kwargs

    encoder_hidden_states_val = encode_prompt_outputs[0]
    # negative_prompt_embeds_val = encode_prompt_outputs[1] # Should be None
    pooled_prompt_embeds_val = encode_prompt_outputs[2]
    # negative_pooled_prompt_embeds_val = encode_prompt_outputs[3] # Should be None

    if encoder_hidden_states_val is None or pooled_prompt_embeds_val is None:
        raise ValueError("Failed to get valid embeddings from encode_prompt. Check diffusers version or prompt inputs.")

    print(
        f"Shape of encoder_hidden_states_val from encode_prompt: {encoder_hidden_states_val.shape}"
    )  # e.g., (B, 77, 2048)
    print(f"Shape of pooled_prompt_embeds_val from encode_prompt: {pooled_prompt_embeds_val.shape}")  # e.g., (B, 1280)

    # 确保数据类型和设备正确
    encoder_hidden_states_val = encoder_hidden_states_val.to(dtype=torch.float16, device="cuda")
    pooled_prompt_embeds_val = pooled_prompt_embeds_val.to(dtype=torch.float16, device="cuda")

    # 2. 获取 add_time_ids
    original_size = (IMAGE_HEIGHT, IMAGE_WIDTH)
    crops_coords_top_left = (0, 0)
    target_size = (IMAGE_HEIGHT, IMAGE_WIDTH)

    # Define default values for the newly required arguments
    # These are typical SDXL defaults, adjust if your model/finetune expects others
    aesthetic_score_val = 6.0
    negative_aesthetic_score_val = 2.5
    negative_original_size_val = original_size  # Mirror positive for non-CFG focus
    negative_crops_coords_top_left_val = crops_coords_top_left  # Mirror positive
    negative_target_size_val = target_size  # Mirror positive

    print("Calling pipeline._get_add_time_ids with aesthetic and negative args...")
    # Determine text_encoder_projection_dim
    # Typically from text_encoder_2.config.projection_dim for SDXL models
    # or unet.config.projection_class_embeddings_input_dim
    text_encoder_projection_dim_val = None
    if hasattr(pipeline.text_encoder_2, "config") and hasattr(pipeline.text_encoder_2.config, "projection_dim"):
        text_encoder_projection_dim_val = pipeline.text_encoder_2.config.projection_dim
        print(
            f"Derived text_encoder_projection_dim_val from text_encoder_2.config.projection_dim: {text_encoder_projection_dim_val}"
        )

    if (
        text_encoder_projection_dim_val is None
        and hasattr(pipeline.unet, "config")
        and hasattr(pipeline.unet.config, "projection_class_embeddings_input_dim")
    ):
        # This UNet config value often matches the expected projection dim for added conditions
        text_encoder_projection_dim_val = pipeline.unet.config.projection_class_embeddings_input_dim
        print(
            f"Derived text_encoder_projection_dim_val from unet.config.projection_class_embeddings_input_dim: {text_encoder_projection_dim_val}"
        )

    if text_encoder_projection_dim_val is None:
        # If still None, it indicates an issue with model config or an unexpected setup.
        # For many SDXL models (e.g., using OpenCLIP-ViT-G/14 like 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'),
        # the projection_dim is 1280. We might have to use a common default as a last resort,
        # but this is less robust.
        default_sdxl_projection_dim = 1280  # Common for SDXL text_encoder_2
        print(
            f"Warning: Could not reliably derive text_encoder_projection_dim. Using default: {default_sdxl_projection_dim}. This might be incorrect for your specific model."
        )
        text_encoder_projection_dim_val = default_sdxl_projection_dim

    if not isinstance(text_encoder_projection_dim_val, int):
        raise ValueError(
            f"text_encoder_projection_dim_val must be an integer, but got: {text_encoder_projection_dim_val} (type: {type(text_encoder_projection_dim_val)})"
        )

    # Also, ensure self.unet.config.addition_time_embed_dim is an integer,
    # as it's used in the problematic line. This should normally be correctly set by diffusers.
    # For SDXL, addition_time_embed_dim is often 256.
    if not hasattr(pipeline.unet.config, "addition_time_embed_dim") or not isinstance(
        pipeline.unet.config.addition_time_embed_dim, int
    ):
        raise ValueError(
            f"pipeline.unet.config.addition_time_embed_dim is invalid or not found. "
            f"Got: {getattr(pipeline.unet.config, 'addition_time_embed_dim', 'Not an attribute')}"
        )
    print(f"Using unet.config.addition_time_embed_dim: {pipeline.unet.config.addition_time_embed_dim}")
    print(f"Using text_encoder_projection_dim_val: {text_encoder_projection_dim_val}")

    print("Calling pipeline._get_add_time_ids with explicit text_encoder_projection_dim...")
    time_ids_val_maybe_batched_for_cfg = pipeline._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        aesthetic_score=aesthetic_score_val,
        negative_aesthetic_score=negative_aesthetic_score_val,
        negative_original_size=negative_original_size_val,
        negative_crops_coords_top_left=negative_crops_coords_top_left_val,
        negative_target_size=negative_target_size_val,
        dtype=encoder_hidden_states_val.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim_val,  # Explicitly pass the determined value
    )

    time_ids_val_maybe_batched_for_cfg = torch.cat(time_ids_val_maybe_batched_for_cfg, dim=0).to(
        device="cuda", dtype=torch.float16
    )
    # time_ids_val_maybe_batched_for_cfg = time_ids_val_maybe_batched_for_cfg[0].to(device="cuda", dtype=torch.float16)

    # IMPORTANT: Check the shape of time_ids_val_maybe_batched_for_cfg
    # If the pipeline's internal do_classifier_free_guidance flag was effectively True
    # OR if _get_add_time_ids now *always* returns concatenated positive and negative
    # time_ids when these arguments are provided, its first dimension might be 2 * BATCH_SIZE.
    # For ONNX export focused on a non-CFG path (BATCH_SIZE inputs to UNet),
    # we only want the "positive" part.
    print(
        f"Shape of time_ids_val from _get_add_time_ids (before potential slicing): {time_ids_val_maybe_batched_for_cfg.shape}"
    )

    # If the first dimension is 2 * BATCH_SIZE, it means positive and negative time_ids are concatenated.
    # We need only the positive part for our non-CFG UNet export.
    if time_ids_val_maybe_batched_for_cfg.shape[0] == 2 * BATCH_SIZE:
        print("Slicing time_ids to get the positive part (first BATCH_SIZE elements).")
        time_ids_val = time_ids_val_maybe_batched_for_cfg[:BATCH_SIZE]
    elif time_ids_val_maybe_batched_for_cfg.shape[0] == BATCH_SIZE:
        time_ids_val = time_ids_val_maybe_batched_for_cfg
    else:
        # This case might occur if BATCH_SIZE > 1 and the internal logic returns
        # something like (2, BATCH_SIZE // 2 * num_time_elements) - less likely
        # Or if the batching logic within _get_add_time_ids is different.
        # For now, assume it's either (BATCH_SIZE, D) or (2 * BATCH_SIZE, D).
        # If BATCH_SIZE is 1, and it returns (2, D), we take the first half.
        if BATCH_SIZE == 1 and time_ids_val_maybe_batched_for_cfg.shape[0] == 2:
            print("Slicing time_ids for BATCH_SIZE=1 to get the positive part.")
            time_ids_val = time_ids_val_maybe_batched_for_cfg[0].unsqueeze(0)  # Take the first row and keep batch dim
        elif time_ids_val_maybe_batched_for_cfg.shape[0] != BATCH_SIZE:  # Fallback if logic is unexpected
            raise ValueError(
                f"Unexpected shape for time_ids: {time_ids_val_maybe_batched_for_cfg.shape}. Expected first dim {BATCH_SIZE} or {2*BATCH_SIZE}"
            )
        else:  # Should be (BATCH_SIZE, D)
            time_ids_val = time_ids_val_maybe_batched_for_cfg

    # If BATCH_SIZE > 1 and we took a slice, the first dimension should now be BATCH_SIZE.
    # If BATCH_SIZE = 1 and we took a slice, the first dimension should now be 1.
    # If no slice was needed, it should also be BATCH_SIZE.
    # Redundant check, but good for sanity:
    if time_ids_val.shape[0] != BATCH_SIZE:
        # This specific check might be too strict if BATCH_SIZE > 1 and _get_add_time_ids
        # already handles the batching correctly for the positive part without doubling.
        # The primary concern is if it doubles for CFG and we need to select.
        # Let's refine the logic above to be more robust.
        # The check `time_ids_val_maybe_batched_for_cfg.shape[0] == 2 * BATCH_SIZE` is key.
        # If that's not met, and it's not `BATCH_SIZE` either, then it's an issue.
        pass  # The logic above should handle common cases.

    print(
        f"Final shape of time_ids_val to be used for UNet: {time_ids_val.shape}"
    )  # Should be (BATCH_SIZE, 6) for SDXL

    # 3. 创建其他伪输入
    # sample (拼接 latents, mask, masked_image_latents)
    sample_val = torch.randn(
        BATCH_SIZE,
        INPAINT_UNET_INPUT_CHANNELS,  # 4 (latents) + 1 (mask) + 4 (masked_image_latents) = 9
        IMAGE_HEIGHT // pipeline.vae_scale_factor,  # 通常是 8
        IMAGE_WIDTH // pipeline.vae_scale_factor,
        dtype=torch.float16,
    ).cuda()
    print(f"Shape of dummy sample_val: {sample_val.shape}")

    timestep_val = torch.tensor([999.0] * BATCH_SIZE, dtype=torch.float32).cuda()  # 示例timestep，确保批处理
    print(f"Shape of dummy timestep_val: {timestep_val.shape}")

    class UNetWrapper(torch.nn.Module):

        def __init__(self, unet):
            super().__init__()
            self.unet = unet

        def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
            return self.unet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                timestep_cond=None,
                cross_attention_kwargs=None,
                added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                return_dict=False,
            )[0]

    # 定义 UNet 的动态轴 (与之前类似，但现在我们有了实际的维度信息)
    dynamic_axes_unet = {
        "sample": {0: "batch_size", 2: "height_div_8", 3: "width_div_8"},
        "timestep": {0: "batch_size"},
        "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},  # sequence_length 通常是 77
        "text_embeds": {0: "batch_size"},  # for added_cond_kwargs['text_embeds']
        "time_ids": {0: "batch_size"},  # for added_cond_kwargs['time_ids']
    }

    # UNetWrapper 保持不变，它负责将扁平化的 text_embeds 和 time_ids 打包到 added_cond_kwargs 字典中
    wrapped_unet = UNetWrapper(unet).cuda().eval()
    input_names = ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"]
    output_names = ["out_sample"]

    dummy_input_unet_export = (
        sample_val,
        timestep_val,
        encoder_hidden_states_val,
        pooled_prompt_embeds_val,  # This is what the wrapper calls 'text_embeds'
        time_ids_val,  # This is what the wrapper calls 'time_ids'
    )
    import pdb

    pdb.set_trace()

    wml = wrapped_unet(sample_val, timestep_val, encoder_hidden_states_val, pooled_prompt_embeds_val, time_ids_val)

    print("Attempting UNet export with torch.onnx.export using derived dummy inputs...")
    # This is the initial export path where torch.onnx.export writes the (potentially large) model
    initial_onnx_export_path = onnx_path  # Store the original path
    try:
        torch.onnx.export(
            wrapped_unet,
            dummy_input_unet_export,
            initial_onnx_export_path,  # Export to the initial path
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes_unet,
        )
        print(f"UNet ONNX initially exported to: {initial_onnx_export_path}")

        # 1. Verify ONNX model using the file path for large models
        print(f"Checking ONNX UNet model at path: {initial_onnx_export_path}")
        try:
            onnx.checker.check_model(initial_onnx_export_path)  # Pass the file path string
            print("ONNX UNet model (from path) checked successfully.")
        except Exception as e:
            # Note: checker.check_model(path) can still fail for very large *graph structures*
            # but it's primarily designed to avoid issues with loading large tensors into memory for checking.
            print(f"Warning: onnx.checker.check_model(initial_onnx_export_path) reported an issue: {e}")
            print(
                "Proceeding with conversion to external data format, which might resolve size issues for the protobuf itself."
            )

        # 2. Convert the saved ONNX model to use external data for weights
        print(f"Reloading ONNX model from {initial_onnx_export_path} to save with external data...")
        # Load the model. If this line itself fails due to model size for the protobuf structure
        # (even without weights fully in memory), then the problem is more severe.
        # However, usually, this load is for the graph definition.
        onnx_model_loaded = onnx.load(
            initial_onnx_export_path, load_external_data=False
        )  # Initially, don't load external data if any (though torch.onnx.export doesn't create it by default)

        # Define the path for the new ONNX model that will reference external data
        onnx_external_data_dir = os.path.dirname(initial_onnx_export_path)
        onnx_model_filename_stem = os.path.splitext(os.path.basename(initial_onnx_export_path))[0]

        # Path for the new ONNX model file (this will be smaller)
        final_onnx_path_for_trt = os.path.join(onnx_external_data_dir, f"{onnx_model_filename_stem}_external.onnx")
        # Name for the external weights file (will be in the same directory as final_onnx_path_for_trt)
        external_weights_filename = f"{onnx_model_filename_stem}_weights.pb"  # You can choose any name

        print(
            f"Saving ONNX model with external data. Graph: {final_onnx_path_for_trt}, Weights: {external_weights_filename}"
        )

        onnx.save_model(
            onnx_model_loaded,
            final_onnx_path_for_trt,
            save_as_external_data=True,
            all_tensors_to_one_file=False,  # Recommended: store all external tensors in one separate file
            location=external_weights_filename,  # The name of the file where weights will be stored
            size_threshold=1024,  # Tensors larger than 1KB will be stored externally. Adjust if needed.
        )
        print("ONNX model saved with external data successfully.")
        print(f"The ONNX model to use for TensorRT conversion is now: {final_onnx_path_for_trt}")

        # Return the path to this new ONNX file
        return final_onnx_path_for_trt

    except Exception as e:
        print(f"Error during ONNX export or external data conversion: {e}")
        import traceback

        traceback.print_exc()
        return None  # Indicate failure


# --- 3. (Optional but Recommended) Export other components (Text Encoders, VAE) ---
# Using optimum.exporters.onnx.export_models is generally better for these.
# This function can export multiple components based on the pipeline structure.
# However, you need to define the "models_and_onnx_configs" carefully.
# For SDXL inpainting, it might look something like this conceptually:

# from optimum.exporters.onnx import OnnxConfig
# from optimum.utils import DEFAULT_DUMMY_SHAPES
# class SDXLUNetInpaintOnnxConfig(OnnxConfig): # Conceptual
#     # ... needs to define inputs, outputs, generate_dummy_inputs correctly for inpainting ...
#     pass

# For simplicity, this example will focus on the UNet. Full pipeline export:
# main_export(
#     model_name_or_path=MODEL_ID,
#     output=Path(OUTPUT_DIR),
#     task="stable-diffusion-xl", # May not have specific "stable-diffusion-xl-inpainting" task
#     fp16=True,
#     device="cuda",
#     # Further arguments to control which components are exported and how
# )
# This would typically be done via `optimum-cli export onnx ...` for full pipelines.
# Given the complexity and potential need for specific handling of `added_cond_kwargs`
# for the UNet, a component-wise export or a specialized script is often more reliable.


# --- 4. Convert ONNX UNet to TensorRT Engine ---
def convert_onnx_to_trt(
    onnx_path,
    trt_engine_path,
    min_batch=1,
    opt_batch=1,
    max_batch=1,
    min_latent_h=64,
    opt_latent_h=128,
    max_latent_h=128,
    min_latent_w=64,
    opt_latent_w=128,
    max_latent_w=128,
    min_seq_len=77,
    opt_seq_len=77,
    max_seq_len=77,
    fp16_mode=True,
    workspace_size_gb=1,
):
    print(f"Converting ONNX UNet to TensorRT: {trt_engine_path}")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)  # or trt.Logger.INFO for more verbosity
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()

    # Workspace size
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size_gb * (1024**3))

    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)
        # For SDXL, TF32 can also be good, but FP16 is usually faster if precision holds.
        # config.set_flag(trt.BuilderFlag.TF32)

    # Create network by parsing ONNX
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    os.chdir(os.path.dirname(onnx_path))
    onnx_path = os.path.basename(onnx_path)
    if not os.path.exists(onnx_path):
        print(f"ONNX file not found: {onnx_path}")
        return
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return
    print("ONNX file parsed successfully.")

    # Define optimization profiles for dynamic shapes
    profile = builder.create_optimization_profile()

    # Input names must match those in the ONNX model
    # For the wrapped UNet: "sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"
    # sample: (batch_size, INPAINT_UNET_INPUT_CHANNELS, height_div_8, width_div_8)
    # timestep: (batch_size,)
    # encoder_hidden_states: (batch_size, seq_len, hidden_size_text_encoder_2)
    # text_embeds: (batch_size, hidden_size_text_encoder_1_pooler)
    # time_ids: (batch_size, num_time_ids)

    # Get hidden sizes from pipeline if available
    text_encoder_2_hidden_size = pipe.text_encoder_2.config.hidden_size
    # For text_embeds from CLIP L (often from text_encoder not text_encoder_2's projection)
    # This depends on how `encode_prompt` structures `text_embeds` for SDXL.
    # Typically, it's the pooled output of one of the text encoders.
    # Let's assume it's from text_encoder (CLIP ViT-L)
    text_encoder_1_pool_dim = pipe.text_encoder.config.hidden_size  # Or a specific projection dim
    # For SDXL, this is usually 1280 if `text_embeds` are from `prompt_embeds` and not `pooled_prompt_embeds`
    # from `pipeline.encode_prompt`. If it's `added_cond_kwargs['text_embeds']` it's typically the pooled output.
    # Example: `added_cond_kwargs['text_embeds']` has shape (batch, 1280) for SDXL base
    added_text_embed_dim = 1280  # Common for SDXL
    added_time_ids_dim = 6  # Common for SDXL (orig_H, orig_W, crop_top, crop_left, target_H, target_W)

    # sample
    profile.set_shape(
        "sample",
        (min_batch, INPAINT_UNET_INPUT_CHANNELS, min_latent_h, min_latent_w),  # Min
        (opt_batch, INPAINT_UNET_INPUT_CHANNELS, opt_latent_h, opt_latent_w),  # Opt
        (max_batch, INPAINT_UNET_INPUT_CHANNELS, max_latent_h, max_latent_w),  # Max
    )
    # timestep
    profile.set_shape("timestep", (min_batch,), (opt_batch,), (max_batch,))
    # encoder_hidden_states
    profile.set_shape(
        "encoder_hidden_states",
        (min_batch, min_seq_len, 2048),
        (opt_batch, opt_seq_len, 2048),
        (max_batch, max_seq_len, 2048),
    )
    # text_embeds
    profile.set_shape(
        "text_embeds",
        (min_batch, added_text_embed_dim),
        (opt_batch, added_text_embed_dim),
        (max_batch, added_text_embed_dim),
    )
    # time_ids
    profile.set_shape(
        "time_ids", (min_batch, added_time_ids_dim), (opt_batch, added_time_ids_dim), (max_batch, added_time_ids_dim)
    )

    config.add_optimization_profile(profile)

    # Build engine
    print("Building TensorRT engine... This may take a while.")
    # serialized_engine = builder.build_serialized_network(network, config) # Deprecated
    plan = builder.build_serialized_network(network, config)

    if plan is None:
        print("ERROR: Failed to build the TensorRT engine.")
        return

    print("TensorRT engine built successfully.")
    trt_engine_path = os.path.basename(trt_engine_path)
    with open(trt_engine_path, "wb") as f:
        f.write(plan)
    print(f"TensorRT engine saved to: {trt_engine_path}")


# --- In your main execution block ---
if __name__ == "__main__":
    # ... (pipeline loading, etc.) ...

    # ONNX_UNET_PATH is the initial target for torch.onnx.export
    # final_onnx_path_for_trt will be the path to the model with external data
    final_onnx_path_for_trt = os.path.join(OUTPUT_DIR, "unet", "model_external.onnx")  # Path convention

    if not os.path.exists(final_onnx_path_for_trt):  # Check for the final, TRT-ready ONNX file
        print(f"Final ONNX model for TRT ({final_onnx_path_for_trt}) not found. Running export process...")

        # Call the modified export function. It will handle the initial export and
        # the conversion to external data, then return the path to the final ONNX file.
        # ONNX_UNET_PATH (defined at the top of your script) is the *initial* export target.
        returned_onnx_path = export_unet_to_onnx(pipe, ONNX_UNET_PATH)

        if returned_onnx_path and returned_onnx_path == final_onnx_path_for_trt:
            print(f"Successfully exported and prepared ONNX for TRT: {returned_onnx_path}")
        elif returned_onnx_path:
            print(
                f"Warning: Exported ONNX path {returned_onnx_path} differs from expected {final_onnx_path_for_trt}. Using returned path."
            )
            final_onnx_path_for_trt = returned_onnx_path  # Use the path actually returned
        else:
            print("ONNX export process failed.")
            exit(1)
    else:
        print(f"Final ONNX UNet for TRT already exists: {final_onnx_path_for_trt}")
        # No need to update final_onnx_path_for_trt if it already exists

    # 2. Convert the (potentially external data) ONNX UNet to TensorRT Engine
    if os.path.exists(final_onnx_path_for_trt) and not os.path.exists(TRT_UNET_PATH):
        print(f"Converting ONNX model {final_onnx_path_for_trt} to TensorRT engine...")
        convert_onnx_to_trt(
            final_onnx_path_for_trt,  # Use the ONNX model with external data
            TRT_UNET_PATH,
            # ... (your min_batch, opt_batch, etc. parameters for TensorRT profiles) ...
            min_batch=BATCH_SIZE,
            opt_batch=BATCH_SIZE,
            max_batch=BATCH_SIZE,  # Example
            min_latent_h=IMAGE_HEIGHT // pipe.vae_scale_factor,
            opt_latent_h=IMAGE_HEIGHT // pipe.vae_scale_factor,
            max_latent_h=IMAGE_HEIGHT // pipe.vae_scale_factor,
            min_latent_w=IMAGE_WIDTH // pipe.vae_scale_factor,
            opt_latent_w=IMAGE_WIDTH // pipe.vae_scale_factor,
            max_latent_w=IMAGE_WIDTH // pipe.vae_scale_factor,
            min_seq_len=(
                encoder_hidden_states_val.shape[1] if "encoder_hidden_states_val" in locals() else 77
            ),  # Use actual if available
            opt_seq_len=encoder_hidden_states_val.shape[1] if "encoder_hidden_states_val" in locals() else 77,
            max_seq_len=encoder_hidden_states_val.shape[1] if "encoder_hidden_states_val" in locals() else 77,
            fp16_mode=True,
            workspace_size_gb=4,
        )
    elif not os.path.exists(final_onnx_path_for_trt):
        print(f"Cannot convert to TensorRT: Final ONNX UNet file not found at {final_onnx_path_for_trt}.")
    else:
        print(f"TensorRT UNet engine already exists: {TRT_UNET_PATH}")

    print("Script finished.")

    # --- 后续步骤 ---
    # 1. 如果需要，导出并转换 VAE (Encoder/Decoder) 和 Text Encoders (CLIP L & G) 为 ONNX/TensorRT。
    #    VAE Decoder 对于整体性能也很重要。
    # 2. 编写一个推理脚本：
    #    - 加载 TensorRT UNet engine (以及其他组件的 PyTorch/ONNX/TRT版本)。
    #    - 实现完整的 Stable Diffusion Inpainting 采样循环。
    #    - 这涉及到创建 TensorRT执行上下文(execution context)，分配GPU内存，
    #      在主机和设备之间复制数据，并调用 TensorRT engine 进行推理。
    #    - `polygraphy run your_model.onnx --trt --save-engine your_engine.plan` 也可以用来构建引擎。
    #    - NVIDIA 的 `TensorRT/demo/Diffusion` 目录下可能有 SD 推理的示例代码，可以借鉴。
