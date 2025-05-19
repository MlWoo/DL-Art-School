import os
import torch
import tensorrt as trt
import numpy as np
from diffusers import StableDiffusionXLPipeline # 或者直接加载 UNet
import onnx

# --- 配置参数 ---
# 假设这是您用于导出 UNet 的脚本
ORIGINAL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0" # 或您的基础模型
OUTPUT_DIR = "./sdxl_unet_onnx_trt_cfg_fp8_1024_f" # UNet 的输出目录
ONNX_UNET_INITIAL_PATH = os.path.join(OUTPUT_DIR, "unet", "model_initial.onnx")
FINAL_ONNX_UNET_PATH = os.path.join(OUTPUT_DIR, "unet", "model_external.onnx")
TRT_UNET_PATH = os.path.join(OUTPUT_DIR, "unet", "model.plan")

# 推理时期望的单张图片批处理大小 (例如，一次处理一张图片)
# 如果您在推理时会处理多张图片 (batch_size > 1)，请相应调整
INFERENCE_BATCH_SIZE = 1 

# UNet 特有的参数 (SDXL Inpainting)
# (B, 9, H/8, W/8) for sample
# (B,) for timestep
# (B, 77, 2048) for encoder_hidden_states
# (B, 1280) for added_cond_kwargs['text_embeds']
# (B, 6) for added_cond_kwargs['time_ids']
LATENT_CHANNELS_UNET_INPAINT = 4 # latents (4) + mask (1) + masked_image_latents (4)
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024
VAE_SCALE_FACTOR = 8 # 与您的 VAE 配置一致
TEXT_SEQ_LENGTH = 77
ENCODER_HIDDEN_STATES_DIM = 2048 # 例如 OpenCLIP ViT-G
ADDED_TEXT_EMBED_DIM = 1280    # 例如 CLIP ViT-L pooled projected
ADDED_TIME_IDS_DIM = 6

TORCH_DTYPE = torch.float16

os.makedirs(os.path.join(OUTPUT_DIR, "unet"), exist_ok=True)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# --- UNet Wrapper for ONNX Export (与您之前的脚本类似) ---
class UNetONNXWrapper(torch.nn.Module):
    def __init__(self, unet_model):
        super().__init__()
        self.unet_model = unet_model

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
        # UNet 的 forward 方法通常接收 added_cond_kwargs 字典
        added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        return self.unet_model(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs
        ).sample # 通常返回 .sample

# --- UNet ONNX 导出函数 ---
def export_unet_to_onnx(
    unet_model, # 预加载的 PyTorch UNet 模型
    initial_onnx_export_path: str,
    final_onnx_path_for_trt: str,
    # 使用 opt_export_batch_size 进行伪输入创建，这应该是考虑了 CFG 的大小
    opt_export_batch_size: int, # 例如 2 * INFERENCE_BATCH_SIZE
    latent_channels: int,
    latent_height: int,
    latent_width: int,
    text_seq_len: int,
    encoder_hidden_dim: int,
    added_text_embed_dim: int,
    added_time_ids_dim: int,
    torch_dtype: torch.dtype = torch.float16,
    opset_version: int = 17
) -> str | None:
    print(f"Starting UNet ONNX export (Opset: {opset_version})")
    print(f"  Optimizing dummy inputs for export batch size: {opt_export_batch_size}")

    unet_model.eval().to("cuda", dtype=torch_dtype)
    wrapped_unet = UNetONNXWrapper(unet_model).cuda().eval()

    # 创建伪输入，批处理大小为 opt_export_batch_size
    dummy_sample = torch.randn(opt_export_batch_size, latent_channels, latent_height, latent_width, dtype=torch_dtype).cuda()
    dummy_timestep = torch.randint(0, 1000, (opt_export_batch_size,), dtype=torch.int64).cuda() # int64 for timestep
    dummy_encoder_hidden_states = torch.randn(opt_export_batch_size, text_seq_len, encoder_hidden_dim, dtype=torch_dtype).cuda()
    dummy_text_embeds = torch.randn(opt_export_batch_size, added_text_embed_dim, dtype=torch_dtype).cuda()
    dummy_time_ids = torch.randn(opt_export_batch_size, added_time_ids_dim, dtype=torch_dtype).cuda()

    input_names = ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"]
    output_names = ["out_sample"]

    dynamic_axes = {
        "sample": {0: "batch_size", 2: "latent_height", 3: "latent_width"},
        "timestep": {0: "batch_size"},
        "encoder_hidden_states": {0: "batch_size", 1: "sequence_length"},
        "text_embeds": {0: "batch_size"},
        "time_ids": {0: "batch_size"},
        "out_sample": {0: "batch_size", 2: "latent_height", 3: "latent_width"},
    }

    try:
        print(f"  Exporting UNet to ONNX (initial pass) at {initial_onnx_export_path}...")
        torch.onnx.export(
            wrapped_unet,
            (dummy_sample, dummy_timestep, dummy_encoder_hidden_states, dummy_text_embeds, dummy_time_ids),
            initial_onnx_export_path,
            export_params=True, opset_version=opset_version, do_constant_folding=True,
            input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes,
        )
        print(f"  UNet ONNX initially exported to: {initial_onnx_export_path}")
        print(f"  Checking ONNX UNet model at path: {initial_onnx_export_path}")
        onnx.checker.check_model(initial_onnx_export_path)
        print("  ONNX UNet model (from path) checked successfully.")

        print(f"  Reloading ONNX model from {initial_onnx_export_path} to save with external data...")
        onnx_model_loaded = onnx.load(initial_onnx_export_path, load_external_data=False)
        external_weights_filename = os.path.splitext(os.path.basename(final_onnx_path_for_trt))[0] + "_weights.dat"
        print(f"  Saving UNet ONNX model with external data to {final_onnx_path_for_trt}")
        onnx.save_model(
            onnx_model_loaded, final_onnx_path_for_trt, save_as_external_data=True,
            all_tensors_to_one_file=True, location=external_weights_filename, size_threshold=1024
        )
        print("  UNet ONNX model saved with external data successfully.")
        return final_onnx_path_for_trt
    except Exception as e:
        print(f"ERROR during UNet ONNX export or external data conversion: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- UNet ONNX 到 TensorRT 转换函数 ---
def convert_unet_onnx_to_trt(
    onnx_path: str, trt_engine_path: str,
    # Profile batch sizes
    profile_min_batch: int, 
    profile_opt_batch: int, 
    profile_max_batch: int,
    # Latent dimensions for profile (can also be dynamic if needed)
    latent_h: int, latent_w: int,
    # Other fixed dimensions for profile
    latent_channels: int,
    text_seq_len: int,
    encoder_hidden_dim: int,
    added_text_embed_dim: int,
    added_time_ids_dim: int,
    fp16_mode: bool = True,
    workspace_size_gb: int = 32 # UNet is larger
) -> bool:
    print(f"Starting ONNX UNet to TensorRT conversion (FP16 mode: {fp16_mode}): {trt_engine_path}")
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size_gb * (1024 ** 3))
    #if fp16_mode: config.set_flag(trt.BuilderFlag.FP16); print("  FP16 mode enabled.")
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    #config.(trt.BuilderPrecision.FP16)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    os.chdir(os.path.dirname(onnx_path))
    onnx_path = os.path.basename(onnx_path)
    if not os.path.exists(onnx_path):
        print(f"ERROR: ONNX file not found at {onnx_path}"); return False
    print(f"  Parsing ONNX UNet model from: {onnx_path}")
    with open(onnx_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            print("ERROR: Failed to parse the ONNX UNet file.")
            for err_idx in range(parser.num_errors): print(f"  Parser error: {parser.get_error(err_idx)}")
            return False
    print("  ONNX UNet file parsed successfully.")

    profile = builder.create_optimization_profile()
    # Input names must match those in the ONNX graph
    profile.set_shape("sample", 
                      min=(profile_min_batch, latent_channels, latent_h, latent_w),
                      opt=(profile_opt_batch, latent_channels, latent_h, latent_w),
                      max=(profile_max_batch, latent_channels, latent_h, latent_w))
    profile.set_shape("timestep", 
                      min=(profile_min_batch,), 
                      opt=(profile_opt_batch,), 
                      max=(profile_max_batch,))
    profile.set_shape("encoder_hidden_states",
                      min=(profile_min_batch, text_seq_len, encoder_hidden_dim),
                      opt=(profile_opt_batch, text_seq_len, encoder_hidden_dim),
                      max=(profile_max_batch, text_seq_len, encoder_hidden_dim))
    profile.set_shape("text_embeds",
                      min=(profile_min_batch, added_text_embed_dim),
                      opt=(profile_opt_batch, added_text_embed_dim),
                      max=(profile_max_batch, added_text_embed_dim))
    profile.set_shape("time_ids", # Assuming 3D for time_ids as per previous error
                      min=(profile_min_batch, added_time_ids_dim), # (B, 1, Features)
                      opt=(profile_opt_batch, added_time_ids_dim),
                      max=(profile_max_batch, added_time_ids_dim))
                      # If time_ids is 2D in your corrected ONNX, use:
                      # min=(profile_min_batch, added_time_ids_dim), etc.

    config.add_optimization_profile(profile)
    print(f"  UNet Optimization profile added. Batch (min/opt/max): ({profile_min_batch}/{profile_opt_batch}/{profile_max_batch})")

    print("  Building TensorRT UNet engine... This may take a significant time.")
    plan = builder.build_serialized_network(network, config)
    if plan is None: print("ERROR: Failed to build the TensorRT UNet engine."); return False
    print("  TensorRT UNet engine built successfully.")
    trt_engine_path = os.path.basename(trt_engine_path)
    with open(trt_engine_path, "wb") as engine_file: engine_file.write(plan)
    print(f"TensorRT UNet engine saved to: {trt_engine_path}")
    return True

# --- 主执行脚本 ---
if __name__ == "__main__":
    print("Starting UNet to ONNX and TensorRT conversion process for CFG...")

    # 1. 加载 UNet 模型
    print(f"Loading UNet model from Hugging Face model ID: {ORIGINAL_MODEL_ID}")
    unet = None
    try:
        # 尝试直接加载UNet，如果模型结构允许
        # 对于SDXL，通常需要加载完整的pipeline来获取正确的UNet配置或直接从子文件夹加载
        # pipeline = StableDiffusionXLPipeline.from_pretrained(
        #     ORIGINAL_MODEL_ID, torch_dtype=TORCH_DTYPE, use_safetensors=True
        # )
        # unet = pipeline.unet
        # del pipeline
        # if torch.cuda.is_available(): torch.cuda.empty_cache()
        # print("UNet extracted from full pipeline successfully.")

        # 或者，如果您的模型将UNet保存在标准子文件夹中：
        from diffusers.models import UNet2DConditionModel
        unet = UNet2DConditionModel.from_pretrained(
            ORIGINAL_MODEL_ID, subfolder="unet", torch_dtype=TORCH_DTYPE, use_safetensors=True
        )
        print("UNet model loaded successfully from subfolder.")

    except Exception as e:
        print(f"ERROR: Failed to load UNet model: {e}")
        exit(1)
    
    if unet is None: print("ERROR: UNet model could not be loaded."); exit(1)

    # 2. 导出 UNet 到 ONNX (如果最终的 ONNX 文件不存在)
    # 这里的 opt_export_batch_size 应该是 2 * INFERENCE_BATCH_SIZE，因为这是 CFG 路径下的常见情况
    # 如果推理时也可能不使用 CFG (batch_size = INFERENCE_BATCH_SIZE)，TRT profile 需要覆盖这个范围
    unet_opt_export_batch_size = INFERENCE_BATCH_SIZE 
    # 如果推理时最小批次是 INFERENCE_BATCH_SIZE (无CFG)，最大是 2 * INFERENCE_BATCH_SIZE (有CFG)

    actual_final_onnx_unet_path = None
    if not os.path.exists(FINAL_ONNX_UNET_PATH):
        print(f"Final ONNX UNet ({FINAL_ONNX_UNET_PATH}) not found. Starting export...")
        actual_final_onnx_unet_path = export_unet_to_onnx(
            unet_model=unet,
            initial_onnx_export_path=ONNX_UNET_INITIAL_PATH,
            final_onnx_path_for_trt=FINAL_ONNX_UNET_PATH,
            opt_export_batch_size=unet_opt_export_batch_size, # 导出时使用 CFG 情况下的批大小
            latent_channels=LATENT_CHANNELS_UNET_INPAINT,
            latent_height=IMAGE_HEIGHT // VAE_SCALE_FACTOR,
            latent_width=IMAGE_WIDTH // VAE_SCALE_FACTOR,
            text_seq_len=TEXT_SEQ_LENGTH,
            encoder_hidden_dim=ENCODER_HIDDEN_STATES_DIM,
            added_text_embed_dim=ADDED_TEXT_EMBED_DIM,
            added_time_ids_dim=ADDED_TIME_IDS_DIM, # 这是原始2D time_ids的特征维度
            torch_dtype=TORCH_DTYPE
        )
        if not actual_final_onnx_unet_path:
            print("ERROR: ONNX UNet export failed."); exit(1)
    else:
        actual_final_onnx_unet_path = FINAL_ONNX_UNET_PATH
        print(f"Final ONNX UNet already exists: {actual_final_onnx_unet_path}")

    # 3. 将 ONNX UNet 转换为 TensorRT 引擎 (如果 TRT 引擎不存在)
    if actual_final_onnx_unet_path and os.path.exists(actual_final_onnx_unet_path):
        if not os.path.exists(TRT_UNET_PATH):
            print(f"TensorRT UNet engine ({TRT_UNET_PATH}) not found. Starting conversion...")
            
            # 定义 TRT Profile 的批处理大小范围
            # 如果推理时最小是 INFERENCE_BATCH_SIZE (无CFG)，最大是 2*INFERENCE_BATCH_SIZE (有CFG)
            # 优化目标 (opt) 通常是 CFG 情况
            profile_min_batch = INFERENCE_BATCH_SIZE
            profile_opt_batch = INFERENCE_BATCH_SIZE
            profile_max_batch = 2 * INFERENCE_BATCH_SIZE # 或者更大，如果您推理时会用更大的批次

            success = convert_unet_onnx_to_trt(
                onnx_path=actual_final_onnx_unet_path,
                trt_engine_path=TRT_UNET_PATH,
                profile_min_batch=profile_min_batch,
                profile_opt_batch=profile_opt_batch,
                profile_max_batch=profile_max_batch,
                latent_h=IMAGE_HEIGHT // VAE_SCALE_FACTOR,
                latent_w=IMAGE_WIDTH // VAE_SCALE_FACTOR,
                latent_channels=LATENT_CHANNELS_UNET_INPAINT,
                text_seq_len=TEXT_SEQ_LENGTH,
                encoder_hidden_dim=ENCODER_HIDDEN_STATES_DIM,
                added_text_embed_dim=ADDED_TEXT_EMBED_DIM,
                added_time_ids_dim=ADDED_TIME_IDS_DIM, # 这是原始2D time_ids的特征维度
                fp16_mode=True
            )
            if not success: print("ERROR: TensorRT UNet conversion failed."); exit(1)
        else:
            print(f"TensorRT UNet engine already exists: {TRT_UNET_PATH}")
    else:
        print(f"ERROR: Cannot convert UNet to TRT. Final ONNX file not found.")
        exit(1)
    
    print("UNet conversion script for CFG finished successfully.")

