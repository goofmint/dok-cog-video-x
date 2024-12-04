from typing import Literal
import argparse

import torch
from diffusers import (
    CogVideoXPipeline,
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
)

from diffusers.utils import export_to_video, load_image, load_video

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument(
    '--output',
    default='/opt/artifact',
    help='出力先ディレクトリを指定します。',
)
arg_parser.add_argument(
    '--prompt',
    default='',
    help='プロンプト',
)
arg_parser.add_argument(
    '--type',
    default='t2v',
    help='実行する処理',
)
arg_parser.add_argument(
    '--image',
    default='',
    help='ベースの画像（または動画）',
)

arg_parser.add_argument(
    '--id',
    default='',
    help='タスクIDを指定します。',
)
arg_parser.add_argument('--s3-bucket', help='S3のバケットを指定します。')
arg_parser.add_argument('--s3-endpoint', help='S3互換エンドポイントのURLを指定します。')
arg_parser.add_argument('--s3-secret', help='S3のシークレットアクセスキーを指定します。')
arg_parser.add_argument('--s3-token', help='S3のアクセスキーIDを指定します。')

args = arg_parser.parse_args()

def generate_video(
    prompt: str,
    model_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    strength: float = 0.8,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"],  # i2v: image to video, v2v: video to video
    seed: int = 42,
):

    image = None
    video = None

    print(image_or_video_path)

    if generate_type == "i2v":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        image = load_image(image=image_or_video_path)
    elif generate_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    else:
        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        video = load_video(image_or_video_path)

    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        pipe.fuse_lora(lora_scale=1 / lora_rank)

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    pipe.enable_sequential_cpu_offload()
    #pipe.enable_model_cpu_offload()

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    if generate_type == "i2v":
        video_generate = pipe(
            prompt=prompt,
            image=image,  # The path of the image to be used as the background of the video
            num_videos_per_prompt=num_videos_per_prompt,  # Number of videos to generate per prompt
            num_inference_steps=num_inference_steps,  # Number of inference steps
            num_frames=49,  # Number of frames to generate，changed to 49 for diffusers version `0.30.3` and after.
            use_dynamic_cfg=True,  # This id used for DPM Sechduler, for DDIM scheduler, it should be False
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
        ).frames[0]
    elif generate_type == "t2v":
        video_generate = pipe(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=49,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
        ).frames[0]
    else:
        video_generate = pipe(
            prompt=prompt,
            video=video,  # The path of the video to be used as the background of the video
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            # num_frames=49,
            use_dynamic_cfg=True,
            strength=strength,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
        ).frames[0]
    # 5. Export the generated frames to a video file. fps must be 8 for original video.
    export_to_video(video_generate, output_path, fps=8)

dtype = torch.bfloat16
generate_type = args.type # t2v: text to video, i2v: image to video, v2v: video 

if not args.image:
    image_or_video_path = args.image

lora_path = None
strength = 0.8
prompt=args.prompt

if generate_type == "t2v":
    model_path = "THUDM/CogVideoX-5b"
elif generate_type == "i2v":
    model_path = "THUDM/CogVideoX-5b-I2V"
else:
    model_path = "THUDM/CogVideoX-5b"

output_path = f'{args.output}/output-{args.id}.mp4'
print(f"Generating video")
generate_video(
    prompt=prompt,
    model_path=model_path,
    lora_path=lora_path,
    lora_rank=128,
    output_path=output_path,
    image_or_video_path=image_or_video_path,
    num_inference_steps=50,
    strength=strength,
    guidance_scale=6.0,
    num_videos_per_prompt=1,
    dtype=dtype,
    generate_type=generate_type,
    seed=42,
)
