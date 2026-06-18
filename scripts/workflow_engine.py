"""
Wanderix — Image Generation Workflow Engine
============================================
Inspired by ComfyUI (117K⭐) node-based workflow pattern.

Key patterns adopted:
- Parameter injection into predefined workflows
- Style presets with override chains
- Batch generation with parameter sweep
- Output metadata tracking
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class StylePreset:
    """Predefined style with parameter overrides."""
    name: str
    prompt_prefix: str
    prompt_suffix: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    guidance_scale: float = 7.5
    steps: int = 30

    def apply(self, user_prompt: str) -> str:
        parts = []
        if self.prompt_prefix:
            parts.append(self.prompt_prefix)
        parts.append(user_prompt)
        if self.prompt_suffix:
            parts.append(self.prompt_suffix)
        return ", ".join(parts)


STYLE_PRESETS = {
    "写实": StylePreset("写实", "photorealistic, 8k, detailed", "professional photography", "cartoon, anime", 1024, 1024, 7.5, 30),
    "插画": StylePreset("插画", "illustration, digital art", "vibrant colors, clean lines", "photorealistic", 1024, 1024, 8.0, 25),
    "赛博朋克": StylePreset("赛博朋克", "cyberpunk style, neon lights", "futuristic, dark atmosphere", "natural, pastoral", 1024, 768, 9.0, 35),
    "水墨": StylePreset("水墨", "chinese ink painting, traditional", "elegant, minimalist", "western style", 768, 1024, 6.0, 20),
    "极简": StylePreset("极简", "minimalist design, clean", "simple, white space", "complex, cluttered", 1024, 1024, 5.0, 20),
}


@dataclass
class ImageTask:
    """Single image generation task."""
    task_id: str
    prompt: str
    style: str = "写实"
    width: int = 1024
    height: int = 1024
    seed: int = -1
    status: str = "pending"
    output_path: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class BatchJob:
    """Batch image generation job with parameter sweep."""
    job_id: str
    tasks: list[ImageTask] = field(default_factory=list)
    base_prompt: str = ""
    styles: list[str] = field(default_factory=list)

    @classmethod
    def from_prompt(cls, prompt: str, styles: list[str] = None, sizes: list[tuple] = None) -> "BatchJob":
        """Create a batch job from a single prompt with style/size variations."""
        job = cls(job_id=f"batch_{hash(prompt) % 10000}", base_prompt=prompt)
        styles = styles or ["写实"]
        sizes = sizes or [(1024, 1024)]

        task_idx = 0
        for style_name in styles:
            preset = STYLE_PRESETS.get(style_name, STYLE_PRESETS["写实"])
            for w, h in sizes:
                task = ImageTask(
                    task_id=f"{job.job_id}_{task_idx}",
                    prompt=preset.apply(prompt),
                    style=style_name,
                    width=w,
                    height=h,
                    metadata={"preset": style_name, "original_prompt": prompt},
                )
                job.tasks.append(task)
                task_idx += 1
        return job


def generate_workflow_params(task: ImageTask) -> dict:
    """Generate API-compatible parameters for image generation."""
    preset = STYLE_PRESETS.get(task.style, STYLE_PRESETS["写实"])
    return {
        "prompt": preset.apply(task.prompt),
        "negative_prompt": preset.negative_prompt,
        "width": task.width or preset.width,
        "height": task.height or preset.height,
        "guidance_scale": preset.guidance_scale,
        "num_inference_steps": preset.steps,
        "seed": task.seed,
    }


if __name__ == "__main__":
    job = BatchJob.from_prompt("一只猫在月光下", styles=["写实", "水墨", "赛博朋克"])
    for t in job.tasks:
        params = generate_workflow_params(t)
        print(f"[{t.style}] {params['prompt'][:60]}... ({params['width']}x{params['height']})")
