Experiment LORA training and evaluation on these models:

| Model | Size | Hugging Face repo | Image editing |
| --- | ---: | --- | --- |
| Stable Diffusion 1.5 | 0.9B | `stable-diffusion-v1-5/stable-diffusion-v1-5` | Yes, via img2img/inpaint ecosystem |
| Stable Diffusion 2.1 | 0.9B | `sd2-community/stable-diffusion-2-1` | Yes, via img2img/inpaint ecosystem |
| Stable Diffusion 3.5 Medium | 2.5B | `stabilityai/stable-diffusion-3.5-medium` | Implementation/API-dependent |
| FLUX.2 Klein 4B | 4B | `black-forest-labs/FLUX.2-klein-base-4B` | Yes, native multi-reference/editing |

## Results so far
- Flux.2 Klein 4B: works well after 2000 LORA training steps
- SD 3.5: not natively img2img. Used Model learned the style, but ignores the structure of the source image
- sd15: Used pix2pix pipeline, but result is weak: learns the style but quality is not as good. Ignores the source image when image guidance is high; fails to transfers style when image guidance is weak.

## Next step
- Flux.2 Klein 4B: transfer to ONNX and then CANN format for device deployment
- SD 3.5: crossed out / on pause. Parameter size is similar to Klein but not for natively editable
- SD 1.5: Try ControlNet + IP-Adapter approach with better base weights.