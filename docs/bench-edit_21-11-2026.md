# Última modificación: 2026-11-22

# Benchmark de Krea 2 Identity Edit

Generado con `make bench-edit` (misma semilla, misma imagen fuente, mismos pasos).
El pico de VRAM es acumulado desde el arranque del contenedor, no por ejecución.
El LoRA debe ir en el prompt (`<lora:...>`): sin el tag el modelo edita sin LoRA y el
tiempo baja ~3x. El benchmark lo verifica en el log y aborta si no se cargó.

# Resultados

| Config | Endpoint | Resolución | Pasos | grounding_px | ref_boost | LoRA | TE | Runs (s) | Mediana (s) | Pico VRAM (GB) | OOM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| active | identity-edit | 640x640 | 8 | 768 | 4 | krea2_identity_edit_v1_2_r64 | cuda | 34.93, 34.99 | 34.96 | 5.09 | 0 |
| te-cpu | identity-edit | 640x640 | 8 | 768 | 4 | krea2_identity_edit_v1_2_r64 | cpu | 148.73, 142.52 | 145.62 | 5.06 | 0 |
| te-gpu | identity-edit | 640x640 | 8 | 768 | 4 | krea2_identity_edit_v1_2_r64 | cuda | 35.51, 34.91 | 35.21 | 4.91 | 0 |
| distill4-step | identity-edit | 640x640 | 4 | 768 | 4 | krea2_identity_edit_v1_2_r64, krea2_turbo_4step_rank_64_lora | cuda | 20.74, 20.60 | 20.67 | 5.38 | 0 |
| distill4-verify | identity-edit | 640x640 | 4 | 768 | 4 | krea2_identity_edit_v1_2_r64, krea2_turbo_4step_rank_64_lora | cuda | 20.67 | 20.67 | 5.38 | 0 |
| distill4-w075-boost2 | identity-edit | 640x640 | 4 | 768 | 2 | krea2_identity_edit_v1_2_r64, krea2_turbo_4step_rank_64_lora | cuda | 21.37, 20.84 | 21.11 | 2.66 | 0 |
| v1.2-baseline-832 | identity-edit | 640x640 | 8 | 768 | 4 | krea2_identity_edit_v1_2_r64 | cuda | 34.83, 35.23 | 35.03 | 3.00 | 0 |
| V2-r64-sin-extension | identity-edit | 640x640 | 8 | 768 | 4 | krea2_identity_edit_v2_r64 | cuda | 33.06, 33.16 | 33.11 | 3.19 | 0 |
| V2-r64-boost2 | identity-edit | 640x640 | 8 | 768 | 2 | krea2_identity_edit_v2_r64 | cuda | 30.54 | 30.54 | 4.28 | 0 |
| V2-r64-boost3 | identity-edit | 640x640 | 8 | 768 | 3 | krea2_identity_edit_v2_r64 | cuda | 30.66 | 30.66 | 4.28 | 0 |
| W4A8-v1.2 | identity-edit | 640x640 | 8 | 768 | 4 | krea2_identity_edit_v1_2_r64 | cuda | 44.89, 44.72 | 44.80 | 5.38 | 0 |
| INT4-restored-check | identity-edit | 640x640 | 8 | 768 | 2 | krea2_identity_edit_v2_r64 | cuda | 32.96, 32.98 | 32.97 | 5.38 | 0 |
| V2+distill4-b2 | identity-edit | 640x640 | 4 | 768 | 2 | krea2_identity_edit_v2_r64, krea2_turbo_4step_rank_64_lora | cuda | 19.97, 19.78 | 19.88 | 5.38 | 0 |
| final-lowvram-832px | identity-edit | 832x832 | 4 | 768 | 2 | krea2_identity_edit_v2_r64, krea2_turbo_4step_rank_64_lora | cuda | 29.76, 29.39 | 29.58 | 5.47 | 0 |
| final-lowvram-1024px | identity-edit | 1024x1024 | 4 | 768 | 2 | krea2_identity_edit_v2_r64, krea2_turbo_4step_rank_64_lora | cuda | 43.84, 43.76 | 43.80 | 5.59 | 0 |
| final-normal-832px | identity-edit | 832x832 | 4 | 768 | 2 | krea2_identity_edit_v2_r64, krea2_turbo_4step_rank_64_lora | cuda | 29.64, 29.58 | 29.61 | 5.38 | 0 |
| final-normal-1024px | identity-edit | 1024x1024 | 4 | 768 | 2 | krea2_identity_edit_v2_r64, krea2_turbo_4step_rank_64_lora | cuda | 43.64, 44.04 | 43.84 | 5.53 | 0 |
| final-normal-1280px | identity-edit | 1280x1280 | 4 | 768 | 2 | krea2_identity_edit_v2_r64, krea2_turbo_4step_rank_64_lora | cuda | 73.26, 73.16, 73.11 | 73.16 | 6.25 | 0 |
| recipe-final-640 | identity-edit | 640x640 | 4 | 768 | 2 | krea2_identity_edit_v2_r64, krea2_turbo_4step_rank_64_lora | cuda | 19.34, 19.36 | 19.35 | 5.22 | 0 |
| TE-oficial-qwen3vl_4b | identity-edit | 640x640 | 4 | 768 | 2 | krea2_identity_edit_v2_r64, krea2_turbo_4step_rank_64_lora | cuda | 19.85, 19.94 | 19.90 | 5.47 | 0 |
| receta-final-restaurada | identity-edit | 640x640 | 4 | 768 | 2 | krea2_identity_edit_v2_r64, krea2_turbo_4step_rank_64_lora | cuda | 20.01, 19.82 | 19.91 | 5.62 | 0 |
| VAE-krea2RealVae_v10 | identity-edit | 640x640 | 4 | 768 | 2 | krea2_identity_edit_v2_r64, krea2_turbo_4step_rank_64_lora | cuda | 24.16, 24.09 | 24.13 | 5.44 | 0 |
| VAE-qwen_image_vae | identity-edit | 640x640 | 4 | 768 | 2 | krea2_identity_edit_v2_r64, krea2_turbo_4step_rank_64_lora | cuda | 24.07, 24.07 | 24.07 | 5.44 | 0 |
| VAE-krea2RealVae_v10 | identity-edit | 640x640 | 4 | 768 | 2 | krea2_identity_edit_v2_r64, krea2_turbo_4step_rank_64_lora | cuda | 33.59, 33.76 | 33.68 | 5.44 | 0 |
| VAE-qwen_image_vae | identity-edit | 640x640 | 4 | 768 | 2 | krea2_identity_edit_v2_r64, krea2_turbo_4step_rank_64_lora | cuda | 24.51, 24.46 | 24.49 | 5.44 | 0 |
| VAE-Krea2-HD-vae | identity-edit | 640x640 | 4 | 768 | 2 | krea2_identity_edit_v2_r64, krea2_turbo_4step_rank_64_lora | cuda | 24.61, 24.95 | 24.78 | 5.44 | 0 |
| VAE-krea2Real-recheck | identity-edit | 640x640 | 4 | 768 | 2 | krea2_identity_edit_v2_r64, krea2_turbo_4step_rank_64_lora | cuda | 23.96, 23.91, 24.02 | 23.96 | 5.44 | 0 |
| estado-final-verificado | identity-edit | 640x640 | 4 | 768 | 2 | krea2_identity_edit_v2_r64, krea2_turbo_4step_rank_64_lora | cuda | 24.24, 23.86, 23.81 | 23.86 | 5.47 | 0 |
| estado-final-post-SPEED | identity-edit | 640x640 | 4 | 768 | 2 | krea2_identity_edit_v2_r64, krea2_turbo_4step_rank_64_lora | cuda | 23.87, 23.58 | 23.72 | 5.53 | 0 |
