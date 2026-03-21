# Security Policy

## Supported Versions

Flash Mode is maintained as a security-critical component of the MLX inference ecosystem. Only the latest version of `mlx-flash` supports active security patches.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1.0 | :x:                |

## Reporting a Vulnerability

If you identify a security vulnerability in `mlx-flash`, please do not open a public issue. Instead, send a detailed report to **[your-email@example.com]**. We will acknowledge your report within 48 hours and provide a timeline for a fix.

## Model Safety

**Flash Mode does not provide a sandbox for model execution.**

*   **Trusted Sources**: Only load models from trusted repositories (e.g., official MLX community repos). Flash Mode uses `mmap()` to map weights directly into user space. While `safetensors` files are generally safe from arbitrary code execution during loading, the model itself runs with the full privileges of your Python process.
*   **Pickle Exploits**: Flash Mode relies on `mlx.core.load()` for `.safetensors` files. We strongly discourage using Flash Mode with old PyTorch `.bin` or `.pt` files, as these use `pickle` and are subject to arbitrary code execution attacks.
*   **Network Shares**: Using Flash Mode on model files stored on network drives (SMB, NFS) may be subject to Time-of-Check Time-of-Use (TOCTOU) gift attacks if the file is modified by a remote actor while being `mmap`'d.

## Research & Academic Use

`mlx-flash` is a research-oriented tool. Use it within isolated environments (containers or dedicated machines) when evaluating experimental or untrusted model architectures.
