## RETFound remote extractor (Hugging Face Space)

This folder is a standalone service that exposes:

- `GET /healthz`
- `POST /extract` (multipart form-data with `file`)

It returns JSON:

```json
{ "features": [0.123, 0.456, ...] }
```

### Deploy on Hugging Face (free)

1. Create a **new Space** on Hugging Face.
2. Choose **SDK: Docker**.
3. Push the contents of this folder to the Space repo (root).
4. In the Space settings, optionally set:
   - `RETFOUND_HF_REPO_ID` (default `gadbertrand/retfound-eye-diagnosis`)
   - `RETFOUND_HF_FILENAME` (default `retfound_cfp_vit_large_clean.pth`)
   - `HF_TOKEN` (only if the weights repo is private)

When the Space is running, your extractor URL is:

`https://<your-space-subdomain>.hf.space`

So your Render web service should set:

`RETFOUND_REMOTE_URL=https://<your-space-subdomain>.hf.space`

### Local run (optional)

```bash
cd remote_extractor_space
docker build -t retfound-extractor .
docker run -p 8000:8000 retfound-extractor
```

Test:

```bash
curl -s http://localhost:8000/healthz
```

