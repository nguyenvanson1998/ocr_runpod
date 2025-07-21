# Khởi tạo môi trường 
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Download config models
```bash
python download_models_hf.py
```

# run app
1. minerU_api: API minerU xử lý pdf với fixed size chunking -> đầu ra gồm: content, metadata, image_description, image_base64 

```bash
python minerU_api.py --host 0.0.0.0 --port 8000 --reload
```

Chạy với workers:
```bash
gunicorn minerU_api:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

2. minerU_2endpoints: API minerU xử lý pdf với page_idx chunking gồm 2 endpoint là text_only và with_ocr

```bash
python minerU_2endpoints.py
```

Chạy với workers:
```bash
gunicorn minerU_2endpoints:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```