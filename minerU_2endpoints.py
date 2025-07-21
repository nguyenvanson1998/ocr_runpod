import uuid
import os
import uvicorn
import click
import shutil
import json
from pathlib import Path
from glob import glob
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from loguru import logger
from base64 import b64encode
from dotenv import load_dotenv
import asyncio
import json

from vertexai.generative_models import GenerativeModel, Image
from PIL import Image as PILImage
from io import BytesIO

load_dotenv()

from mineru.cli.common import aio_do_parse, read_fn, pdf_suffixes, image_suffixes
from mineru.utils.cli_parser import arg_parse
from mineru.version import __version__

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

WARMUP_PDF_PATH = "./files_test/dummy.pdf"

@app.on_event("startup")
async def startup_event():
    logger.info("MinerU API is starting up...")
    warmup_dir_temp = f"./mineru_warmup_temp_{uuid.uuid4().hex}"
    os.makedirs(warmup_dir_temp, exist_ok=True)
    try:
        if not os.path.exists(WARMUP_PDF_PATH):
            logger.error(f"Warmup file not found: {WARMUP_PDF_PATH}.")
            return
        
        warmup_pdf_name = Path(WARMUP_PDF_PATH).stem
        dummy_pdf_bytes = read_fn(Path(WARMUP_PDF_PATH))

        await aio_do_parse(
            output_dir=warmup_dir_temp,
            pdf_file_names=[warmup_pdf_name],
            pdf_bytes_list=[dummy_pdf_bytes],
            p_lang_list=["en"],
            backend="pipeline",
            parse_method="auto",
            formula_enable=True,
            table_enable=True,
            server_url=None,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=False,
            f_dump_middle_json=False,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=True,
            f_dump_images=True,
            start_page_id=0,
            end_page_id=2,
            **getattr(app.state, "config", {})
        )

        logger.info("Warmup completed successfully!")
    except Exception as e:
        logger.error(f"Error during warmup: {str(e)}")
        logger.exception(e)
    finally:
        if os.path.exists(warmup_dir_temp):
            shutil.rmtree(warmup_dir_temp, ignore_errors=True)
            logger.info(f"Cleaned up warmup directory: {warmup_dir_temp}")

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return b64encode(f.read()).decode()

def generate_img_desc(img_path: str, img_caption: str) -> str:
    try:
        model_name = "gemini-2.0-flash"
        chat_model = GenerativeModel(model_name)
        source_img = Image.load_from_file(img_path)
        prompt = f"""You are an expert in image analysis. Your task is to describe the image in detail, including objects, actions, and context.
                     The image already has a caption: \"{img_caption}\". Incorporate this information into your detailed description.
                     If the image contains text, include a brief summary of the text content as well.
                     Provide a comprehensive description that captures the essence of the image."""
        content = [source_img, prompt]
        generation_config = {"temperature": 0.7}
        response = chat_model.generate_content(
            content,
            generation_config=generation_config
        )
        return response.text
    except Exception as e:
        logger.error(f"Error generating image description for {img_path}: {e}")
        return f"Error generating image description: {e}"

def load_content_list_from_file(pdf_name: str, parse_dir: str) -> List[Dict]:
    """Load content list from the content_list.json file"""
    content_list_path = os.path.join(parse_dir, f"{pdf_name}_content_list.json")
    if not os.path.exists(content_list_path):
        logger.warning(f"Content list file not found: {content_list_path}")
        return []
    
    try:
        with open(content_list_path, 'r', encoding='utf-8') as f:
            content_data = json.load(f)
        logger.info(f"Loaded content list from {content_list_path}")
        return content_data
    except Exception as e:
        logger.error(f"Error loading content list from {content_list_path}: {e}")
        return []

async def _process_files_and_parse(
    files: List[UploadFile],
    output_dir: str,
    lang_list: List[str],
    backend: str,
    parse_method: str,
    formula_enable: bool,
    table_enable: bool,
    server_url: Optional[str],
    start_page_id: int,
    end_page_id: int,
    config: Dict[str, Any],
    dump_images: bool = True,
):
    batch_unique_dir = None
    try:
        batch_unique_dir = os.path.join(output_dir, str(uuid.uuid4()))
        os.makedirs(batch_unique_dir, exist_ok=True)
        logger.info(f"Processing files in batch directory: {batch_unique_dir}")
        
        pdf_file_names = []
        pdf_bytes_list = []
        actual_lang_list_for_parse = []

        for file_obj in files:
            file_content = await file_obj.read()
            file_path_obj = Path(file_obj.filename)
            file_stem = file_path_obj.stem

            if file_path_obj.suffix.lower() in pdf_suffixes + image_suffixes:
                temp_file_path = Path(batch_unique_dir) / file_path_obj.name

                with open(temp_file_path, "wb") as f:
                    f.write(file_content)

                try:
                    processed_bytes = read_fn(temp_file_path)
                    pdf_bytes_list.append(processed_bytes)
                    pdf_file_names.append(file_stem)
                    actual_lang_list_for_parse.append(lang_list[0] if lang_list else "en")

                    os.remove(temp_file_path)
                except Exception as e:
                    logger.error(f"Failed to load or process file {file_obj.filename}: {str(e)}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to process file {file_obj.filename}: {str(e)}"
                    )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file type: {file_path_obj.suffix} for file {file_obj.filename}"
                )

        if not pdf_file_names:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid PDF provided"
            )

        await aio_do_parse(
            output_dir=batch_unique_dir,
            pdf_file_names=pdf_file_names,
            pdf_bytes_list=pdf_bytes_list,
            p_lang_list=actual_lang_list_for_parse,
            backend=backend,
            parse_method=parse_method,
            formula_enable=formula_enable,
            table_enable=table_enable,
            server_url=server_url,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
            f_dump_md=False,
            f_dump_middle_json=False,
            f_dump_model_output=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=True,  
            f_dump_images=dump_images,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            **config
        )
        logger.info(f"Parsing completed for {batch_unique_dir}")
        return batch_unique_dir, pdf_file_names
    except Exception as e:
        if batch_unique_dir and os.path.exists(batch_unique_dir):
            shutil.rmtree(batch_unique_dir, ignore_errors=True)
        raise e

@app.post(path="/process_document_ocr_mode")
async def process_document_ocr_mode(
    files: List[UploadFile] = File(...),
    output_dir: str = Form("./output_minerU"),
    lang_list: List[str] = Form(["en"]),
    backend: str = Form("pipeline"),
    parse_method: str = Form("auto"),
    formula_enable: bool = Form(True),
    table_enable: bool = Form(True),
    server_url: Optional[str] = Form(None),
    start_page_id: int = Form(0),
    end_page_id: int = Form(9999),
):
    config = getattr(app.state, "config", {})
    processed_results = []
    batch_unique_dir = None
    loop = asyncio.get_event_loop()

    try:
        batch_unique_dir, pdf_file_names = await _process_files_and_parse(
            files=files,
            output_dir=output_dir,
            lang_list=lang_list,
            backend=backend,
            parse_method=parse_method,
            formula_enable=formula_enable,
            table_enable=table_enable,
            server_url=server_url,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            config=config,
            dump_images=True
        )

        for file_idx, pdf_name in enumerate(pdf_file_names):
            if backend.startswith("pipeline"):
                parse_dir_for_file = os.path.join(batch_unique_dir, pdf_name, parse_method)
            else:
                parse_dir_for_file = os.path.join(batch_unique_dir, pdf_name, "vlm")

            current_file_content_list = load_content_list_from_file(pdf_name, parse_dir_for_file)
            
            if not current_file_content_list:
                logger.warning(f"No content found for file {pdf_name}")
                continue

            current_page_content_buffer = {} 

            for content_item in current_file_content_list:
                if not isinstance(content_item, dict):
                    logger.warning(f"Invalid content item: {type(content_item)}")
                    continue
                    
                item_type = content_item.get("type")
                item_page_idx = content_item.get("page_idx", 0) + 1 

                if item_page_idx not in current_page_content_buffer:
                    current_page_content_buffer[item_page_idx] = {
                        "text_buffer": "",
                        "images": []
                    }

                if item_type in ["text", "equation"]:
                    text_content = content_item.get("text", "")
                    if text_content.strip(): 
                        current_page_content_buffer[item_page_idx]["text_buffer"] += text_content + "\n"
                
                elif item_type == "table":
                    table_caption = " ".join(content_item.get("table_caption", [])).strip()
                    table_body = content_item.get("table_body", "")
                    table_footnote = " ".join(content_item.get("table_footnote", [])).strip()
                    
                    parts = []
                    if table_caption:
                        parts.append(table_caption)
                    if table_body:
                        parts.append(table_body)
                    if table_footnote:
                        parts.append(table_footnote)

                    table_text = "\n".join(parts)
                    if table_text.strip():
                        current_page_content_buffer[item_page_idx]["text_buffer"] += table_text + "\n"

                elif item_type == "image":
                    image_caption = content_item.get("image_caption", [])
                    image_relative_path = content_item.get("img_path")
                    
                    if not image_relative_path:
                        logger.warning(f"No image path found for image on page {item_page_idx}")
                        continue
                        
                    image_full_path = os.path.join(parse_dir_for_file, image_relative_path)

                    if image_caption:
                        image_caption = " ".join(image_caption).strip()
                    else:
                        image_caption = f"Image on page {item_page_idx} without specific caption."

                    current_page_content_buffer[item_page_idx]["images"].append({
                        "path": image_full_path,
                        "caption": image_caption
                    })

            for page_idx in sorted(current_page_content_buffer.keys()):
                page_data = current_page_content_buffer[page_idx]
                text_buffer = page_data["text_buffer"].strip()

                if text_buffer:
                    processed_results.append({
                        "content": text_buffer,
                        "metadata": {
                            "pageStart": page_idx,
                            "pageEnd": page_idx,
                        },
                        "image_description": None,
                        "image_base64": None
                    })
                    logger.info(f"Added text chunk for page {page_idx}.")

                for img_info in page_data["images"]:
                    image_full_path = img_info["path"]
                    image_caption = img_info["caption"]

                    if os.path.exists(image_full_path):
                        try:
                            encoded_image = encode_image(image_full_path)
                            image_description = await loop.run_in_executor(
                                None,
                                generate_img_desc,
                                image_full_path,
                                image_caption
                            )
                            
                            processed_results.append({
                                "content": image_caption,
                                "metadata": {"page": page_idx},
                                "image_description": image_description,
                                "image_base64": encoded_image
                            })
                            logger.info(f"Added image chunk for page {page_idx}.")
                        except Exception as e:
                            logger.error(f"Error processing image {image_full_path}: {e}")
                    else:
                        logger.warning(f"Image file not found at {image_full_path}")

        logger.info(f"All files processed and post-processed for full content. Total chunks: {len(processed_results)}")

        return JSONResponse(
            status_code=200,
            content={
                "minerU_processed_document": processed_results,
            }
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during processing: {str(e)}"
        )
    finally:
        if batch_unique_dir and os.path.exists(batch_unique_dir):
            logger.info(f"Cleaning up temporary directory: {batch_unique_dir}")
            shutil.rmtree(batch_unique_dir, ignore_errors=True)


@app.post(path="/process_document_text_only")
async def process_document_text_only(
    files: List[UploadFile] = File(...),
    output_dir: str = Form("./output_minerU"),
    lang_list: List[str] = Form(["en"]),
    backend: str = Form("pipeline"),
    parse_method: str = Form("auto"),
    formula_enable: bool = Form(True),
    table_enable: bool = Form(True),
    server_url: Optional[str] = Form(None),
    start_page_id: int = Form(0),
    end_page_id: int = Form(9999),
):
    config = getattr(app.state, "config", {})
    processed_results = []
    batch_unique_dir = None

    try:
        batch_unique_dir, pdf_file_names = await _process_files_and_parse(
            files=files,
            output_dir=output_dir,
            lang_list=lang_list,
            backend=backend,
            parse_method=parse_method,
            formula_enable=formula_enable,
            table_enable=table_enable,
            server_url=server_url,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            config=config,
            dump_images=False
        )

        for file_idx, pdf_name in enumerate(pdf_file_names):
            if backend.startswith("pipeline"):
                parse_dir_for_file = os.path.join(batch_unique_dir, pdf_name, parse_method)
            else:
                parse_dir_for_file = os.path.join(batch_unique_dir, pdf_name, "vlm")

            current_file_content_list = load_content_list_from_file(pdf_name, parse_dir_for_file)
            
            if not current_file_content_list:
                logger.warning(f"No content found for file {pdf_name}")
                continue

            current_page_text_buffer = {}

            for content_item in current_file_content_list:
                if not isinstance(content_item, dict):
                    logger.warning(f"Invalid content item: {type(content_item)}")
                    continue
                    
                item_type = content_item.get("type")
                item_page_idx = content_item.get("page_idx", 0) + 1 

                if item_page_idx not in current_page_text_buffer:
                    current_page_text_buffer[item_page_idx] = ""

                if item_type in ["text", "equation"]:
                    text_content = content_item.get("text", "")
                    if text_content.strip():
                        current_page_text_buffer[item_page_idx] += text_content + "\n"
                
                elif item_type == "table":
                    table_caption = " ".join(content_item.get("table_caption", [])).strip()
                    table_body = content_item.get("table_body", "")
                    table_footnote = " ".join(content_item.get("table_footnote", [])).strip()
                    
                    parts = []
                    if table_caption:
                        parts.append(table_caption)
                    if table_body:
                        parts.append(table_body)
                    if table_footnote:
                        parts.append(table_footnote)

                    table_text = "\n".join(parts)
                    if table_text.strip():
                        current_page_text_buffer[item_page_idx] += table_text + "\n"

            for page_idx in sorted(current_page_text_buffer.keys()):
                text_content = current_page_text_buffer[page_idx].strip()
                if text_content:
                    processed_results.append({
                        "content": text_content,
                        "metadata": {
                            "pageStart": page_idx,
                            "pageEnd": page_idx,
                        }
                    })
                    logger.info(f"Added text chunk for page {page_idx}.")

        logger.info(f"All files processed for text-only content. Total chunks: {len(processed_results)}")

        return JSONResponse(
            status_code=200,
            content={
                "minerU_processed_document": processed_results,
            }
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during processing: {str(e)}"
        )
    finally:
        if batch_unique_dir and os.path.exists(batch_unique_dir):
            logger.info(f"Cleaning up temporary directory: {batch_unique_dir}")
            shutil.rmtree(batch_unique_dir, ignore_errors=True)

@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
@click.option('--host', default='0.0.0.0', help='Server host')
@click.option('--port', default=8000, type=int, help='Server port (default: 8000)')
@click.option('--reload', is_flag=True, help='Enable auto-reload (development mode)')
def main(ctx, host, port, reload, **kwargs):
    kwargs.update(arg_parse(ctx))
    app.state.config = kwargs

    print(f"Start MinerU FastAPI Service (development mode): http://{host}:{port}")
    print("The API documentation can be accessed at the following address:")
    print(f"- Swagger UI: http://{host}:{port}/docs")
    print(f"- ReDoc: http://{host}:{port}/redoc")

    uvicorn.run(
        "minerU_2endpoints:app",
        host=host,
        port=port,
        reload=reload,
    )

if __name__ == "__main__":
    main()