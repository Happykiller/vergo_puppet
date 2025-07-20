# app/apis/o6.py
from pathlib import Path
from datetime import datetime
from multiprocessing import Process
from pydantic import BaseModel # type: ignore
from fastapi import APIRouter, Depends, HTTPException # type: ignore

from app.services.logger import logger
from app.apis.common import format_duration
from app.apis.deps import verify_access_token
from app.usecases.usecase_summarize_mr import usecase_summarize_mr

o6_router = APIRouter()

class SummarizeMRData(BaseModel):
    prompt_path: str
    markdown_path: str


def summarize_mr_worker(prompt_path: str, markdown_path: str, output_path: str):
    """
    Process worker to summarize a MR and write to a file.
    """
    from time import perf_counter
    from traceback import format_exc

    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"[summarize_mr_worker] ⏳ Starting summary for: {markdown_path}")
        start_time = perf_counter()

        summary = usecase_summarize_mr(prompt_path, markdown_path)

        duration = perf_counter() - start_time
        Path(output_path).write_text(summary, encoding="utf-8")
        logger.info(f"[summarize_mr_worker] ✅ Summary written to {output_path} in {format_duration(duration)}")
    except Exception as e:
        logger.error(f"[summarize_mr_worker] ❌ Exception: {e}\n{format_exc()}")
        Path(output_path).write_text(f"[ERROR] {str(e)}", encoding="utf-8")


@o6_router.post("/o6")
async def summarize_mr(data: SummarizeMRData, payload: dict = Depends(verify_access_token)):
    """
    Launches a background summarization of a Merge Request.
    """
    try:
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        output_path = f"output/o6_{timestamp}.md"

        p = Process(target=summarize_mr_worker, args=(
            data.prompt_path,
            data.markdown_path,
            output_path
        ))
        p.start()

        return {
            "status": "started",
            "output_path": output_path
        }
    except Exception as e:
        logger.error(f"Failed to start summarization process: {e}")
        raise HTTPException(status_code=500, detail="Unable to start process.")