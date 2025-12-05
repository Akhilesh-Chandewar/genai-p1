from fastapi import FastAPI, Query
from client.rq_client import queue
from queues.worker import process_query

app = FastAPI()

@app.get("/")
def read_root():
    return {"Message": "Server is running."}

@app.post("/chat/")
def chat(query: str = Query(..., description="The user query to process.")):
    job = queue.enqueue(process_query, query)
    return {"job_id": job.id, "status": "Job enqueued."}

@app.get("/result/")
def get_result(job_id: str = Query(..., description="The job ID to fetch the result for.")):
    job = queue.fetch_job(job_id)
    if job is None:
        return {"error": "Job not found."}
    if job.is_finished:
        return {"result": job.result}
    elif job.is_failed:
        return {"error": "Job failed."}
    else:
        return {"status": "Job is still processing."} 