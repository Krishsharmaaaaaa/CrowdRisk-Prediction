import requests
import time
import sys

url = 'https://crowdrisk-api.onrender.com/api/analyze'
print('Uploading demo.mp4...', flush=True)
with open('data/videos/demo.mp4', 'rb') as f:
    r = requests.post(url, files={'file': ('demo.mp4', f, 'video/mp4')}, timeout=30)

print(f'SUBMIT STATUS: {r.status_code}', flush=True)
data = r.json()
print(f'RESPONSE: {data}', flush=True)
job_id = data.get('analysis_id')

if job_id:
    start = time.time()
    for i in range(80):
        try:
            res = requests.get(f'https://crowdrisk-api.onrender.com/api/analysis/{job_id}', timeout=10)
            st = res.json()
            elapsed = time.time() - start
            status = st.get('status')
            progress = st.get('progress')
            stage = st.get('current_stage')
            print(f'[{elapsed:.1f}s] status={status} progress={progress}% stage={stage}', flush=True)
            if status in ('completed', 'failed'):
                print(f'FINAL SUMMARY: {st.get("summary")}', flush=True)
                print(f'ERROR: {st.get("error_message")}', flush=True)
                break
        except Exception as e:
            print(f'Poll error: {e}', flush=True)
        time.sleep(4)
