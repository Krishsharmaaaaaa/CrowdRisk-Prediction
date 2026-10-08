import requests
import time

origin = 'https://crowd-risk-prediction.vercel.app'
for i in range(20):
    try:
        r = requests.get('https://crowdrisk-api.onrender.com/api/demo/umn_indoor_clip4', headers={'Origin': origin}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            tl_len = len(data.get('timeline', []))
            crda_len = len(data.get('crda', []))
            print(f'Attempt {i+1}: STATUS {r.status_code}, Timeline={tl_len}, CRDA={crda_len}', flush=True)
            if tl_len > 0:
                print('SUCCESS: Rich demo data is live on Render!', flush=True)
                break
        else:
            print(f'Attempt {i+1}: STATUS {r.status_code}', flush=True)
    except Exception as e:
        print(f'Attempt {i+1}: ERROR {e}', flush=True)
    time.sleep(5)
