import requests
import re

r = requests.get('https://crowd-risk-prediction.vercel.app', timeout=10)
print('VERCEL HTML STATUS:', r.status_code)
print('HTML SNIPPET:\n', r.text[:500])

scripts = re.findall(r'src=["\']([^"\']+\.js)["\']', r.text)
print('SCRIPTS FOUND:', scripts)

for s in scripts:
    s_url = 'https://crowd-risk-prediction.vercel.app' + s if s.startswith('/') else s
    res = requests.get(s_url, timeout=10)
    js = res.text
    print(f'\n--- Script {s_url} (size: {len(js)} bytes) ---')
    
    # Check for localhost
    lh_matches = re.findall(r'https?://localhost:[0-9]+', js)
    print('  Localhost URLs:', set(lh_matches))
    
    # Check for render
    render_matches = re.findall(r'https?://[a-zA-Z0-9.-]+\.onrender\.com', js)
    print('  Render URLs:', set(render_matches))

    # Check for API_BASE or fetch URLs
    api_matches = re.findall(r'/api/[a-zA-Z0-9_/-]+', js)
    print('  /api routes found in bundle:', set(api_matches))

    # Find the exact Me or API_BASE assignment
    idx = js.find('crowdrisk-api')
    if idx != -1:
        print('  Snippet around crowdrisk-api:\n', js[max(0, idx-80):min(len(js), idx+120)])
    else:
        print('  WARNING: crowdrisk-api NOT FOUND IN DEPLOYED JS BUNDLE!')
