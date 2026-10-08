"""
Render deployment entry point.
Run: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
"""
# This file is not used directly — render.yaml specifies the start command.
# It is provided as a local convenience entry point.

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port, reload=False)
