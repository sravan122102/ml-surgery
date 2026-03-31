"""Quick status check for running jobs."""
import requests
import json
import time
import sys

BASE = "http://127.0.0.1:5000"

# Health check
print("=== Health Check ===")
r = requests.get(f"{BASE}/api/health", timeout=5)
health = r.json()
print(json.dumps(health, indent=2))

# Check if previous job exists
print("\n=== Checking job aae783e4 ===")
r = requests.get(f"{BASE}/status/aae783e4", timeout=5)
data = r.json()
print(f"Status: {data.get('status')}")
if data.get("status") == "done":
    result = data.get("result", {})
    print(f"Overall: {result.get('overall')}/100")
    print(f"Grade: {result.get('grade')}")
    print(f"Tracking: {result.get('tracking_method')}")
    print(f"AI: {result.get('ai_comment', '')[:100]}...")
elif data.get("status") == "processing":
    print("Still processing... waiting 30s")
    for i in range(6):
        time.sleep(5)
        r = requests.get(f"{BASE}/status/aae783e4", timeout=5)
        data = r.json()
        print(f"  ...{data.get('status')} ({(i+1)*5}s)")
        if data.get("status") in ("done", "error"):
            if data.get("status") == "done":
                result = data.get("result", {})
                print(f"  Overall: {result.get('overall')}/100")
                print(f"  Grade: {result.get('grade')}")
                print(f"  Tracking: {result.get('tracking_method')}")
                print(f"  AI: {result.get('ai_comment', '')[:100]}...")
            else:
                print(f"  Error: {data.get('error')}")
            break
elif data.get("status") == "error":
    print(f"Error: {data.get('error')}")
else:
    print(f"Response: {json.dumps(data, indent=2)}")

# Now upload a small test video
print("\n=== Upload Test (smallest video) ===")
import os
uploads = "uploads"
videos = sorted(
    [f for f in os.listdir(uploads) if f.endswith(".mp4")],
    key=lambda f: os.path.getsize(os.path.join(uploads, f))
)
if videos:
    vpath = os.path.join(uploads, videos[0])
    vsize = os.path.getsize(vpath) / (1024*1024)
    print(f"Using: {videos[0]} ({vsize:.1f} MB)")
    
    with open(vpath, "rb") as f:
        r = requests.post(
            f"{BASE}/upload",
            files={"video": (videos[0], f, "video/mp4")},
            data={"name": "QuickTest"},
            timeout=30,
        )
    job_id = r.json().get("job_id")
    print(f"Job ID: {job_id}")
    
    # Poll
    start = time.time()
    while time.time() - start < 180:
        r = requests.get(f"{BASE}/status/{job_id}", timeout=5)
        data = r.json()
        elapsed = time.time() - start
        if data["status"] == "done":
            result = data.get("result", {})
            print(f"\n✅ Done in {elapsed:.0f}s!")
            print(f"  Overall: {result.get('overall')}/100")
            print(f"  Grade: {result.get('grade')}")
            print(f"  Stability: {result.get('stability')}")
            print(f"  Efficiency: {result.get('efficiency')}")
            print(f"  Precision: {result.get('precision')}")
            print(f"  Smoothness: {result.get('smoothness')}")
            print(f"  Tracking: {result.get('tracking_method')}")
            print(f"  Frames: {result.get('frames')}")
            print(f"  AI: {result.get('ai_comment', '')[:120]}...")
            
            # Check report
            ri = result.get("report_image", "")
            if ri and os.path.exists(os.path.join("results", ri)):
                sz = os.path.getsize(os.path.join("results", ri)) / 1024
                print(f"  Report: {ri} ({sz:.0f} KB) ✅")
            
            # Check results page
            r2 = requests.get(f"{BASE}/results/{job_id}", timeout=5)
            print(f"  Results page: HTTP {r2.status_code} ({len(r2.text)} bytes) ✅")
            break
        elif data["status"] == "error":
            print(f"\n❌ Error: {data.get('error')}")
            break
        else:
            print(f"  ⏳ {data['status']}... ({elapsed:.0f}s)", end="\r")
            time.sleep(3)
    else:
        print("\n❌ Timed out")
else:
    print("No videos found in uploads/")
