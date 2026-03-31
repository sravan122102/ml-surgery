"""
test_pipeline.py — End-to-End Pipeline Test for SurgiScore
============================================================
Tests the full flow: server boot → upload → analysis → results.

Usage:
    1. Start the server:  python app.py
    2. In another terminal: python test_pipeline.py

Tests:
    1. Health check endpoint
    2. Index page loads
    3. Single-video upload → poll → results
    4. Multi-video upload → poll → results
    5. Report image generation
"""

import os
import sys
import time
import json
import requests

BASE_URL = "http://127.0.0.1:5000"
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")

# Use the smallest video file available for fast testing
def find_test_video():
    """Find the smallest video in uploads/ for testing."""
    if not os.path.exists(UPLOAD_DIR):
        return None
    videos = [
        f for f in os.listdir(UPLOAD_DIR)
        if f.lower().endswith(('.mp4', '.avi', '.mov'))
    ]
    if not videos:
        return None
    # Sort by size, pick smallest
    videos.sort(key=lambda f: os.path.getsize(os.path.join(UPLOAD_DIR, f)))
    return os.path.join(UPLOAD_DIR, videos[0])


def test_health():
    """Test 1: Health check endpoint."""
    print("\n[TEST 1] Health Check...")
    try:
        r = requests.get(f"{BASE_URL}/api/health", timeout=5)
        assert r.status_code == 200, f"Status {r.status_code}"
        data = r.json()
        assert data["status"] == "ok"
        print(f"  ✅ PASS — YOLO: {data['yolo_model']}, Active jobs: {data['active_jobs']}")
        return True
    except requests.ConnectionError:
        print("  ❌ FAIL — Server not running. Start with: python app.py")
        return False
    except Exception as e:
        print(f"  ❌ FAIL — {e}")
        return False


def test_index():
    """Test 2: Index page loads."""
    print("\n[TEST 2] Index Page...")
    try:
        r = requests.get(f"{BASE_URL}/", timeout=5)
        assert r.status_code == 200, f"Status {r.status_code}"
        assert "SurgiScore" in r.text or "surgiscore" in r.text.lower(), "Page doesn't contain SurgiScore"
        print(f"  ✅ PASS — Index loaded ({len(r.text)} bytes)")
        return True
    except Exception as e:
        print(f"  ❌ FAIL — {e}")
        return False


def test_single_upload(video_path):
    """Test 3: Single-video upload → analysis → results."""
    print("\n[TEST 3] Single-Video Upload...")
    try:
        # Upload
        with open(video_path, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/upload",
                files={"video": (os.path.basename(video_path), f, "video/mp4")},
                data={"name": "PipelineTest"},
                timeout=30,
            )
        assert r.status_code == 200, f"Upload failed: {r.status_code} — {r.text}"
        job_id = r.json()["job_id"]
        print(f"  📤 Uploaded — job_id: {job_id}")

        # Poll status
        max_wait = 300  # 5 minutes max
        start = time.time()
        while time.time() - start < max_wait:
            r = requests.get(f"{BASE_URL}/status/{job_id}", timeout=5)
            data = r.json()
            status = data["status"]

            if status == "done":
                result = data.get("result", {})
                print(f"  ✅ Analysis complete in {time.time()-start:.0f}s")
                print(f"     Overall: {result.get('overall', '?')}/100")
                print(f"     Grade: {result.get('grade', '?')}")
                print(f"     Tracking: {result.get('tracking_method', '?')}")
                print(f"     AI Comment: {result.get('ai_comment', '?')[:80]}...")

                # Verify all expected fields
                expected = [
                    "stability", "efficiency", "precision", "smoothness",
                    "overall", "grade", "frames", "ai_comment", "tracking_method",
                ]
                missing = [k for k in expected if k not in result]
                if missing:
                    print(f"  ⚠️  Missing fields: {missing}")
                else:
                    print(f"  ✅ All expected fields present")

                # Check report
                report_img = result.get("report_image", "")
                if report_img:
                    report_path = os.path.join("results", report_img)
                    if os.path.exists(report_path):
                        size_kb = os.path.getsize(report_path) / 1024
                        print(f"  ✅ Report PNG generated: {report_img} ({size_kb:.0f} KB)")
                    else:
                        print(f"  ⚠️  Report listed but file not found: {report_path}")

                # Check results page
                r2 = requests.get(f"{BASE_URL}/results/{job_id}", timeout=5)
                if r2.status_code == 200:
                    print(f"  ✅ Results page renders ({len(r2.text)} bytes)")
                else:
                    print(f"  ⚠️  Results page status: {r2.status_code}")

                return True

            elif status == "error":
                print(f"  ❌ Analysis error: {data.get('error', 'unknown')}")
                return False

            else:
                elapsed = time.time() - start
                print(f"  ⏳ {status}... ({elapsed:.0f}s)", end="\r")
                time.sleep(3)

        print(f"\n  ❌ Timed out after {max_wait}s")
        return False

    except Exception as e:
        print(f"  ❌ FAIL — {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_upload(video_path):
    """Test 4: Multi-video upload (same video twice, simulates 2 cameras)."""
    print("\n[TEST 4] Multi-Video Upload...")
    try:
        with open(video_path, "rb") as f1, open(video_path, "rb") as f2:
            r = requests.post(
                f"{BASE_URL}/upload-multi",
                files=[
                    ("videos", (f"cam1_{os.path.basename(video_path)}", f1, "video/mp4")),
                    ("videos", (f"cam2_{os.path.basename(video_path)}", f2, "video/mp4")),
                ],
                data={"name": "MultiViewTest"},
                timeout=30,
            )
        assert r.status_code == 200, f"Upload failed: {r.status_code} — {r.text}"
        data = r.json()
        job_id = data["job_id"]
        cameras = data.get("cameras", [])
        print(f"  📤 Uploaded — job_id: {job_id}, cameras: {cameras}")

        # Poll status
        max_wait = 600  # 10 minutes for multi-view
        start = time.time()
        while time.time() - start < max_wait:
            r = requests.get(f"{BASE_URL}/status/{job_id}", timeout=5)
            sdata = r.json()
            status = sdata["status"]

            if status == "done":
                result = sdata.get("result", {})
                print(f"  ✅ Multi-view analysis complete in {time.time()-start:.0f}s")
                print(f"     Overall: {result.get('overall', '?')}/100")
                print(f"     Grade: {result.get('grade', '?')}")
                print(f"     Tracking: {result.get('tracking_method', '?')}")
                print(f"     Cameras: {result.get('cameras_used', '?')}")
                print(f"     Triangulated: {result.get('triangulated_frames', '?')} frames")

                # Check dashboard route
                r2 = requests.get(f"{BASE_URL}/dashboard/{job_id}", timeout=5)
                if r2.status_code == 200:
                    print(f"  ✅ Dashboard page renders ({len(r2.text)} bytes)")
                else:
                    print(f"  ⚠️  Dashboard page status: {r2.status_code}")

                return True

            elif status == "error":
                print(f"  ❌ Analysis error: {sdata.get('error', 'unknown')}")
                return False

            else:
                elapsed = time.time() - start
                print(f"  ⏳ {status}... ({elapsed:.0f}s)", end="\r")
                time.sleep(5)

        print(f"\n  ❌ Timed out after {max_wait}s")
        return False

    except Exception as e:
        print(f"  ❌ FAIL — {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("SurgiScore — End-to-End Pipeline Test")
    print("=" * 60)

    results = {}

    # Test 1: Health
    results["health"] = test_health()
    if not results["health"]:
        print("\n⛔ Server not running. Please start: python app.py")
        sys.exit(1)

    # Test 2: Index
    results["index"] = test_index()

    # Find test video
    video = find_test_video()
    if not video:
        print("\n⚠️  No test videos found in uploads/. Skipping upload tests.")
        results["single_upload"] = False
        results["multi_upload"] = False
    else:
        size_mb = os.path.getsize(video) / (1024 * 1024)
        print(f"\n📁 Test video: {os.path.basename(video)} ({size_mb:.1f} MB)")

        # Test 3: Single upload
        results["single_upload"] = test_single_upload(video)

        # Test 4: Multi upload (only if single passed)
        if results["single_upload"]:
            results["multi_upload"] = test_multi_upload(video)
        else:
            print("\n[TEST 4] Skipped — single upload failed")
            results["multi_upload"] = False

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for test, result in results.items():
        icon = "✅" if result else "❌"
        print(f"  {icon} {test}")
    print(f"\n  {passed}/{total} tests passed")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
