# 🚀 SurgiScore: Hackathon Presentation Overview

Here is a structured overview of the SurgiScore project designed specifically to help you build your **Medathon 2026** PowerPoint presentation. This outlines the problem, the solution, the technical architecture, and the core features.

---

## Slide 1: Title Slide
* **Title:** SurgiScore
* **Tagline:** AI-Powered Surgical Performance Analyzer
* **Event:** Medathon 2026
* **Concept:** A computer-vision platform that objectively tracks, scores, and mentors surgical trainees using standard operative video footage.

## Slide 2: The Problem
* **Subjective Assessment:** Surgical training often relies on subjective feedback from senior surgeons, which can be inconsistent and time-consuming.
* **Lack of Data:** Trainees lack objective, data-driven metrics to track their micro-movements (like hand tremors or inefficient paths).
* **Delayed Feedback:** Meaningful clinical feedback is delayed, preventing rapid iterative improvement and extending the learning curve.

## Slide 3: Our Solution (SurgiScore)
* **What it does:** SurgiScore ingests standard surgical video (e.g., arthroscopy) and outputs a comprehensive skill report in minutes.
* **Objective Scoring:** Uses advanced AI to calculate exact mathematical grades for Stability, Efficiency, and Precision.
* **AI Mentor:** Generates immediate, clinical, and highly actionable feedback based on the exact mistakes detected in the video.

## Slide 4: System Architecture & Technologies
Use this slide to flex the technical stack!
* **Frontend:** Clean, responsive UI built with HTML/CSS and Chart.js for data visualization.
* **Backend:** Flask web server handling asynchronous video processing.
* **Computer Vision Pipeline:** 
  * **YOLOv8:** Custom-trained object detection bounding boxes to track the arthroscope and surgical probe in real-time.
  * **Farneback Optical Flow:** Computes relative frame-by-frame instrument displacement.
  * **Smart Jump Filter:** Automatically detects and filters out camera cuts to preserve metric accuracy.
* **Advanced ML (Under the Hood):**
  * **LSTM Networks:** Predicts trajectory to measure controlled vs. erratic movements.
  * **Autoencoders:** Detects anomalies like unexpected jitter.
* **GenAI Integration:** **Anthropic Claude 3.5 Sonnet** analyzes the raw data array to act as a senior surgical mentor.

## Slide 5: The Scoring Algorithm
Explain how you derive the 0-100 grade:
* 🎯 **Stability (40%):** Measures arthroscope steadiness. Evaluated by calculating "jitter" (unexpected pixel jumps between consecutive frames).
* ⚡ **Efficiency (30%):** Measures probe movement control. High efficiency means smooth, deliberate actions rather than erratic searching.
* 📏 **Precision (30%):** Measures the path ratio (the actual distance the probe traveled vs. the ideal straight line). 
* **Baselines:** The system compares these scores against pre-configured baselines for *Beginner*, *Intermediate*, and *Expert* surgeons.

## Slide 6: The Output & User Dashboard
Showcase the UI (Insert screenshots if possible!):
* **Performance Report:** A visual "Skill Radar" and summary gauges displaying the final grade.
* **AI Mentor Feedback:** 3 distinct sentences: 
  1. Identifies the strongest metric.
  2. Pinpoints the weakest metric and the clinical risk it poses (e.g., tissue trauma).
  3. Provides a targeted drill to improve.
* **Tracking Progress:** A historical dashboard showing session-over-session improvement curves.

## Slide 7: Future Roadmap / Impact
* **Clinical Impact:** Shortens the training curve, reduces dependence on senior surgeon availability, and lowers the risk of surgical errors.
* **Next Steps:** Integration with live operating room video feeds for real-time alerts, expanding YOLOv8 to detect anatomical structures (tissues, ligaments), and adding VR integrations.
