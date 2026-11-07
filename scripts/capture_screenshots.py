"""Automated screenshot capture for Telco Customer Churn Intelligence Dashboard.

Uses Playwright to capture genuine high-resolution screenshots from the running
Streamlit application (port 8501) for inclusion in documentation and GitHub presentation.
"""

import time
from pathlib import Path
from playwright.sync_api import sync_playwright

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
SCREENSHOTS_DIR = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def capture_all_screenshots():
    print("Launching Chrome for screenshot capture...")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=CHROME_PATH,
            headless=True,
            args=["--disable-gpu", "--no-sandbox"],
        )
        context = browser.new_context(viewport={"width": 1440, "height": 960}, device_scale_factor=2)
        page = context.new_page()

        # 1. Main Dashboard & Prediction Form
        print("Navigating to http://localhost:8501...")
        page.goto("http://localhost:8501", wait_until="networkidle")
        time.sleep(3)

        shot1_path = SCREENSHOTS_DIR / "01_customer_prediction.png"
        page.screenshot(path=str(shot1_path), full_page=False)
        print(f"Captured: {shot1_path}")

        # 2. Run prediction on the At-Risk persona to capture results & SHAP
        print("Clicking Predict Churn Risk button...")
        try:
            # Click the form submit button
            submit_btn = page.locator('button:has-text("Run Churn Risk Analysis")')
            if submit_btn.count() > 0:
                submit_btn.first.click()
                time.sleep(4)
                shot2_path = SCREENSHOTS_DIR / "02_churn_risk_and_shap.png"
                page.screenshot(path=str(shot2_path), full_page=False)
                print(f"Captured: {shot2_path}")
        except Exception as e:
            print(f"Could not click submit button: {e}")

        # 3. Navigate to Executive Overview
        print("Switching to Executive Overview...")
        try:
            # Radio option for Executive Overview
            page.locator('label:has-text("Executive Overview")').click()
            time.sleep(2)
            shot3_path = SCREENSHOTS_DIR / "03_executive_overview.png"
            page.screenshot(path=str(shot3_path), full_page=False)
            print(f"Captured: {shot3_path}")
        except Exception as e:
            print(f"Could not switch to Executive Overview: {e}")

        # 4. Navigate to Model Insights & Benchmark
        print("Switching to Model Insights & Benchmark...")
        try:
            page.locator('label:has-text("Model Insights")').click()
            time.sleep(2)
            shot4_path = SCREENSHOTS_DIR / "04_model_benchmarks.png"
            page.screenshot(path=str(shot4_path), full_page=False)
            print(f"Captured: {shot4_path}")
        except Exception as e:
            print(f"Could not switch to Model Insights: {e}")

        browser.close()
        print("Screenshot capture sequence finished successfully!")


if __name__ == "__main__":
    capture_all_screenshots()
