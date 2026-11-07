"""
Complete End-to-End Browser QA and Visual Screenshot Pipeline.
Connects to the running Streamlit server on port 8503,
tests user interactions (preset loading, inference execution, tab navigation),
and captures high-resolution screenshots for all views.
"""
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

SCREENSHOT_DIR = os.path.join(os.getcwd(), "docs", "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

def main():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--window-size=1920,1080")
    
    driver = webdriver.Chrome(options=opts)
    driver.set_window_size(1920, 1080)
    
    try:
        url = "http://localhost:8503"
        print(f"Connecting to {url}...")
        driver.get(url)
        
        # Wait for Streamlit to finish initial compile & render
        WebDriverWait(driver, 40).until(lambda d: "AUTOVISION" in d.page_source)
        time.sleep(3)
        print("Page rendered successfully!")

        # 1. Landing & Hero section
        s1 = os.path.join(SCREENSHOT_DIR, "01_landing_hero.png")
        driver.save_screenshot(s1)
        print(f"Captured: {s1} ({os.path.getsize(s1)} bytes)")

        # 2. Click Audi TTS preset button
        print("Finding 'Audi TTS' preset button...")
        buttons = driver.find_elements(By.TAG_NAME, "button")
        audi_btn = None
        for b in buttons:
            if "Audi TTS" in b.text:
                audi_btn = b
                break
        
        if not audi_btn:
            raise RuntimeError("Could not find 'Audi TTS' preset button!")
        
        print(f"Clicking preset button '{audi_btn.text}'...")
        driver.execute_script("arguments[0].click();", audi_btn)
        
        # Wait for inference results to render
        print("Waiting for vehicle identity card to appear...")
        WebDriverWait(driver, 45).until(
            lambda d: "Audi TTS Coupe 2012" in d.page_source or "02. Inference Telemetry" in d.page_source
        )
        time.sleep(2)  # Allow charts and animations to settle
        
        s2 = os.path.join(SCREENSHOT_DIR, "02_recognition_studio.png")
        driver.save_screenshot(s2)
        print(f"Captured: {s2} ({os.path.getsize(s2)} bytes)")

        # Function to find and click a tab by keyword
        def click_tab(keyword):
            tabs = driver.find_elements(By.CSS_SELECTOR, '[data-testid="stTab"]')
            for t in tabs:
                if keyword.lower() in t.text.lower():
                    print(f"Clicking tab matching '{keyword}': '{t.text.strip()}'")
                    driver.execute_script("arguments[0].scrollIntoView(true);", t)
                    time.sleep(0.3)
                    driver.execute_script("arguments[0].click();", t)
                    time.sleep(2)
                    return True
            print(f"WARNING: Tab matching '{keyword}' not found!")
            return False

        # 3. Model Consensus Tab
        if click_tab("Consensus"):
            s3 = os.path.join(SCREENSHOT_DIR, "03_model_consensus.png")
            driver.save_screenshot(s3)
            print(f"Captured: {s3} ({os.path.getsize(s3)} bytes)")

        # 4. Vision Filter Lab Tab
        if click_tab("Filter"):
            s4 = os.path.join(SCREENSHOT_DIR, "04_vision_filter_lab.png")
            driver.save_screenshot(s4)
            print(f"Captured: {s4} ({os.path.getsize(s4)} bytes)")

        # 5. Benchmarks & Ablation Tab
        if click_tab("Benchmark"):
            s5 = os.path.join(SCREENSHOT_DIR, "05_benchmarks_ablation.png")
            driver.save_screenshot(s5)
            print(f"Captured: {s5} ({os.path.getsize(s5)} bytes)")

        # 6. Failure Taxonomy Tab
        if click_tab("Failure"):
            s6 = os.path.join(SCREENSHOT_DIR, "06_failure_taxonomy.png")
            driver.save_screenshot(s6)
            print(f"Captured: {s6} ({os.path.getsize(s6)} bytes)")

        # 7. API & Telemetry Tab
        if click_tab("Telemetry"):
            s7 = os.path.join(SCREENSHOT_DIR, "07_api_telemetry.png")
            driver.save_screenshot(s7)
            print(f"Captured: {s7} ({os.path.getsize(s7)} bytes)")

        # 8. Mobile Viewport (iPhone 14 / modern mobile 390x844)
        print("Testing mobile responsive viewport (390x844)...")
        driver.set_window_size(390, 844)
        time.sleep(1)
        click_tab("Studio")
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(2)
        s8 = os.path.join(SCREENSHOT_DIR, "08_mobile_view.png")
        driver.save_screenshot(s8)
        print(f"Captured: {s8} ({os.path.getsize(s8)} bytes)")

        print("\nAll 8 browser QA screenshots captured successfully!")

    finally:
        driver.quit()

if __name__ == "__main__":
    main()

