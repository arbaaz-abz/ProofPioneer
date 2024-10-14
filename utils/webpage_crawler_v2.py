import argparse
import os
from time import sleep
import trafilatura
from trafilatura.settings import DEFAULT_CONFIG
from trafilatura.meta import reset_caches
import sys
import json

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Increase max file size to 50MB
DEFAULT_CONFIG.MAX_FILE_SIZE = 50000000

def initialize_webdriver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode.
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        # Prevent detection
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {
            "userAgent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                         'AppleWebKit/537.36 (KHTML, like Gecko) '
                         'Chrome/129.0.0.0 Safari/537.36'
        })
    except WebDriverException as e:
        print(f"Error initializing WebDriver: {e}", file=sys.stderr)
        driver = None
    return driver


def handle_cookie_popup(driver, timeout=5):
    try:
        # Common button texts; extend this list as needed
        button_texts = ['Accept', 'I Agree', 'Agree', 'Consent', 'Allow All']
        for text in button_texts:
            try:
                cookie_button = WebDriverWait(driver, timeout).until(
                    EC.element_to_be_clickable((By.XPATH, f"//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{text.lower()}')]"))
                )
                cookie_button.click()
                print(f"Clicked '{text}' button for cookie consent.", file=sys.stderr)
                return True
            except TimeoutException:
                continue
        print("No cookie popup detected or unable to handle.", file=sys.stderr)
    except Exception as e:
        print(f"Error handling cookie popup: {e}", file=sys.stderr)
    return False


def get_page_with_selenium(url, timeout=10):
    driver = initialize_webdriver()
    if not driver:
        return None
    driver.implicitly_wait(5)
    
    try:
        driver.get(url)
        
        # Wait for the page to load
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Handle cookie popup if present
        handle_cookie_popup(driver, timeout=5)
        
        # Optional: Wait for additional dynamic content
        sleep(1)  # Adjust as necessary
        
        page_source = driver.page_source
    except Exception as e:
        print(f"Selenium failed for {url}: {e}", file=sys.stderr)
        page_source = None
    finally:
        driver.quit()
    
    return page_source

def get_page(url):
    page = None
    
    # First attempt with trafilatura
    for i in range(2):
        try:
            page = trafilatura.fetch_url(url, config=DEFAULT_CONFIG)
            assert page is not None
            # print(f"Successfully fetched {url} with trafilatura.", file=sys.stderr)
            break
        except Exception as e:
            print(f"Attempt {i+1} with trafilatura failed for {url}: {str(e)}", file=sys.stderr)
            sleep(3)
    
    # If trafilatura fails, try Selenium
    if page is None:
        print(f"Falling back to Selenium for {url}", file=sys.stderr)
        page = get_page_with_selenium(url)
    
    return page

def html2lines(page):
    if not page or len(page.strip()) == 0:
        print("No page found")
        return {}

    text = trafilatura.extract(page, 
                               favor_recall=True, 
                               with_metadata = True,    
                               no_fallback=False,
                               output_format="json"
                            )
    try:
        return json.loads(text)
    except Exception as e:
        print(e)
        return {}
    # Disabling increases performance but may lead to memory leaks in some cases, particularly in large-scale applications
    # reset_caches() 

def url2lines(url):
    page = get_page(url)
    return html2lines(page)

# def save_content(url, output_dir):
#     lines = url2lines(url)
#     if not lines:
#         print(f"No content extracted from {url}", file=sys.stderr)
#         return

#     # Create a filename from the URL
#     filename = url.split("//")[-1].replace("/", "_")
#     filename = os.path.join(output_dir, f"{filename}.txt")

#     with open(filename, "w", encoding="utf-8") as f:
#         f.write(url + "\n")  # Write URL as the first line
#         f.write("\n".join(lines))

#     print(f"Content saved to {filename}", file=sys.stderr)

# if __name__ == "__main__":
#     test_urls = [
#         "https://www.trailrunnermag.com/races/the-worlds-most-amazing-race/",
#         "https://dictionary.cambridge.org/us/dictionary/portuguese-english/nadar",
#         "https://www.nature.com/articles/nature.2015.17802",
#         "https://www.bls.gov/opub/reports/womens-databook/2020/"
#     ]

#     l = []
#     for url in test_urls:
#         lines = url2lines(url)
#         l.append(lines)
#         # Process the lines as needed