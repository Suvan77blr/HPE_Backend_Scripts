from typing import Optional, List, Dict, Any 
import requests
from bs4 import BeautifulSoup
import time
import logging
from urllib.parse import urljoin
import os
import io
import PyPDF2
import re
from diskcache import Cache 

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.service import Service as ChromeService 
from webdriver_manager.chrome import ChromeDriverManager 
import random 

logger = logging.getLogger(__name__)

class NetworkDocScraper:
    def __init__(self, cache_dir: Optional[str] = None, selenium_timeout: int = 30, default_scrape_delay: float = 1.5):
        if cache_dir is None:
            cache_dir = os.path.join(os.getcwd(), "scraper_cache")
        os.makedirs(cache_dir, exist_ok=True)
        try:
            self.cache = Cache(cache_dir)
            logger.info(f"NetworkDocScraper initialized. Diskcache active at: {cache_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize DiskCache at {cache_dir}: {e}. Caching will be disabled.")
            self.cache = None
        
        self.user_agents = [ 
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0'
        ]
        self.selenium_timeout = selenium_timeout
        self.default_scrape_delay = default_scrape_delay
        self.driver = None 
        self._get_selenium_driver() 

    def _get_selenium_driver(self) -> Optional[webdriver.Chrome]:
        if self.driver is None:
            try:
                chrome_options = ChromeOptions()
                chrome_options.add_argument("--headless=new") 
                chrome_options.add_argument("--disable-gpu")
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument("--window-size=1920,1080")
                chrome_options.add_argument("--ignore-certificate-errors")
                chrome_options.add_argument("--ignore-ssl-errors")
                chrome_options.add_argument(f"user-agent={random.choice(self.user_agents)}")
                
                self.driver = webdriver.Chrome(
                    service=ChromeService(ChromeDriverManager().install()),
                    options=chrome_options
                )
                logger.info("Selenium WebDriver initialized via webdriver_manager.")
            except Exception as e:
                logger.error(f"Failed to initialize Selenium WebDriver: {e}", exc_info=True)
                self.driver = None 
        return self.driver





    def _quit_selenium_driver(self):
        if self.driver:
            try:
                self.driver.quit()
                logger.info("Selenium WebDriver quit.")
            except Exception as e:
                logger.error(f"Error quitting Selenium WebDriver: {e}")
            finally:
                self.driver = None




    def __del__(self):
        if self.cache:
            try: self.cache.close()
            except: pass 
        self._quit_selenium_driver()

    def get_page(self, url: str, use_selenium_override: Optional[bool] = None, force_live: bool = False) -> Optional[str]:
        # (Code from response #35 - remains the same)
        time.sleep(self.default_scrape_delay + random.uniform(0.2, 0.8))
        cache_key = f"page_html:{url}"
        if not force_live and self.cache:
            cached_content = self.cache.get(cache_key)
            if cached_content is not None:
                logger.debug(f"HTML Cache hit for URL: {url}")
                return cached_content
            logger.debug(f"HTML Cache miss for URL: {url}")
        html_content = None
        should_use_selenium = use_selenium_override if use_selenium_override is not None else (self._get_selenium_driver() is not None)
        if should_use_selenium:
            driver = self._get_selenium_driver()
            if driver:
                try:
                    logger.info(f"Fetching with Selenium: {url}")
                    driver.get(url)
                    WebDriverWait(driver, self.selenium_timeout).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                    html_content = driver.page_source
                    logger.info(f"Successfully fetched with Selenium: {url}")
                except Exception as e_sel:
                    logger.warning(f"Selenium fetch failed for {url}: {e_sel}. Content remains None.")
                    html_content = None
            else:
                logger.warning(f"Selenium driver not available for {url}, requested Selenium. Content remains None.")
                html_content = None
        if html_content is None and (use_selenium_override is False or not should_use_selenium) : 
            try:
                logger.info(f"Fetching with Requests: {url}")
                session = requests.Session() 
                session.headers.update({"User-Agent": random.choice(self.user_agents)})
                response = session.get(url, timeout=20)
                response.raise_for_status()
                html_content = response.text
                logger.info(f"Successfully fetched with Requests: {url}")
            except requests.exceptions.RequestException as e_req:
                logger.error(f"Requests fetch failed for {url}: {e_req}")
                html_content = None
        if html_content is not None and self.cache and not force_live:
            self.cache.set(cache_key, html_content, expire=3600*24) # TTL for HTML pages
            logger.debug(f"Cached HTML content for URL: {url}")
        return html_content
    
    def process_pdf_using_selenium_session(self, url: str) -> Optional[bytes]:
        # (Code from response #35 - remains the same)
        driver = self._get_selenium_driver()
        if not driver:
            logger.error(f"Selenium driver not available for PDF processing of {url}.")
            return None
        try:
            logger.debug(f"Selenium navigating to establish session context for PDF: {url}")
            driver.get(url) 
            time.sleep(random.uniform(1.5, 3.0)) 
            selenium_cookies = driver.get_cookies()
            req_session = requests.Session()
            for cookie in selenium_cookies:
                req_session.cookies.set(cookie['name'], cookie['value'])
            req_headers = {'User-Agent': random.choice(self.user_agents), 'Accept': 'application/pdf,application/octet-stream,*/*;q=0.8', 'Accept-Language': 'en-US,en;q=0.5', 'Referer': driver.current_url }
            logger.info(f"Streaming PDF from: {url} (using Selenium-derived session)")
            response = req_session.get(url, headers=req_headers, timeout=90, stream=True)
            response.raise_for_status()
            if 'application/pdf' in response.headers.get('Content-Type', '').lower():
                logger.info(f"Successfully streamed PDF bytes from: {url}")
                return response.content
            else:
                logger.error(f"Content from {url} is not a PDF (Content-Type: {response.headers.get('Content-Type')}).")
                return None
        except Exception as e:
            logger.error(f"Error processing PDF {url} via Selenium session: {e}", exc_info=True)
            return None

    def extract_text_from_pdf_bytes(self, pdf_bytes: bytes, source_url_for_logging: str) -> Optional[str]:
        # (Code from response #35 - remains the same)
        try:
            pdf_stream = io.BytesIO(pdf_bytes)
            reader = PyPDF2.PdfReader(pdf_stream)
            text_parts = []
            if reader.is_encrypted:
                try: reader.decrypt('')
                except: logger.warning(f"Encrypted PDF {source_url_for_logging} could not be decrypted."); return None
            for i, page in enumerate(reader.pages):
                try: 
                    page_text = page.extract_text()
                    if page_text: text_parts.append(page_text)
                except Exception as e_page_extract: 
                    logger.warning(f"Could not extract text from page {i+1} of PDF {source_url_for_logging}: {e_page_extract}")
            return "\n\n".join(text_parts) if text_parts else None
        except Exception as e:
            logger.error(f"Error extracting text from PDF bytes (source: {source_url_for_logging}): {e}", exc_info=True)
            return None


    

    def get_pdf_text_content(self, url: str, doc_meta: Optional[Dict[str, Any]] = None, force_live: bool = False) -> Optional[str]:
        # (Code from response #35 - remains the same)
        if doc_meta is None: doc_meta = {}
        cache_key_pdf_text = f"pdf_content:{url}"
        if not force_live and self.cache:
            cached_text = self.cache.get(cache_key_pdf_text)
            if cached_text is not None:
                logger.debug(f"PDF text cache hit for: {url}")
                return cached_text
            logger.debug(f"PDF text cache miss for: {url}")
        pdf_bytes = self.process_pdf_using_selenium_session(url)
        if not pdf_bytes:
            logger.warning(f"Failed to get PDF bytes for {url} using Selenium session approach.")
            return None
        text = self.extract_text_from_pdf_bytes(pdf_bytes, url)
        if text and self.cache and not force_live: 
            self.cache.set(cache_key_pdf_text, text, expire=3600*24*7) # Default 7-day TTL for PDF text
            logger.debug(f"Cached PDF text from {url}")
        elif not text:
            logger.warning(f"No text extracted from PDF bytes: {url}. Not caching.")
        return text


    def extract_document_content(self, doc_meta: Dict[str, Any], force_live: bool = False) -> Optional[str]:
        # (Code from response #35 - remains the same)
        url = doc_meta.get("url")
        if not url:
            logger.warning(f"Missing URL in doc_meta for content extraction: {doc_meta}")
            return None
        doc_type = doc_meta.get("doc_type", "").lower()
        if not doc_type and str(url).lower().endswith(".pdf"): 
             doc_type = "pdf"
        time.sleep(self.default_scrape_delay / 2 + random.uniform(0.1, 0.3))
        if doc_type == "pdf":
            logger.info(f"Extracting content from PDF: {url} (force_live={force_live})")
            return self.get_pdf_text_content(url, doc_meta, force_live=force_live)
        else: 
            logger.info(f"Extracting content from HTML: {url} (force_live={force_live})")
            html_content = self.get_page(url, use_selenium_override=True, force_live=force_live)
            if not html_content: return None
            soup = BeautifulSoup(html_content, 'html.parser')
            for element in soup.select('nav, header, footer, script, style, aside, .sidebar, #sidebar, .noprint, .cookie-banner, #onetrust-consent-sdk'):
                element.decompose()
            main_content_selectors = ['main', 'article', 'div[role="main"]', 'div.main-content', 'div#main-content', 'div.content', 'div#content', 'body']
            main_content = None
            for selector in main_content_selectors:
                main_content = soup.select_one(selector)
                if main_content: break
            return main_content.get_text(separator='\n', strip=True) if main_content else None
    

    def parse_juniper_sitemap_xml(self, sitemap_url: str) -> List[Dict[str, Any]]:
        logger.info(f"Parsing Juniper XML Sitemap: {sitemap_url}")
        # Use Selenium for initial fetch in case of any JS challenges, though XML should be static
        # force_live=True to always get the latest sitemap
        xml_content = self.get_page(sitemap_url, use_selenium_override=False, force_live=True) 
        if not xml_content:
            logger.error(f"Failed to fetch Juniper sitemap XML from {sitemap_url}")
            return []

        doc_links_metadata = []
        try:
            # Ensure BeautifulSoup is parsing as XML
            soup = BeautifulSoup(xml_content, 'xml') # Use 'xml' parser
            
            # Sitemaps usually have <url><loc>URL</loc>...</url>
            # Or <sitemap><loc>URL_to_another_sitemap.xml</loc></sitemap>
            # For now, assuming it's a direct list of content URLs.
            
            # Check for nested sitemaps first (sitemap index file)
            sitemap_tags = soup.find_all('sitemap')
            if sitemap_tags:
                logger.info(f"Found {len(sitemap_tags)} nested sitemaps in {sitemap_url}. Parsing them recursively (depth 1).")
                for sitemap_tag in sitemap_tags:
                    nested_sitemap_loc = sitemap_tag.find('loc')
                    if nested_sitemap_loc and nested_sitemap_loc.string:
                        nested_sitemap_url = nested_sitemap_loc.string.strip()
                        logger.info(f"Fetching links from nested sitemap: {nested_sitemap_url}")
                        # Recursive call (limited depth for safety, or handle iteratively)
                        # For now, let's just process one level deep to avoid infinite loops if any.
                        # A more robust solution would handle sitemap indexes properly.
                        # This simple recursion might re-fetch if not careful.
                        # For this exercise, let's assume sitemap3.xml is the final level for content.
                        pass # We'll assume sitemap3.xml is the content sitemap for now
                    else:
                        logger.warning(f"Nested sitemap found but <loc> tag is missing or empty in {sitemap_url}")
            
            # Process <url> tags which contain page locations
            url_tags = soup.find_all('url')
            logger.info(f"Found {len(url_tags)} <url> entries in Juniper sitemap: {sitemap_url}")
            
            for url_tag in url_tags:
                loc_tag = url_tag.find('loc')
                if loc_tag and loc_tag.string:
                    page_url = loc_tag.string.strip()
                    
                    # Try to get a title - sitemaps don't usually have rich titles for individual URLs
                    # We might need to fetch the page to get a title, or use part of the URL.
                    # For now, we'll use a generated title. The ingestion pipeline will get actual title.
                    title_from_url = os.path.basename(page_url.rstrip('/')) or f"Juniper Doc from {sitemap_url}"
                    
                    # Determine if it's HTML or PDF based on URL (basic check)
                    doc_type = "pdf" if page_url.lower().endswith(".pdf") else "html"

                    doc_links_metadata.append({
                        "url": page_url,
                        "title": title_from_url, # Placeholder title, actual title fetched during ingestion
                        "doc_type": doc_type, 
                        "vendor": "Juniper",
                        "source_sitemap": sitemap_url
                    })
                    logger.debug(f"  Juniper Sitemap: Found URL: {page_url} (type: {doc_type})")
                else:
                    logger.warning(f"  Juniper Sitemap: <url> tag found without a valid <loc> in {sitemap_url}")

        except Exception as e:
            logger.error(f"Error parsing Juniper sitemap XML {sitemap_url}: {e}", exc_info=True)
            return []
            
        logger.info(f"Extracted {len(doc_links_metadata)} document links from Juniper sitemap: {sitemap_url}")
        return doc_links_metadata



    def _parse_arista_pdf_links_from_context(self, soup_context: BeautifulSoup, base_url: str, vendor: str, context_dict: Optional[Dict[str, Any]]=None) -> List[Dict[str, Any]]:
    # --- NEW: Specific Arista PDF link parsing based on ico24-pdf class ---
        if context_dict is None: context_dict = {}
        found_links = []
        for a_tag in soup_context.select('a:has(span.ico24-pdf)'): 
            href_original = a_tag.get('href')
            if not href_original: continue
            full_url = urljoin(base_url, href_original)
            title_text_parts = []
            for content_element in a_tag.contents:
                if isinstance(content_element, str):
                    stripped_text = content_element.strip()
                    if stripped_text and stripped_text != '.': 
                        title_text_parts.append(stripped_text)
                elif hasattr(content_element, 'name') and content_element.name != 'span' and 'ico24-pdf' not in content_element.get('class', []):
                    stripped_text = content_element.get_text(strip=True)
                    if stripped_text:
                        title_text_parts.append(stripped_text)
            title = " ".join(title_text_parts).strip()
            if not title or len(title) < 3:
                parent_tr = a_tag.find_parent('tr')
                if parent_tr:
                    title_cells = parent_tr.find_all('td', limit=2) 
                    candidate_titles = []
                    for cell in title_cells:
                        cell_text_cleaned = cell.get_text(separator=' ', strip=True)
                        for pdf_icon_link in cell.select('a:has(span.ico24-pdf)'):
                            cell_text_cleaned = cell_text_cleaned.replace(pdf_icon_link.get_text(strip=True), '')
                        cell_text_cleaned = re.sub(r'\s+', ' ', cell_text_cleaned).strip()
                        if cell_text_cleaned and len(cell_text_cleaned) > 3: 
                            candidate_titles.append(cell_text_cleaned)
                    if candidate_titles: title = " - ".join(candidate_titles) 
                    elif not title: title = a_tag.get_text(strip=True)
                else:
                    parent_li = a_tag.find_parent('li')
                    if parent_li and (not title or len(title) < 5):
                        li_text_cleaned = parent_li.get_text(separator=' ', strip=True)
                        a_tag_text_to_remove = a_tag.get_text(strip=True)
                        if a_tag_text_to_remove: li_text_cleaned = li_text_cleaned.replace(a_tag_text_to_remove, '')
                        title = re.sub(r'\s+', ' ', li_text_cleaned).strip()
            if not title or len(title) < 5 : title = os.path.basename(full_url.split('?')[0])
            link_data = { "url": full_url, "title": title if title else "Untitled Arista PDF", "doc_type": "pdf", "vendor": vendor }
            link_data.update(context_dict or {}) 
            found_links.append(link_data)
            logger.debug(f"  Found Arista PDF (via ico24-pdf): '{title}' ({full_url})")
        return found_links





    def _parse_generic_pdf_links_from_soup(self, soup_context: BeautifulSoup, base_url: str, vendor: str, context_dict: Optional[Dict[str, Any]]=None) -> List[Dict[str, Any]]:
        # This is the one used by Aruba's MadCap Flare MCDropDownBody (div > ul > li > a)
        if context_dict is None: context_dict = {}
        found_links = []
        for a_tag in soup_context.select('ul > li > a[href]'): # Specific to MadCap Flare Body structure
            href_original = a_tag.get('href')
            if not href_original: continue
            href_lower = href_original.lower()
            if href_lower.endswith('.pdf') or ('.pdf?' in href_lower):
                full_url = urljoin(base_url, href_original) 
                title = a_tag.get_text(strip=True)
                # ... (rest of title refinement as in response #35's version of this method) ...
                if not title or title.lower() == 'pdf' or len(title) < 5:
                    parent_li = a_tag.find_parent('li')
                    if parent_li:
                        title = parent_li.get_text(separator=' ', strip=True)
                        title = re.sub(r'\s+', ' ', title).strip()[:250]
                if not title or title.lower() == 'pdf': title = os.path.basename(full_url.split('?')[0])

                link_data = { "url": full_url, "title": title if title else f"Untitled {vendor} PDF", "doc_type": "pdf", "vendor": vendor, }
                link_data.update(context_dict or {})
                found_links.append(link_data)
                logger.debug(f"  Found PDF link for {vendor} (generic): '{title}' ({full_url})")
        return found_links

    # --- NEW: Specific parser for Arista Software/General Documentation Page ---
    def parse_arista_software_documentation_page(self, url: str) -> List[Dict[str, Any]]:
        logger.info(f"Parsing Arista Software/General Documentation page: {url}")
        html_content = self.get_page(url, use_selenium_override=True, force_live=True)
        if not html_content: return []
        soup = BeautifulSoup(html_content, 'html.parser')
        doc_links = []
        sections_to_ignore = ["Product Bulletins"] 
        
        # Using broader selectors for sections, adjust as needed from live page inspection
        potential_section_headers = soup.select('h2, h3, h4, .panel-title, .section-title')

        for header in potential_section_headers:
            section_title = header.get_text(strip=True)
            if not section_title: continue
            logger.debug(f"Arista Software Page: Checking section '{section_title}'")
            if any(ignore_term.lower() in section_title.lower() for ignore_term in sections_to_ignore):
                logger.info(f"Arista Software Page: Ignoring section '{section_title}'.")
                continue

            # Find the content container associated with this header
            content_container = None
            # Common pattern: header is child of a div, content is sibling div or child of same parent
            parent_container = header.find_parent(['div.panel', 'div.section-block', 'article']) # Example containers
            if parent_container:
                # Look for specific content body class, or just take the whole parent if specific body not found
                content_container = parent_container.find(['div.panel-body', 'div.section-content', 'table', 'ul', 'dl'])
                if not content_container: content_container = parent_container # Fallback to parent
            
            if not content_container: # If header is standalone, look for next sibling table/ul/dl
                next_element = header.next_sibling
                while next_element:
                    if hasattr(next_element, 'name') and next_element.name in ['table', 'ul', 'dl']:
                        content_container = next_element
                        break
                    if hasattr(next_element, 'name') and next_element.name in ['h2','h3','h4']: break # Reached another header
                    next_element = next_element.next_sibling
            
            if content_container:
                logger.debug(f"Arista Software Page: Processing links for section '{section_title}'.")
                # --- CALL THE NEW ARISTA PDF LINK PARSER ---
                section_links = self._parse_arista_pdf_links_from_context(content_container, url, "Arista", {"section_source": section_title})
                doc_links.extend(section_links)
            else:
                logger.debug(f"Arista Software Page: No content container found for section '{section_title}'.")
                
        logger.info(f"Extracted {len(doc_links)} PDF links from Arista Software/General page '{url}'.")
        return doc_links

    # --- NEW: Specific parser for Arista Hardware Documentation Page ---
    def parse_arista_hardware_documentation_page(self, url: str) -> List[Dict[str, Any]]:
        logger.info(f"Parsing Arista Hardware Documentation page: {url}")
        html_content = self.get_page(url, use_selenium_override=True, force_live=True)
        if not html_content: return []
        soup = BeautifulSoup(html_content, 'html.parser')
        doc_links = []
        product_families_to_ignore = ["CloudVision Appliance", "Awake Security", "Edge Threat Management"]
        
        # Try to find the main hardware table; this selector might need adjustment
        main_hardware_table = soup.select_one('table.docTable, table#hardware-docs-table') # Example table selectors
        if not main_hardware_table:
            logger.warning(f"Arista Hardware Page: Specific hardware table not found. Trying to find largest table.")
            tables = soup.find_all('table')
            if tables: main_hardware_table = max(tables, key=lambda t: len(str(t)), default=None)

        if main_hardware_table:
            logger.debug(f"Arista Hardware Page: Processing main hardware table.")
            rows = main_hardware_table.find_all('tr')
            current_product_family = "Unknown Hardware Family" # Keep track of current family if it spans rows

            for row in rows[1:]: # Skip header row usually
                cells = row.find_all('td')
                if not cells: continue

                # Determine Product Family - often in the first cell, may span rows
                # If first cell has rowspan or is less indented, it might be a family header
                product_family_cell = cells[0]
                pf_text_candidate = product_family_cell.get_text(strip=True)
                if pf_text_candidate and (product_family_cell.has_attr('rowspan') or len(cells) < 3): # Heuristic for family row
                    current_product_family = pf_text_candidate
                
                if any(ignore_pf.lower() in current_product_family.lower() for ignore_pf in product_families_to_ignore):
                    if pf_text_candidate and current_product_family.lower() == pf_text_candidate.lower(): # Only log once per ignored family
                        logger.info(f"Arista Hardware Page: Ignoring Product Family '{current_product_family}'.")
                    continue 
                
                # "Additional Media" column is usually the last one with PDF icons, or one of the last
                # We'll parse the whole row for PDF icons to be safe.
                # --- CALL THE NEW ARISTA PDF LINK PARSER ---
                # Pass the current_product_family for context.
                row_links = self._parse_arista_pdf_links_from_context(row, url, "Arista", {"product_family": current_product_family})
                doc_links.extend(row_links)
        else:
            logger.error(f"Arista Hardware Page: Main hardware table not found on {url}. Cannot extract links.")
            
        logger.info(f"Extracted {len(doc_links)} PDF links from Arista Hardware page '{url}'.")
        return doc_links

    # --- ARUBA PARSERS (from response #35, no changes needed here if they work) ---
    def scrape_aruba_expandable_section_pdfs(self, series_page_url: str) -> List[Dict[str, Any]]:
        # (Code from response #35 - for MadCap Flare MCDropDownHotSpot)
        logger.info(f"Selenium: Scraping [+] expandable section PDFs from: {series_page_url}")
        all_expanded_pdf_links_map = {} 
        driver = self._get_selenium_driver()
        if not driver: return []
        try:
            driver.get(series_page_url)
            WebDriverWait(driver, self.selenium_timeout).until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.MCDropDown"))) 
            expander_hotspots = driver.find_elements(By.CSS_SELECTOR, "a.MCDropDownHotSpot")
            logger.info(f"Selenium: Found {len(expander_hotspots)} MCDropDownHotSpot elements on {series_page_url}.")
            if not expander_hotspots: return []
            for i in range(len(expander_hotspots)):
                current_hotspots = driver.find_elements(By.CSS_SELECTOR, "a.MCDropDownHotSpot")
                if i >= len(current_hotspots): logger.warning("Selenium: Aruba MCDropDownHotSpot list changed during iteration."); break
                hotspot_to_click = current_hotspots[i]
                section_text_for_log = (hotspot_to_click.text if hotspot_to_click.text else f"hotspot_{i+1}").strip()
                try:
                    parent_mc_dropdown_div = hotspot_to_click.find_element(By.XPATH, "./ancestor::div[contains(@class, 'MCDropDown')]")
                    is_already_open = "MCDropDown_Open" in parent_mc_dropdown_div.get_attribute("class") or parent_mc_dropdown_div.get_attribute("data-mc-state") == "open"
                    if is_already_open: logger.info(f"Selenium: Section '{section_text_for_log}' is already open.")
                    else:
                        logger.info(f"Selenium: Clicking to expand section: '{section_text_for_log}'")
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", hotspot_to_click)
                        time.sleep(0.5) 
                        WebDriverWait(driver, self.selenium_timeout).until(EC.element_to_be_clickable(hotspot_to_click))
                        hotspot_to_click.click()
                        WebDriverWait(driver, self.selenium_timeout).until(lambda d: "MCDropDown_Open" in hotspot_to_click.find_element(By.XPATH, "./ancestor::div[contains(@class, 'MCDropDown')]").get_attribute("class") or hotspot_to_click.find_element(By.XPATH, "./ancestor::div[contains(@class, 'MCDropDown')]").get_attribute("data-mc-state") == "open")
                        logger.info(f"Selenium: Section '{section_text_for_log}' confirmed expanded.")
                        time.sleep(random.uniform(0.5, 1.0))
                    dropdown_body_id = hotspot_to_click.get_attribute("aria-controls")
                    if dropdown_body_id:
                        try:
                            dropdown_body_element = WebDriverWait(driver, self.selenium_timeout).until(EC.visibility_of_element_located((By.ID, dropdown_body_id)))
                            body_html = dropdown_body_element.get_attribute("outerHTML")
                            soup_for_section = BeautifulSoup(body_html, 'html.parser')
                            context_info = {"vendor": "Aruba", "version_context": f"section_{section_text_for_log.replace(' ', '_')}"}
                            pdf_links_in_section = self._parse_generic_pdf_links_from_soup(soup_for_section, series_page_url, "Aruba", context_info)
                            for link_info in pdf_links_in_section: all_expanded_pdf_links_map[link_info["url"]] = link_info
                        except Exception as e_body_parse: logger.warning(f"Selenium: Could not find/parse Aruba dropdown body ID '{dropdown_body_id}' for '{section_text_for_log}': {e_body_parse}")
                    else: logger.warning(f"Selenium: No 'aria-controls' for Aruba hotspot '{section_text_for_log}'.")
                except Exception as e_click: logger.warning(f"Selenium: Error processing Aruba hotspot '{section_text_for_log}': {e_click}", exc_info=True)
        except Exception as e: logger.error(f"Selenium: Main error scraping Aruba MCDropDown sections on {series_page_url}: {e}", exc_info=True)
        final_links_list = list(all_expanded_pdf_links_map.values())
        logger.info(f"Selenium: Found {len(final_links_list)} unique PDF links from MCDropDown sections on {series_page_url}")
        return final_links_list

    def parse_aruba_documentation_series_page(self, series_page_url: str) -> List[Dict[str, Any]]:
        # (Code from response #35 - calls scrape_aruba_expandable_section_pdfs)
        logger.info(f"Parsing Aruba Series Index Page (static + MCDropDown expandable sections): {series_page_url}")
        html_content = self.get_page(series_page_url, use_selenium_override=True, force_live=True)
        if not html_content: return []
        soup = BeautifulSoup(html_content, 'html.parser')
        context_info_static = {"vendor": "Aruba", "version_context": "default_static_view"}
        static_pdf_links = self._parse_generic_pdf_links_from_soup(soup, series_page_url, "Aruba", context_info_static)
        logger.info(f"Found {len(static_pdf_links)} initial/static PDF links on {series_page_url}")
        expanded_section_pdf_links = self.scrape_aruba_expandable_section_pdfs(series_page_url)
        combined_links_map = {} 
        for link_info in static_pdf_links + expanded_section_pdf_links:
            if link_info.get("url"): combined_links_map[link_info["url"]] = link_info
        final_links = list(combined_links_map.values())
        logger.info(f"Total {len(final_links)} unique PDF links found for Aruba Series Page {series_page_url}.")
        self._quit_selenium_driver() 
        return final_links

    # --- CISCO & HACKER NEWS PARSERS (from response #35, ensure selectors are verified) ---
    def parse_cisco_release_notes(self, url: str) -> List[Dict[str, Any]]:
        # (Code from response #35 - ensure selectors are up-to-date)
        html = self.get_page(url, use_selenium_override=False, force_live=True)
        if not html: return []
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        for row in soup.select("table.listingTable tr, div.cisco-table-row"): 
            link_tag = row.select_one("td a[href*='release-note'], td a[href*='rn-']") 
            if link_tag:
                title = link_tag.get_text(strip=True)
                href = link_tag.get('href')
                if href and title:
                    absolute_url = urljoin(url, href)
                    results.append({"title": title, "url": absolute_url, "vendor": "Cisco", "doc_type": "Release Notes"})
        logger.info(f"Parsed {len(results)} Cisco Release Notes from {url}")
        return results

    def parse_cisco_config_guides(self, url: str) -> List[Dict[str, Any]]:
        # (Code from response #35 - ensure selectors are up-to-date)
        html = self.get_page(url, use_selenium_override=False, force_live=True)
        if not html: return []
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        for row in soup.select("table.listingTable tr, div.cisco-table-row"):
            link_tag = row.select_one("td a[href*='configuration/guide'], td a[href*='cg-']")
            if link_tag:
                title = link_tag.get_text(strip=True)
                href = link_tag.get('href')
                if href and title:
                    absolute_url = urljoin(url, href)
                    results.append({"title": title, "url": absolute_url, "vendor": "Cisco", "doc_type": "Configuration Guide"})
        logger.info(f"Parsed {len(results)} Cisco Config Guides from {url}")
        return results

    def parse_hacker_news(self, url: str) -> List[Dict[str, Any]]:
        # (Code from response #35 - seems generally stable)
        html = self.get_page(url, use_selenium_override=False, force_live=True)
        if not html: return []
        soup = BeautifulSoup(html, 'html.parser')
        results = []
        from datetime import datetime 
        for row in soup.select("tr.athing"):
            title_link = row.select_one('td.title > span.titleline > a')
            if title_link:
                title = title_link.text.strip()
                href = title_link.get('href')
                if href and not href.startswith(('http://', 'https://')): href = urljoin(url, href)
                if href and title:
                    results.append({"title": title, "url": href, "date": datetime.now().isoformat(), "vendor": "HackerNews", "doc_type": "posts"})
        logger.info(f"Parsed {len(results)} stories from Hacker News: {url}")
        return results

