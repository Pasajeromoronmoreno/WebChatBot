import asyncio
from playwright.async_api import async_playwright
from urllib.parse import urljoin, urlparse
from typing import Set, List, Dict, Optional
import logging
import json
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

@dataclass
class PageContent:
    """Estructura para almacenar el contenido de una página"""
    url: str
    content: str

class APICrawler:
    def __init__(self, start_url: str):
        # Validar URL
        parsed_url = urlparse(start_url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            raise ValueError(f"URL inválida: {start_url}")
        
        self.start_url = start_url
        self.base_domain = urlparse(start_url).netloc
        self.visited_urls: Set[str] = set()
        self.navigation_structure: Dict = {}
        
        # Crear nombre para la documentación basado en la URL
        parsed_url = urlparse(start_url)
        domain_parts = parsed_url.netloc.split('.')
        path_parts = [p for p in parsed_url.path.split('/') if p]
        
        # Usar dominio principal y primer nivel de path
        doc_name = domain_parts[-2] if len(domain_parts) > 1 else domain_parts[0]
        if path_parts:
            doc_name += f"_{path_parts[0]}"
        
        # Sanitizar el nombre del directorio
        doc_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in doc_name)
        
        # Crear estructura de carpetas
        self.docs_base = Path("docs_data")
        self.current_doc_dir = self.docs_base / doc_name
        self.api_docs_dir = self.current_doc_dir / "api_docs"
        self.endpoints_dir = self.api_docs_dir / "endpoints"
        self.metadata_dir = self.current_doc_dir / "metadata"
        
        # Crear todas las carpetas necesarias
        for directory in [self.docs_base, self.current_doc_dir, self.api_docs_dir, 
                         self.endpoints_dir, self.metadata_dir]:
            directory.mkdir(exist_ok=True, parents=True)
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.current_doc_dir / 'crawler.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Guardar configuración inicial
        self._save_config()

    def _save_config(self):
        """Guarda la configuración de esta documentación"""
        config = {
            "url": self.start_url,
            "domain": self.base_domain,
            "last_update": str(datetime.now()),
            "version": "1.0.0",
            "parser_version": "1.0.0"
        }
        with open(self.metadata_dir / "config.json", "w", encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    @staticmethod
    def get_last_docs() -> Optional[str]:
        """Obtiene la última documentación parseada"""
        config_file = Path("docs_data/last_docs.json")
        if config_file.exists():
            try:
                with open(config_file, encoding='utf-8') as f:
                    return json.load(f)["last_url"]
            except:
                return None
        return None

    @staticmethod
    def save_last_docs(url: str):
        """Guarda la última documentación parseada"""
        config_file = Path("docs_data/last_docs.json")
        config_file.parent.mkdir(exist_ok=True)
        with open(config_file, "w", encoding='utf-8') as f:
            json.dump({
                "last_url": url,
                "timestamp": str(datetime.now())
            }, f, indent=2, ensure_ascii=False)

    def _save_navigation_structure(self):
        """Guarda la estructura de navegación"""
        nav_file = self.api_docs_dir / 'navigation.json'
        with open(nav_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": str(datetime.now()),
                "structure": self.navigation_structure
            }, f, indent=2, ensure_ascii=False)
        self.logger.info(f"\nEstructura de navegación guardada en {nav_file}")

    def _save_page_content(self, content: PageContent):
        """Guarda el texto de la página"""
        filename = urlparse(content.url).path.replace('/', '_') or 'index'
        if not filename.endswith('.json'):
            filename += '.json'
        
        filepath = self.endpoints_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "url": content.url,
                "text": content.content,
                "timestamp": str(datetime.now()),
                "parser_version": "1.0.0"
            }, f, indent=2, ensure_ascii=False)

    async def crawl(self):
        """Punto de entrada principal para el crawler"""
        async with async_playwright() as p:
            try:
                self.logger.info("Iniciando navegador...")
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36'
                )
                
                # Extraer estructura de navegación
                await self._extract_navigation_structure(context)
                self._save_navigation_structure()
                
                # Procesar páginas
                await self._process_all_pages(context)
                
                # Mostrar resumen final
                status = self.check_docs_status()
                failed_urls = self._get_failed_urls()
                
                self.logger.info(f"\n{'='*80}")
                self.logger.info("RESUMEN FINAL")
                self.logger.info(f"- Archivos totales: {status['total_files']}")
                self.logger.info(f"- URLs esperadas: {status['total_expected']}")
                self.logger.info(f"- URLs fallidas: {status['failed_urls']}")
                self.logger.info(f"- Última actualización: {status['last_update']}")
                self.logger.info(f"- Tamaño total: {status['size_bytes'] / 1024:.2f} KB")
                self.logger.info(f"- Documentación completa: {'Sí' if status['is_complete'] else 'No'}")
                if status['is_complete']:
                    self.logger.info("  (Se han descargado todos los archivos posibles)")
                if failed_urls:
                    self.logger.info(f"  Las URLs fallidas se guardaron en metadata/failed_urls.json")
                self.logger.info(f"{'='*80}\n")
                
                # Si la documentación está completa, limpiar archivos temporales
                if status['is_complete']:
                    self._cleanup_temp_files()
                
            finally:
                await context.close()
                await browser.close()

    async def _extract_navigation_structure(self, context):
        """Extrae la estructura de navegación del sitio de forma recursiva"""
        # Intentar cargar estructura previa
        nav_file = self.api_docs_dir / 'navigation.json'
        if nav_file.exists():
            try:
                with open(nav_file, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                    if saved_data.get("structure"):
                        self.logger.info("Cargando estructura de navegación guardada...")
                        self.navigation_structure = saved_data["structure"]
                        return
            except Exception as e:
                self.logger.warning(f"Error cargando estructura guardada: {e}")

        async def explore_page(url: str, depth=0, max_depth=3, visited=None):  # Reducido a 3 niveles
            if visited is None:
                visited = set()
                
            if depth > max_depth or url in visited:
                return []
                
            visited.add(url)
            page = await context.new_page()
            try:
                self.logger.info(f"Explorando nivel {depth}: {url}")
                
                # Aumentar timeouts y manejar errores mejor
                try:
                    await page.goto(url, timeout=60000)  # 60 segundos
                    await page.wait_for_load_state('networkidle', timeout=60000)
                    await page.wait_for_timeout(1000)  # reducido a 1 segundo para no demorar tanto
                except Exception as e:
                    self.logger.error(f"Error cargando {url}: {str(e)}")
                    return []
                
                # Obtener enlaces con timeout
                try:
                    links = await asyncio.wait_for(
                        page.evaluate('''() => {
                            function getAllLinks(element, visited = new Set()) {
                                let links = [];
                                try {
                                    const elements = element.querySelectorAll('a[href]');
                                    for (const el of elements) {
                                        if (!visited.has(el.href)) {
                                            visited.add(el.href);
                                            links.push({
                                                text: el.textContent.trim(),
                                                href: el.href,
                                                isNav: !!el.closest('nav, .sidebar, .navigation, .menu, .toc')
                                            });
                                        }
                                    }
                                } catch (e) {
                                    console.error('Error in getAllLinks:', e);
                                }
                                return links;
                            }
                            return getAllLinks(document.body);
                        }'''),
                        timeout=30.0  # 30 segundos máximo para extraer enlaces
                    )
                except asyncio.TimeoutError:
                    self.logger.error(f"Timeout extrayendo enlaces de {url}")
                    return []
                except Exception as e:
                    self.logger.error(f"Error extrayendo enlaces en {url}: {str(e)}")
                    return []
                
                valid_links = []
                for link in links:
                    if self._should_crawl_url(link['href']):
                        valid_links.append(link)
                        # Explorar recursivamente con delay entre llamadas
                        if depth < max_depth:
                            await asyncio.sleep(0.1)  # pequeño delay entre llamadas recursivas
                            sub_links = await explore_page(link['href'], depth + 1, max_depth, visited)
                            valid_links.extend(sub_links)
                
                # Guardar progreso parcial cada 10 URLs
                if len(visited) % 10 == 0:
                    self._save_navigation_structure()
                    
                return valid_links
                
            finally:
                await page.close()
        
        try:
            self.logger.info(f"Iniciando exploración recursiva desde {self.start_url}")
            all_links = await explore_page(self.start_url)
            
            # Procesar y organizar los enlaces encontrados
            structure = {}
            for link in all_links:
                url = link['href']
                text = link['text']
                
                if url and text:
                    # Usar un texto único si hay duplicados
                    base_text = text
                    counter = 1
                    while text in structure:
                        text = f"{base_text} ({counter})"
                        counter += 1
                    
                    structure[text] = {
                        "url": url,
                        "subsections": {},
                        "is_nav": link['isNav']
                    }
            
            self.navigation_structure = structure
            total_urls = len(self._extract_all_urls(self.navigation_structure))
            self.logger.info(f"Total de URLs únicas encontradas: {total_urls}")
            
        except Exception as e:
            self.logger.error(f"Error en exploración recursiva: {e}")
            raise

    def _should_crawl_url(self, url: str) -> bool:
        """Determina si una URL debe ser procesada"""
        if not url:
            return False
        
        parsed_url = urlparse(url)
        base_url = urlparse(self.start_url)
        
        # Verificar que sea del mismo dominio
        if parsed_url.netloc != base_url.netloc:
            return False
        
        # Para LangChain, asegurarse de que estamos en la sección api_reference
        if 'api_reference' not in parsed_url.path:
            return False
        
        # Ignorar ciertos patrones
        ignored_patterns = [
            '.pdf', '.zip', '.jpg', '.png', '.gif',  # archivos
            '/download/', '/assets/', '/images/',     # recursos
            'javascript:', 'mailto:', 'tel:',         # protocolos
            '#'                                      # anchors
        ]
        
        return not any(pattern in url.lower() for pattern in ignored_patterns)

    def _extract_all_urls(self, structure: Dict) -> Set[str]:
        """Extrae todas las URLs de la estructura"""
        urls = set()
        for item in structure.values():
            if isinstance(item, dict):
                if "url" in item and item["url"]:
                    urls.add(item["url"])
                if "subsections" in item:
                    urls.update(self._extract_all_urls(item["subsections"]))
        return urls

    async def _process_all_pages(self, context):
        pages_to_process = self._extract_all_urls(self.navigation_structure)
        existing_files = self._get_existing_files()
        pages_to_download = {url for url in pages_to_process if url not in existing_files}
        
        if not pages_to_download:
            self.logger.info("\nNo hay nuevas páginas para descargar")
            return
        
        # Agregar esta línea al inicio de _process_all_pages
        progress = {
            "completed_urls": set(),
            "failed_urls": set(),
            "last_batch": 0
        }

        # Cargar estado de procesamiento previo
        saved_progress = self._load_detailed_progress()
        if saved_progress:
            self.logger.info(f"Retomando desde progreso guardado...")
            progress.update(saved_progress)
            pages_to_download = {url for url in pages_to_download 
                                if url not in progress["completed_urls"]}
        
        total = len(pages_to_download)
        workers = 15  # Aumentado a 15 workers
        
        # Crear semáforo y estado de workers
        semaphore = asyncio.Semaphore(workers)
        worker_status = {i: {"active": False, "current_url": None} for i in range(1, workers + 1)}
        
        # Nuevo: Batch processing
        batch_size = 50
        all_urls = sorted(pages_to_download)
        
        total_successful = 0  # Contador para éxitos totales
        
        async def process_with_semaphore(url, idx):
            async with semaphore:
                worker_id = next(i for i, status in worker_status.items() if not status["active"])
                worker_status[worker_id]["active"] = True
                worker_status[worker_id]["current_url"] = url
                
                try:
                    self.logger.info(f"Worker {worker_id}: Iniciando página {idx}/{total}")
                    self.logger.info(f"Worker {worker_id}: Procesando {url}")
                    content = await self._process_page(context, url)
                    if content:
                        self._save_page_content(content)
                        if progress is not None:
                            progress["completed_urls"].add(url)
                        self._save_progress(url)
                        self.logger.info(f"Worker {worker_id}: [OK] {url} descargada correctamente")
                        return True
                    else:
                        if progress is not None:
                            progress["failed_urls"].add(url)
                        return False
                except Exception as e:
                    self.logger.error(f"Worker {worker_id}: [X] Error en {url}: {str(e)}")
                    if progress is not None:
                        progress["failed_urls"].add(url)
                    return False
                finally:
                    worker_status[worker_id]["active"] = False
                    worker_status[worker_id]["current_url"] = None
        
        for i in range(0, len(all_urls), batch_size):
            batch_urls = all_urls[i:i + batch_size]
            self.logger.info(f"\nProcesando batch {i//batch_size + 1}/{len(all_urls)//batch_size + 1}")
            tasks = [process_with_semaphore(url, idx) for idx, url in enumerate(batch_urls, start=i+1)]
            batch_results = await asyncio.gather(*tasks)
            total_successful += sum(1 for r in batch_results if r)
            
            # Guardar progreso después de cada batch
            self._save_detailed_progress(progress)
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info("¡PARSEO FINALIZADO!")
        self.logger.info(f"- Páginas descargadas exitosamente: {total_successful}/{total}")
        self.logger.info(f"- Total en el sistema: {len(existing_files) + total_successful}")
        self.logger.info(f"- Ubicación: {self.endpoints_dir}")
        self.logger.info(f"{'='*80}\n")

    async def _process_page(self, context, url: str) -> Optional[PageContent]:
        # Verificar si la URL ya falló antes
        if self._is_known_failed_url(url):
            self.logger.info(f"Saltando URL previamente fallida: {url}")
            return None
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            page = await context.new_page()
            try:
                self.logger.info(f"Intento {retry_count + 1}/{max_retries} para {url}")
                
                # Verificar si la página existe
                response = await page.goto(url, wait_until='networkidle', timeout=60000)
                
                if response is None:
                    self.logger.error(f"No se pudo acceder a {url}")
                    retry_count += 1
                    await asyncio.sleep(2)
                    continue
                
                if response.status == 404:
                    self.logger.error(f"Página no encontrada (404): {url}")
                    return None  # No reintentar si la página no existe
                
                if not response.ok:
                    self.logger.error(f"Error HTTP {response.status} en {url}")
                    retry_count += 1
                    await asyncio.sleep(2)
                    continue
                
                # Esperar a que el contenido se cargue
                try:
                    await page.wait_for_load_state('networkidle', timeout=10000)
                    await page.wait_for_timeout(3000)
                    
                    # Obtener el HTML para diagnóstico
                    html = await page.content()
                    if len(html) < 100:  # HTML demasiado corto
                        self.logger.warning(f"HTML muy corto ({len(html)} caracteres) en {url}")
                        retry_count += 1
                        await asyncio.sleep(2)
                        continue
                    
                    # Verificar si hay contenido antes de extraer
                    content_exists = await page.evaluate('''() => {
                        const selectors = [
                            "main article",
                            "[data-testid='docs-content']",
                            "main",
                            "#main-content",
                            "article",
                            ".docusaurus-content",
                            ".markdown-body",  // Agregado para GitHub-style docs
                            ".documentation"   // Selector genérico
                        ];
                        const found = selectors.find(s => document.querySelector(s));
                        return {
                            exists: !!found,
                            selector: found || null,
                            bodyText: document.body.textContent.length
                        };
                    }''')
                    
                    if not content_exists['exists']:
                        self.logger.warning(
                            f"No se encontró el contenido principal en {url}. "
                            f"Texto total: {content_exists['bodyText']} caracteres"
                        )
                        retry_count += 1
                        await asyncio.sleep(2)
                        continue
                    
                    content = await self._extract_main_content(page)
                    
                    # Verificar si el contenido es válido
                    if content and len(content.strip()) > 50:
                        self.logger.info(f"Contenido extraído exitosamente de {url} ({len(content)} caracteres)")
                        return PageContent(url=url, content=content)
                    
                    self.logger.warning(
                        f"Contenido extraído muy corto ({len(content.strip()) if content else 0} caracteres) "
                        f"en {url} usando selector: {content_exists['selector']}"
                    )
                    retry_count += 1
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    self.logger.error(f"Error esperando contenido en {url}: {str(e)}")
                    retry_count += 1
                    await asyncio.sleep(2)
                    
            except Exception as e:
                self.logger.error(f"Error procesando {url}: {str(e)}")
                retry_count += 1
                await asyncio.sleep(2)
            finally:
                await page.close()
        
        # Si llegamos aquí, fallaron todos los intentos
        self.logger.error(f"Fallaron todos los intentos ({max_retries}) para {url}")
        # Guardar URL fallida para análisis posterior
        self._save_failed_url(url)
        return None

    def _save_failed_url(self, url: str):
        """Guarda las URLs que fallaron para análisis posterior"""
        failed_urls_file = self.metadata_dir / "failed_urls.json"
        failed_urls = []
        
        # Cargar URLs fallidas existentes
        if failed_urls_file.exists():
            try:
                with open(failed_urls_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    failed_urls = data.get("urls", [])
            except Exception:
                pass
        
        # Agregar nueva URL fallida con timestamp
        failed_urls.append({
            "url": url,
            "timestamp": str(datetime.now()),
            "reason": "Max retries exceeded"
        })
        
        # Guardar lista actualizada
        with open(failed_urls_file, 'w', encoding='utf-8') as f:
            json.dump({
                "urls": failed_urls,
                "last_update": str(datetime.now())
            }, f, indent=2, ensure_ascii=False)

    async def _extract_main_content(self, page) -> str:
        """Extrae el contenido principal de la página"""
        try:
            await page.wait_for_load_state('networkidle')
            await page.wait_for_timeout(3000)
            
            content = await page.evaluate('''() => {
                function extractContent() {
                    const selectors = [
                        "main article",
                        "[data-testid='docs-content']",
                        "main",
                        "#main-content",
                        "article",
                        ".docusaurus-content"
                    ];
                    
                    let mainElement = null;
                    for (const selector of selectors) {
                        const element = document.querySelector(selector);
                        if (element) {
                            mainElement = element;
                            break;
                        }
                    }
                    
                    if (!mainElement) return '';
                    
                    const clone = mainElement.cloneNode(true);
                    
                    // Remover elementos no deseados
                    const removeSelectors = [
                        'nav', 'header', 'footer', 'script', 'style', 'noscript',
                        '[role="navigation"]', '.table-of-contents', '.sidebar',
                        '.menu', '.navigation', '.breadcrumbs', '.edit-this-page',
                        '.theme-edit-this-page', '.pagination-nav', '.tocCollapsible_',
                        '.theme-doc-version-badge', '.theme-last-updated', '.hash-link'
                    ];
                    
                    removeSelectors.forEach(selector => {
                        clone.querySelectorAll(selector).forEach(el => el.remove());
                    });
                    
                    function formatType(text) {
                        // Formatear tipos de datos
                        return text.replace(/(\w+)\s*\[\s*Required\s*\]/g, '`$1` (Required)')
                                  .replace(/(\w+)\s*\[\s*Optional\s*\]/g, '`$1` (Optional)');
                    }
                    
                    function extractText(element, level = 0) {
                        let text = [];
                        
                        for (const node of element.childNodes) {
                            if (node.nodeType === 3) { // Nodo de texto
                                const trimmed = node.textContent.trim();
                                if (trimmed) text.push(trimmed);
                            } else if (node.nodeType === 1) { // Elemento
                                const tag = node.tagName.toLowerCase();
                                
                                if (tag.match(/^h[1-6]$/)) {
                                    // Títulos
                                    const level = parseInt(tag[1]);
                                    const title = node.textContent.trim()
                                        .replace(/\s*#\s*$/, '')  // Remover # al final
                                        .replace(/\[source\]/i, '') // Remover [source]
                                        .trim();
                                    text.push(`\\n${'#'.repeat(level)} ${title}\\n`);
                                } else if (tag === 'p') {
                                    // Párrafos
                                    const content = extractText(node, level);
                                    if (content) text.push(content + '\\n');
                                } else if (tag === 'code') {
                                    // Código inline
                                    text.push(`\`${node.textContent.trim()}\``);
                                } else if (tag === 'pre') {
                                    // Bloques de código
                                    text.push(`\\n\`\`\`\\n${node.textContent.trim()}\\n\`\`\`\\n`);
                                } else if (['ul', 'ol'].includes(tag)) {
                                    // Listas
                                    text.push('\\n');
                                    const items = extractText(node, level + 1);
                                    if (items) text.push(items + '\\n');
                                } else if (tag === 'li') {
                                    // Elementos de lista con indentación
                                    const content = extractText(node, level);
                                    if (content) {
                                        // Si es un parámetro, darle formato especial
                                        if (content.includes('[Required]') || content.includes('[Optional]')) {
                                            text.push(`${'  '.repeat(level)}- ${formatType(content)}`);
                                        } else {
                                            text.push(`${'  '.repeat(level)}- ${content}`);
                                        }
                                    }
                                } else if (tag === 'dt') {
                                    // Términos de definición
                                    const content = extractText(node, level);
                                    if (content) text.push(`\\n**${content}**:`);
                                } else if (tag === 'dd') {
                                    // Descripción de definición
                                    const content = extractText(node, level);
                                    if (content) text.push(` ${content}\\n`);
                                } else {
                                    const content = extractText(node, level);
                                    if (content) text.push(content);
                                }
                            }
                        }
                        
                        return text.join(' ').replace(/\\s+/g, ' ').trim();
                    }
                    
                    let text = extractText(clone);
                    
                    // Limpiar el texto final
                    return text
                        .split('\\n')
                        .map(line => line.trim())
                        .filter(line => line)
                        .join('\\n')
                        .replace(/\\n{3,}/g, '\\n\\n')  // Máximo 2 saltos de línea
                        .replace(/#+\s*$/g, '')         // Eliminar # al final
                        .replace(/\[source\]/gi, '')    // Eliminar [source]
                        .replace(/\s*#\s*(?=\\n|$)/g, '') // Eliminar # al final de líneas
                        .trim();
                }
                
                return extractContent();
            }''')
            
            if not content or len(content.strip()) < 50:
                self.logger.warning(f"Contenido extraído muy corto o vacío")
                return ""
            
            return content.strip()
            
        except Exception as e:
            self.logger.error(f"Error extrayendo contenido: {str(e)}")
            return ""

    def check_docs_status(self) -> Dict:
        """Verifica el estado de la documentación descargada"""
        status = {
            "total_files": 0,
            "last_update": None,
            "size_bytes": 0,
            "is_complete": False,
            "failed_urls": 0,
            "total_expected": 0
        }
        
        try:
            # Contar archivos y obtener última actualización
            for file in self.endpoints_dir.glob('*.json'):
                if file.name != 'navigation.json':
                    status["total_files"] += 1
                    status["size_bytes"] += file.stat().st_size
                    
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        timestamp = datetime.fromisoformat(data["timestamp"].split(".")[0])
                        if not status["last_update"] or timestamp > status["last_update"]:
                            status["last_update"] = timestamp
            
            # Obtener URLs fallidas
            failed_urls = self._get_failed_urls()
            status["failed_urls"] = len(failed_urls)
            
            # Verificar si la documentación está completa
            nav_file = self.api_docs_dir / 'navigation.json'
            if nav_file.exists():
                with open(nav_file, 'r', encoding='utf-8') as f:
                    nav_data = json.load(f)
                    total_expected = len(self._extract_all_urls(nav_data["structure"]))
                    status["total_expected"] = total_expected
                    # La documentación está completa si tenemos todos los archivos posibles
                    # (archivos descargados + URLs fallidas = total esperado)
                    status["is_complete"] = (status["total_files"] + status["failed_urls"]) >= total_expected
            
            return status
        except Exception as e:
            self.logger.error(f"Error verificando estado: {str(e)}")
            return status

    def _get_existing_files(self):
        """Obtiene las URLs de los archivos ya descargados"""
        existing_files = set()
        for file in self.endpoints_dir.glob('*.json'):
            if file.name != 'navigation.json':
                try:
                    # Leer el archivo y obtener la URL original
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'url' in data:
                            existing_files.add(data['url'])
                except Exception as e:
                    self.logger.error(f"Error leyendo archivo {file}: {e}")
        return existing_files

    def _save_progress(self, last_processed_url: str):
        """Guarda el progreso del parseo"""
        progress_file = self.metadata_dir / "progress.json"
        progress = {
            "last_processed_url": last_processed_url,
            "timestamp": str(datetime.now()),
            "total_processed": len(self._get_existing_files())
        }
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)

    def _load_progress(self) -> Optional[str]:
        """Carga el último URL procesado"""
        progress_file = self.metadata_dir / "progress.json"
        if progress_file.exists():
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    return json.load(f)["last_processed_url"]
            except:
                return None
        return None

    def _verify_downloaded_files(self):
        """Verifica la integridad de los archivos descargados"""
        existing_files = self._get_existing_files()
        corrupted_files = []
        
        for file in self.endpoints_dir.glob('*.json'):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if not data.get("text"):
                        corrupted_files.append(file.name)
            except:
                corrupted_files.append(file.name)
        
        if corrupted_files:
            self.logger.warning(f"Se encontraron {len(corrupted_files)} archivos corruptos")
            return corrupted_files
        return []

    def _save_detailed_progress(self, progress):
        """Guarda un progreso detallado del procesamiento"""
        progress_file = self.metadata_dir / "detailed_progress.json"
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": str(datetime.now()),
                "completed_urls": list(progress.get("completed_urls", set())),
                "failed_urls": list(progress.get("failed_urls", set())),
                "last_batch": progress.get("last_batch", 0)
            }, f, indent=2, ensure_ascii=False)

    def _load_detailed_progress(self) -> Dict:
        """Carga el progreso detallado"""
        progress_file = self.metadata_dir / "detailed_progress.json"
        if progress_file.exists():
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return {
                        "completed_urls": set(data.get("completed_urls", [])),
                        "failed_urls": set(data.get("failed_urls", [])),
                        "last_batch": data.get("last_batch", 0)
                    }
            except Exception as e:
                self.logger.error(f"Error cargando progreso detallado: {e}")
        return {
            "completed_urls": set(),
            "failed_urls": set(),
            "last_batch": 0
        }

    def _get_failed_urls(self) -> List[Dict]:
        """Obtiene la lista de URLs que fallaron"""
        failed_urls_file = self.metadata_dir / "failed_urls.json"
        if failed_urls_file.exists():
            try:
                with open(failed_urls_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("urls", [])
            except Exception:
                pass
        return []

    def _cleanup_temp_files(self):
        """Limpia archivos temporales y de progreso después de una ejecución exitosa"""
        try:
            temp_files = [
                self.metadata_dir / "progress.json",
                self.metadata_dir / "detailed_progress.json"
            ]
            for file in temp_files:
                if file.exists():
                    file.unlink()
        except Exception as e:
            self.logger.warning(f"Error limpiando archivos temporales: {e}")

    def _is_known_failed_url(self, url: str) -> bool:
        """Verifica si una URL ya ha fallado anteriormente"""
        failed_urls = self._get_failed_urls()
        return any(failed['url'] == url for failed in failed_urls)

async def main():
    try:
        print(f"\n{'='*80}")
        print("PARSEO DE DOCUMENTACIÓN API")
        print(f"{'='*80}\n")
        
        # Obtener última URL parseada
        last_url = APICrawler.get_last_docs()
        if last_url:
            print(f"Última documentación parseada: {last_url}")
        
        # Solicitar URL al usuario
        while True:
            url = input("\nIngresá la URL de la documentación (o presioná Enter para usar la última): ").strip()
            
            if not url and last_url:
                url = last_url
                print(f"Usando última URL: {url}")
            elif not url:
                print("Error: Debés ingresar una URL")
                continue
            
            try:
                # Crear instancia del crawler para validar URL
                crawler = APICrawler(url)
                
                # Verificar si ya existe documentación
                status = crawler.check_docs_status()
                failed_urls = crawler._get_failed_urls()
                
                if status["total_files"] > 0:
                    print(f"\nYa existe documentación para esta URL:")
                    print(f"- Archivos descargados: {status['total_files']}")
                    print(f"- URLs esperadas: {status['total_expected']}")
                    print(f"- URLs fallidas: {status['failed_urls']}")
                    print(f"- Última actualización: {status['last_update']}")
                    print(f"- Documentación completa: {'Sí' if status['is_complete'] else 'No'}")
                    if status['is_complete']:
                        print("  (Se han descargado todos los archivos posibles)")
                    if failed_urls:
                        print("  Las URLs fallidas se guardaron en metadata/failed_urls.json")
                    
                    action = None
                    if status['is_complete']:
                        print("\nOpciones:")
                        print("1. Volver a parsear todo desde cero")
                        print("2. Cancelar")
                        
                        while True:
                            action = input("\nElegí una opción (1-2): ").strip()
                            if action in ['1', '2']:
                                break
                            print("Opción inválida. Elegí 1 o 2.")
                        
                        if action == '2':
                            print("\nOperación cancelada")
                            return
                        action = '2'  # Si eligió 1, lo convertimos a 2 para borrar todo
                    else:
                        print("\nOpciones:")
                        print("1. Continuar desde donde quedó (solo descarga lo que falta)")
                        print("2. Volver a parsear todo desde cero")
                        print("3. Cancelar")
                        
                        while True:
                            action = input("\nElegí una opción (1-3): ").strip()
                            if action in ['1', '2', '3']:
                                break
                            print("Opción inválida. Elegí 1, 2 o 3.")
                        
                        if action == '3':
                            print("\nOperación cancelada")
                            return
                    
                    if action == '2':
                        # Borrar archivos existentes para empezar de cero
                        for file in crawler.endpoints_dir.glob('*.json'):
                            file.unlink()
                        for file in crawler.metadata_dir.glob('*.json'):
                            file.unlink()
                        print("\nArchivos previos eliminados. Iniciando parseo completo...")
                
                print(f"\nIniciando parseo de {url}...")
                await crawler.crawl()
                return  # Es mejor usar return para salir de la función
                
            except ValueError as e:
                print(f"Error: {str(e)}")
            except Exception as e:
                print(f"Error inesperado: {str(e)}")
                raise
        
    except KeyboardInterrupt:
        print("\n\nProceso interrumpido por el usuario. Los archivos descargados hasta ahora están guardados.")
    except Exception as e:
        print(f"\n\nError inesperado: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
