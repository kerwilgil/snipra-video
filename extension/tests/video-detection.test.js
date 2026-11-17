const { JSDOM } = require('jsdom');

const videoDetection = require('../video-detection.js');

function createDom(html) {
  const dom = new JSDOM(html, { url: 'https://example.com/page', pretendToBeVisual: true });
  global.document = dom.window.document;
  global.window = dom.window;
  global.navigator = dom.window.navigator;
  global.HTMLElement = dom.window.HTMLElement;
  global.HTMLVideoElement = dom.window.HTMLVideoElement;
  global.HTMLSourceElement = dom.window.HTMLSourceElement;
  global.HTMLIFrameElement = dom.window.HTMLIFrameElement;
  global.HTMLMetaElement = dom.window.HTMLMetaElement;
  global.HTMLLinkElement = dom.window.HTMLLinkElement;
  global.HTMLScriptElement = dom.window.HTMLScriptElement;
  global.NodeList = dom.window.NodeList;
  global.Element = dom.window.Element;
  global.Node = dom.window.Node;
}

function runDetectionInDom() {
  return videoDetection.runDetection();
}

function assertEqual(actual, expected, message) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(`${message}\nExpected: ${JSON.stringify(expected)}\nActual: ${JSON.stringify(actual)}`);
  }
}

function assertTrue(condition, message) {
  if (!condition) throw new Error(message);
}

function assertFalse(condition, message) {
  if (condition) throw new Error(message);
}

console.log('Running video-detection tests...\n');

let passed = 0;
let failed = 0;

function test(name, fn) {
  try {
    fn();
    console.log(`  ✓ ${name}`);
    passed++;
  } catch (e) {
    console.log(`  ✗ ${name}`);
    console.log(`    ${e.message}`);
    failed++;
  }
}

// 1. página sin video
test('página sin video', () => {
  createDom('<html><head><title>Test</title></head><body><p>Hola</p></body></html>');
  const result = runDetectionInDom();
  assertEqual(result.candidates.length, 0);
  assertEqual(result.primaryCandidate, null);
  assertEqual(result.pageUrl, 'https://example.com/page');
});

// 2. canonical URL
test('canonical URL', () => {
  createDom('<html><head><link rel="canonical" href="https://canonical.example/video123"></head><body></body></html>');
  const result = runDetectionInDom();
  assertEqual(result.canonicalUrl, 'https://canonical.example/video123');
});

// 3. video src
test('video src', () => {
  createDom('<html><body><video src="https://example.com/video.mp4"></video></body></html>');
  const result = runDetectionInDom();
  assertEqual(result.candidates.length, 1);
  assertEqual(result.candidates[0].url, 'https://example.com/video.mp4');
  assertEqual(result.candidates[0].type, 'HTML5_VIDEO');
  assertTrue(result.candidates[0].usable);
});

// 4. video currentSrc (simulado - en jsdom currentSrc === src)
test('video currentSrc', () => {
  createDom('<html><body><video src="https://example.com/video.mp4"></video></body></html>');
  const result = runDetectionInDom();
  assertTrue(result.candidates.some(c => c.url === 'https://example.com/video.mp4' && c.type === 'HTML5_VIDEO'));
});

// 5. video + source
test('video + source', () => {
  createDom('<html><body><video><source src="https://example.com/video.webm"></video></body></html>');
  const result = runDetectionInDom();
  assertEqual(result.candidates.length, 1);
  assertEqual(result.candidates[0].url, 'https://example.com/video.webm');
  assertEqual(result.candidates[0].type, 'HTML5_SOURCE');
});

// 6. candidatos duplicados
test('candidatos duplicados', () => {
  createDom('<html><body><video src="https://example.com/video.mp4"></video><video src="https://example.com/video.mp4"></video></body></html>');
  const result = runDetectionInDom();
  assertEqual(result.candidates.length, 1);
});

// 7. blob candidate unusable
test('blob candidate unusable', () => {
  createDom('<html><body><video src="blob:https://example.com/abc-123"></video></body></html>');
  const result = runDetectionInDom();
  assertEqual(result.candidates.length, 1);
  assertEqual(result.candidates[0].type, 'BLOB_URL');
  assertFalse(result.candidates[0].usable);
});

// 8. OpenGraph
test('OpenGraph og:video', () => {
  createDom('<html><head><meta property="og:video" content="https://og.example/video.mp4"></head><body></body></html>');
  const result = runDetectionInDom();
  assertEqual(result.candidates.length, 1);
  assertEqual(result.candidates[0].url, 'https://og.example/video.mp4');
  assertEqual(result.candidates[0].type, 'OPEN_GRAPH_VIDEO');
  assertTrue(result.candidates[0].usable);
});

// 9. JSON-LD VideoObject simple
test('JSON-LD VideoObject simple', () => {
  createDom('<html><head><script type="application/ld+json">{"@type":"VideoObject","contentUrl":"https://jsonld.example/video.mp4","name":"Test Video"}</script></head><body></body></html>');
  const result = runDetectionInDom();
  assertEqual(result.candidates.length, 1);
  assertEqual(result.candidates[0].url, 'https://jsonld.example/video.mp4');
  assertEqual(result.candidates[0].type, 'JSON_LD_VIDEO');
});

// 10. JSON-LD array
test('JSON-LD array', () => {
  createDom('<html><head><script type="application/ld+json">[{"@type":"VideoObject","contentUrl":"https://a.example/v.mp4"},{"@type":"VideoObject","contentUrl":"https://b.example/v.mp4"}]</script></head><body></body></html>');
  const result = runDetectionInDom();
  assertEqual(result.candidates.length, 2);
});

// 11. JSON-LD @graph
test('JSON-LD @graph', () => {
  createDom('<html><head><script type="application/ld+json">{"@graph":[{"@type":"VideoObject","contentUrl":"https://graph.example/v.mp4"}]}</script></head><body></body></html>');
  const result = runDetectionInDom();
  assertEqual(result.candidates.length, 1);
  assertEqual(result.candidates[0].url, 'https://graph.example/v.mp4');
});

// 12. JSON-LD inválido
test('JSON-LD inválido no rompe', () => {
  createDom('<html><head><script type="application/ld+json">{invalid json</script></head><body></body></html>');
  const result = runDetectionInDom();
  assertEqual(result.candidates.length, 0);
});

// 13. iframe YouTube
test('iframe YouTube', () => {
  createDom('<html><body><iframe src="https://www.youtube.com/embed/abc123"></iframe></body></html>');
  const result = runDetectionInDom();
  assertEqual(result.candidates.length, 1);
  assertEqual(result.candidates[0].url, 'https://www.youtube.com/embed/abc123');
  assertEqual(result.candidates[0].type, 'EMBED_IFRAME');
});

// 14. iframe Vimeo
test('iframe Vimeo', () => {
  createDom('<html><body><iframe src="https://player.vimeo.com/video/123456"></iframe></body></html>');
  const result = runDetectionInDom();
  assertEqual(result.candidates.length, 1);
  assertEqual(result.candidates[0].type, 'EMBED_IFRAME');
});

// 15. múltiples candidatos
test('múltiples candidatos', () => {
  createDom('<html><head><meta property="og:video" content="https://og.example/v.mp4"></head><body><video src="https://html5.example/v.mp4"></video></body></html>');
  const result = runDetectionInDom();
  assertTrue(result.candidates.length >= 2);
});

// 16. URLs no utilizables
test('URLs no utilizables filtradas', () => {
  createDom('<html><body><video src="javascript:alert(1)"></video><video src="data:text/plain,foo"></video><video src="file:///test.mp4"></video></body></html>');
  const result = runDetectionInDom();
  assertEqual(result.candidates.length, 0);
});

// 17. límite máximo de candidatos
test('límite máximo de candidatos', () => {
  let html = '<html><body>';
  for (let i = 0; i < 30; i++) {
    html += `<video src="https://example.com/v${i}.mp4"></video>`;
  }
  html += '</body></html>';
  createDom(html);
  const result = runDetectionInDom();
  assertTrue(result.candidates.length <= 25);
});

// 18. duration NaN/Infinity
test('duration NaN/Infinity ignorada', () => {
  createDom('<html><body><video src="https://example.com/v.mp4"></video></body></html>');
  const result = runDetectionInDom();
  const c = result.candidates[0];
  assertTrue(c.duration === null || Number.isFinite(c.duration));
});

// 19. documento sin canonical
test('documento sin canonical', () => {
  createDom('<html><body><video src="https://example.com/v.mp4"></video></body></html>');
  const result = runDetectionInDom();
  assertEqual(result.canonicalUrl, null);
});

// 20. título vacío
test('título vacío', () => {
  createDom('<html><head></head><body><video src="https://example.com/v.mp4"></video></body></html>');
  const result = runDetectionInDom();
  assertEqual(result.candidates[0].title, '');
});

console.log(`\n--- Results: ${passed} passed, ${failed} failed ---`);

if (failed > 0) {
  process.exit(1);
}