'use strict';

// A fresh scope makes on-demand injection safe when the popup is reopened.
(() => {
const MAX_CANDIDATES = 25;
const KNOWN_EMBED_DOMAINS = [
  'youtube.com',
  'youtu.be',
  'vimeo.com',
  'player.vimeo.com',
  'twitch.tv',
  'player.twitch.tv',
];

function isUsableUrl(url) {
    if (!url || typeof url !== 'string') return false;
    try {
      const u = new URL(url);
      return u.protocol === 'http:' || u.protocol === 'https:';
    } catch {
      return false;
    }
  }

  function normalizeUrl(url) {
    try {
      const u = new URL(url);
      u.hash = '';
      return u.toString();
    } catch {
      return url;
    }
  }

  function deduplicateCandidates(candidates) {
    const seen = new Set();
    const result = [];
    for (const c of candidates) {
      const key = normalizeUrl(c.url);
      if (!seen.has(key)) {
        seen.add(key);
        result.push(c);
      }
    }
    return result;
  }

  function getCanonicalUrl() {
    const link = document.querySelector('link[rel="canonical"]');
    if (link && link.href && isUsableUrl(link.href)) {
      return link.href;
    }
    return null;
  }

  function getTitle() {
    const ogTitle = document.querySelector('meta[property="og:title"]');
    if (ogTitle && ogTitle.content) return ogTitle.content.trim();
    const twitterTitle = document.querySelector('meta[name="twitter:title"]');
    if (twitterTitle && twitterTitle.content) return twitterTitle.content.trim();
    return document.title || '';
  }

  function collectHtml5Videos() {
    const candidates = [];
    const videos = document.querySelectorAll('video');
    for (const video of videos) {
      const src = video.currentSrc || video.src;
      if (isUsableUrl(src)) {
        candidates.push({
          url: src,
          type: 'HTML5_VIDEO',
          title: getTitle(),
          duration: Number.isFinite(video.duration) ? video.duration : null,
          poster: video.poster || null,
          usable: true,
        });
      }
      const sources = video.querySelectorAll('source');
      for (const source of sources) {
        if (source.src && isUsableUrl(source.src)) {
          candidates.push({
            url: source.src,
            type: 'HTML5_SOURCE',
            title: getTitle(),
            duration: Number.isFinite(video.duration) ? video.duration : null,
            poster: video.poster || null,
            usable: true,
          });
        }
      }
    }
    return candidates;
  }

  function collectOpenGraphVideos() {
    const candidates = [];
    const properties = ['og:video', 'og:video:url', 'og:video:secure_url'];
    for (const prop of properties) {
      const meta = document.querySelector(`meta[property="${prop}"]`);
      if (meta && meta.content && isUsableUrl(meta.content)) {
        candidates.push({
          url: meta.content,
          type: 'OPEN_GRAPH_VIDEO',
          title: getTitle(),
          duration: null,
          poster: null,
          usable: true,
        });
      }
    }
    return candidates;
  }

  function collectJsonLdVideos() {
    const candidates = [];
    const scripts = document.querySelectorAll('script[type="application/ld+json"]');
    for (const script of scripts) {
      try {
        const data = JSON.parse(script.textContent);
        const items = Array.isArray(data) ? data : [data];
        for (const item of items) {
          processJsonLdItem(item, candidates);
        }
      } catch {
        // JSON malformado, ignorar silenciosamente
      }
    }
    return candidates;
  }

  function processJsonLdItem(item, candidates, depth = 0) {
    if (!item || typeof item !== 'object' || depth > 3) return;
    if (Array.isArray(item)) {
      for (const i of item) processJsonLdItem(i, candidates, depth + 1);
      return;
    }
    if (item['@graph'] && Array.isArray(item['@graph'])) {
      for (const g of item['@graph']) processJsonLdItem(g, candidates, depth + 1);
      return;
    }
    const type = item['@type'];
    const types = Array.isArray(type) ? type : [type];
    if (types.includes('VideoObject')) {
      const url = item.contentUrl || item.embedUrl;
      if (isUsableUrl(url)) {
        candidates.push({
          url,
          type: 'JSON_LD_VIDEO',
          title: item.name || getTitle(),
          duration: item.duration ? parseISODuration(item.duration) : null,
          poster: item.thumbnailUrl || null,
          usable: true,
        });
      }
    }
  }

  function parseISODuration(iso) {
    const match = iso.match(/PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?/);
    if (!match) return null;
    const h = parseInt(match[1] || '0', 10);
    const m = parseInt(match[2] || '0', 10);
    const s = parseFloat(match[3] || '0');
    return h * 3600 + m * 60 + s;
  }

  function collectIframeEmbeds() {
    const candidates = [];
    const iframes = document.querySelectorAll('iframe[src]');
    for (const iframe of iframes) {
      const src = iframe.src;
      if (!isUsableUrl(src)) continue;
      try {
        const url = new URL(src);
        const hostname = url.hostname.replace(/^www\./, '');
        if (KNOWN_EMBED_DOMAINS.some(d => hostname === d || hostname.endsWith('.' + d))) {
          candidates.push({
            url: src,
            type: 'EMBED_IFRAME',
            title: getTitle(),
            duration: null,
            poster: null,
            usable: true,
          });
        }
      } catch {
        // URL inválida, ignorar
      }
    }
    return candidates;
  }

  function detectBlobVideos() {
    const candidates = [];
    const videos = document.querySelectorAll('video');
    for (const video of videos) {
      const src = video.currentSrc || video.src;
      if (src && src.startsWith('blob:')) {
        candidates.push({
          url: src,
          type: 'BLOB_URL',
          title: getTitle(),
          duration: Number.isFinite(video.duration) ? video.duration : null,
          poster: video.poster || null,
          usable: false,
        });
      }
    }
    return candidates;
  }

  function runDetection() {
  const pageUrl = window.location.href;
  const canonicalUrl = getCanonicalUrl();
  const title = getTitle();

  let allCandidates = [];
  allCandidates.push(...collectHtml5Videos());
  allCandidates.push(...collectOpenGraphVideos());
  allCandidates.push(...collectJsonLdVideos());
  allCandidates.push(...collectIframeEmbeds());
  allCandidates.push(...detectBlobVideos());

  allCandidates = deduplicateCandidates(allCandidates);

  if (allCandidates.length > MAX_CANDIDATES) {
    allCandidates = allCandidates.slice(0, MAX_CANDIDATES);
  }

  const usableCandidates = allCandidates.filter(c => c.usable);
  const primaryCandidate = usableCandidates[0] || null;

  const hostname = new URL(pageUrl).hostname;
  const instagram = ['instagram.com', 'www.instagram.com', 'm.instagram.com'].includes(hostname);
  const path = new URL(pageUrl).pathname;
  const reel = /^\/reels?\/[^/]+\/?$/.test(path);
  const post = /^\/p\/[^/]+\/?$/.test(path);
  const story = /^\/stories\/[^/]+(?:\/\d+)?\/?$/.test(path);
  const authRequired = instagram && Boolean(
    document.querySelector('input[name="username"], input[name="password"], form[action*="login"]')
    || path.startsWith('/accounts/login')
  );
  const instagramContent = reel || post || story;
  const supportedPage = platformPage(hostname, path, pageUrl);
  const videoDetected = instagram
    ? instagramContent && !authRequired && (reel || usableCandidates.length > 0 || Boolean(document.querySelector('video')))
    : usableCandidates.length > 0 || supportedPage;
  // Page context can identify a lazy-loaded Reel; profiles/explore never become
  // download targets just because Instagram renders a suggested video there.
  const platform = instagram ? 'Instagram'
    : /(^|\.)(youtube\.com|youtu\.be)$/.test(hostname) ? 'YouTube'
    : /(^|\.)vimeo\.com$/.test(hostname) ? 'Vimeo' : 'Página web';
  const context = {
    platform,
    contentType: instagram ? (reel ? 'reel' : post ? 'post' : story ? 'story' : 'page') : 'video',
    contentLabel: instagram ? (reel ? 'Reel' : post ? (videoDetected ? 'Publicación con video' : 'Publicación') : story ? 'Story' : 'Página') : 'Video',
    videoDetected: Boolean(videoDetected),
    downloadable: Boolean(videoDetected && !authRequired),
    authRequired,
    authenticationMayBeRequired: instagram && story,
  };

  function platformPage(host, pathname, url) {
    if (host === 'youtu.be') return pathname.length > 1;
    if (/(^|\.)youtube\.com$/.test(host)) return /^\/(shorts|live)\/[^/]+/.test(pathname) || (pathname === '/watch' && Boolean(new URL(url).searchParams.get('v')));
    if (/(^|\.)tiktok\.com$/.test(host)) return /^\/@[^/]+\/video\/\d+/.test(pathname);
    if (/(^|\.)twitch\.tv$/.test(host)) return /^\/videos\/\d+/.test(pathname) || host === 'clips.twitch.tv';
    if (/(^|\.)facebook\.com$/.test(host)) return /^\/(reel|videos)\/\d+/.test(pathname) || (pathname === '/watch/' && Boolean(new URL(url).searchParams.get('v')));
    return /(^|\.)vimeo\.com$/.test(host) && /^\/(?:video\/)?\d+/.test(pathname);
  }

  return {
    pageUrl,
    canonicalUrl,
    title,
    candidates: allCandidates,
    primaryCandidate,
    context,
    timestamp: Date.now(),
  };
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { runDetection };
} else if (typeof globalThis !== 'undefined') {
  globalThis.SnipraVideoDetection = Object.freeze({ runDetection });
}
})();
