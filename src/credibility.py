"""
credibility.py
Source credibility analysis engine.
Scores news articles based on:
  1. Domain reputation  (known credible / known fake domains)
  2. URL pattern analysis (clickbait patterns, suspicious TLDs)
  3. Author credibility heuristics
  4. Headline analysis (ALL CAPS, excessive punctuation, sensationalism)

Final credibility score: 0 (very suspicious) → 100 (very credible)
Combined with ML model: final_score = 0.6 × ML + 0.4 × credibility
"""

import re
import urllib.parse
from typing import Optional


# ─── Domain Reputation Lists ─────────────────────────────────────────────────

CREDIBLE_DOMAINS = {
    # Tier 1: Highly credible international sources
    'reuters.com', 'apnews.com', 'bbc.com', 'bbc.co.uk', 'theguardian.com',
    'nytimes.com', 'washingtonpost.com', 'economist.com', 'ft.com',
    'wsj.com', 'bloomberg.com', 'npr.org', 'pbs.org', 'cbsnews.com',
    'nbcnews.com', 'abcnews.go.com', 'cnn.com', 'time.com',
    # Tier 1: Indian credible sources
    'thehindu.com', 'ndtv.com', 'hindustantimes.com', 'timesofindia.indiatimes.com',
    'indianexpress.com', 'livemint.com', 'business-standard.com',
    'theprint.in', 'thewire.in', 'scroll.in',
    # Fact-check sources
    'snopes.com', 'factcheck.org', 'politifact.com', 'altnews.in',
    'boomlive.in', 'factly.in', 'vishvasnews.com',
    # Academic / government
    'who.int', 'cdc.gov', 'nih.gov', 'pib.gov.in', 'mygov.in',
}

SUSPICIOUS_DOMAINS = {
    'beforeitsnews.com', 'worldnewsdailyreport.com', 'infowars.com',
    'naturalnews.com', 'yournewswire.com', 'newspunch.com',
    'realnewsrightnow.com', 'empirenews.net', 'theonion.com',  # satire
    'dailybuzzlive.com', 'newsexaminer.net', 'huzlers.com',
    'nationalreport.net', 'thevalleyreport.com', 'nbc.com.co',
    'abcnews.com.co', 'cnn.com.de',  # domain spoofing
}

SUSPICIOUS_TLDS = {'.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top',
                   '.click', '.loan', '.racing', '.win', '.webcam'}

CREDIBLE_TLDS = {'.gov', '.edu', '.ac.uk', '.ac.in', '.org'}


# ─── Headline Clickbait Patterns ─────────────────────────────────────────────

CLICKBAIT_PATTERNS = [
    r'\bYOU WON\'T BELIEVE\b',
    r'\bSHOCKING\b',
    r'\bBREAKING\b.{0,10}\bBREAKING\b',  # repeated BREAKING
    r'\b\d+\s+THINGS\b',
    r'\bDOCTORS HATE\b',
    r'\bGOVERNMENT DOESN\'T WANT\b',
    r'\bTHEY DON\'T WANT YOU TO KNOW\b',
    r'\bSECRET\s+THEY\b',
    r'!!!',  # multiple exclamation marks
    r'\?\?\?',
    r'[A-Z]{4,}',  # 4+ consecutive caps words (shouting)
]

SENSATIONALIST_WORDS = {
    'shocking', 'bombshell', 'explosive', 'cover-up', 'conspiracy',
    'secret', 'exposed', 'leaked', 'banned', 'censored', 'suppressed',
    'cure', 'miracle', 'hoax', 'fraud', 'lie', 'fake', 'rigged',
    'stolen', 'corrupt', 'crisis', 'urgent', 'emergency', 'breaking',
}


# ─── Core Scorer ─────────────────────────────────────────────────────────────

class CredibilityScorer:
    """
    Scores the credibility of a news article from 0 (suspicious) to 100 (credible).
    """

    def score(self, text: str, url: Optional[str] = None,
              headline: Optional[str] = None,
              author: Optional[str] = None) -> dict:
        """
        Compute full credibility score.

        Args:
            text:     Article body text
            url:      Article URL (optional but recommended)
            headline: Article headline (optional)
            author:   Author name (optional)

        Returns:
            dict:
                overall_score  — 0–100 (higher = more credible)
                domain_score   — 0–100
                url_score      — 0–100
                headline_score — 0–100
                text_score     — 0–100
                verdict        — 'CREDIBLE' | 'SUSPICIOUS' | 'UNKNOWN'
                flags          — list of warning strings
                breakdown      — dict of individual component scores
        """
        flags = []
        components = {}

        # 1. Domain scoring
        domain_result = self._score_domain(url)
        components['domain'] = domain_result['score']
        flags.extend(domain_result['flags'])

        # 2. URL pattern analysis
        url_result = self._score_url(url)
        components['url'] = url_result['score']
        flags.extend(url_result['flags'])

        # 3. Headline analysis
        headline_result = self._score_headline(headline or '')
        components['headline'] = headline_result['score']
        flags.extend(headline_result['flags'])

        # 4. Text quality analysis
        text_result = self._score_text(text)
        components['text'] = text_result['score']
        flags.extend(text_result['flags'])

        # Weighted average
        # Domain is most important; URL next; headline and text support
        weights = {'domain': 0.40, 'url': 0.20, 'headline': 0.20, 'text': 0.20}
        overall = sum(components[k] * weights[k] for k in components)
        overall = round(overall, 1)

        if overall >= 65:
            verdict = 'CREDIBLE'
        elif overall >= 40:
            verdict = 'UNKNOWN'
        else:
            verdict = 'SUSPICIOUS'

        return {
            'overall_score': overall,
            'verdict': verdict,
            'flags': list(set(flags)),
            'breakdown': components,
            'domain': domain_result.get('domain', 'unknown'),
        }

    def _score_domain(self, url: Optional[str]) -> dict:
        if not url:
            return {'score': 50, 'flags': ['No URL provided — domain unknown'], 'domain': 'unknown'}

        try:
            parsed = urllib.parse.urlparse(url if '://' in url else 'https://' + url)
            domain = parsed.netloc.lower().lstrip('www.')
        except Exception:
            return {'score': 50, 'flags': ['Could not parse URL'], 'domain': 'unknown'}

        flags = []

        # Check credible list (exact and subdomain match)
        for credible in CREDIBLE_DOMAINS:
            if domain == credible or domain.endswith('.' + credible):
                return {'score': 95, 'flags': [], 'domain': domain}

        # Check suspicious list
        for suspicious in SUSPICIOUS_DOMAINS:
            if domain == suspicious or domain.endswith('.' + suspicious):
                flags.append(f'Domain "{domain}" is on the suspicious sources list')
                return {'score': 5, 'flags': flags, 'domain': domain}

        # Check TLD
        tld = '.' + domain.split('.')[-1]
        if tld in SUSPICIOUS_TLDS:
            flags.append(f'Suspicious TLD: {tld}')
            return {'score': 25, 'flags': flags, 'domain': domain}

        if tld in CREDIBLE_TLDS:
            return {'score': 80, 'flags': [], 'domain': domain}

        # Domain spoofing detection: looks like a real domain but isn't
        for credible in ['nytimes', 'bbc', 'cnn', 'reuters', 'ndtv', 'thehindu']:
            if credible in domain and domain not in CREDIBLE_DOMAINS:
                flags.append(f'Domain may be spoofing "{credible}" — verify carefully')
                return {'score': 15, 'flags': flags, 'domain': domain}

        return {'score': 50, 'flags': ['Unknown domain — credibility unverified'], 'domain': domain}

    def _score_url(self, url: Optional[str]) -> dict:
        if not url:
            return {'score': 50, 'flags': []}

        url_lower = url.lower()
        flags = []
        score = 70  # start neutral-good

        # Suspicious URL patterns
        if re.search(r'\d{5,}', url):
            score -= 10  # lots of numbers in URL
        if url.count('-') > 8:
            flags.append('URL contains excessive hyphens (clickbait pattern)')
            score -= 15
        if any(kw in url_lower for kw in ['shocking', 'viral', 'breaking', 'secret', 'exposed']):
            flags.append('URL contains sensationalist keywords')
            score -= 20
        if url.count('/') > 7:
            score -= 5  # deep nested URL

        # Positive signals
        if re.search(r'/\d{4}/\d{2}/\d{2}/', url):
            score += 10  # dated article URL (structured publishing)
        if any(url_lower.startswith(p) for p in ['https://']):
            score += 5  # HTTPS

        return {'score': max(0, min(100, score)), 'flags': flags}

    def _score_headline(self, headline: str) -> dict:
        if not headline:
            return {'score': 50, 'flags': []}

        flags = []
        score = 80

        # ALL CAPS words
        caps_words = re.findall(r'\b[A-Z]{4,}\b', headline)
        if len(caps_words) >= 2:
            flags.append(f'Headline uses excessive ALL CAPS: {", ".join(caps_words[:3])}')
            score -= 25

        # Clickbait patterns
        for pattern in CLICKBAIT_PATTERNS:
            if re.search(pattern, headline, re.IGNORECASE):
                flags.append('Headline matches clickbait pattern')
                score -= 20
                break

        # Excessive punctuation
        if headline.count('!') >= 2 or headline.count('?') >= 2:
            flags.append('Excessive punctuation in headline')
            score -= 15

        # Sensationalist words
        headline_lower = headline.lower()
        found_sensational = [w for w in SENSATIONALIST_WORDS if w in headline_lower]
        if len(found_sensational) >= 2:
            flags.append(f'Multiple sensationalist words: {", ".join(found_sensational[:3])}')
            score -= 20
        elif len(found_sensational) == 1:
            score -= 8

        return {'score': max(0, min(100, score)), 'flags': flags}

    def _score_text(self, text: str) -> dict:
        if not text or len(text) < 50:
            return {'score': 40, 'flags': ['Article text too short']}

        flags = []
        score = 75

        # Very short articles are suspicious
        word_count = len(text.split())
        if word_count < 100:
            flags.append(f'Very short article ({word_count} words) — may lack substance')
            score -= 20
        elif word_count > 500:
            score += 5  # long articles tend to be more credible

        # Sensationalist word density
        text_lower = text.lower()
        sensational_count = sum(1 for w in SENSATIONALIST_WORDS if w in text_lower)
        density = sensational_count / max(word_count / 100, 1)
        if density > 3:
            flags.append('High density of sensationalist language in text body')
            score -= 20
        elif density > 1.5:
            score -= 10

        # First-person claims (opinion masquerading as news)
        first_person = len(re.findall(r'\b(I believe|I think|in my opinion|we must|you need to)\b',
                                       text_lower))
        if first_person >= 3:
            flags.append('Article uses many opinion phrases as if factual')
            score -= 15

        # External source citations (positive signal)
        citations = len(re.findall(r'\b(according to|said|reported|confirmed by|study shows|researchers)\b',
                                    text_lower))
        if citations >= 3:
            score += 10
        elif citations == 0:
            flags.append('No citations or attributed sources found')
            score -= 10

        return {'score': max(0, min(100, score)), 'flags': flags}


# ─── Combined Final Score ─────────────────────────────────────────────────────

def combined_score(ml_fake_prob: float, credibility_score: float,
                   ml_weight: float = 0.6, cred_weight: float = 0.4) -> dict:
    """
    Combine ML model output with credibility score for final verdict.

    Args:
        ml_fake_prob:      Probability of being fake from ML model (0–1)
        credibility_score: Source credibility score (0–100)
        ml_weight:         Weight for ML model in final score
        cred_weight:       Weight for credibility score

    Returns:
        dict with final_fake_prob, final_verdict, confidence_label
    """
    # Normalize credibility to fake probability: high credibility → low fake prob
    cred_fake_prob = 1 - (credibility_score / 100)

    final_fake_prob = ml_weight * ml_fake_prob + cred_weight * cred_fake_prob
    final_fake_prob = round(final_fake_prob, 4)

    if final_fake_prob >= 0.7:
        verdict = 'FAKE'
        confidence_label = 'High confidence'
    elif final_fake_prob >= 0.5:
        verdict = 'FAKE'
        confidence_label = 'Moderate confidence'
    elif final_fake_prob >= 0.35:
        verdict = 'UNCERTAIN'
        confidence_label = 'Low confidence'
    else:
        verdict = 'REAL'
        confidence_label = 'High confidence' if final_fake_prob < 0.2 else 'Moderate confidence'

    return {
        'final_fake_prob': final_fake_prob,
        'final_real_prob': round(1 - final_fake_prob, 4),
        'final_verdict': verdict,
        'confidence_label': confidence_label,
        'ml_contribution': round(ml_weight * ml_fake_prob, 4),
        'credibility_contribution': round(cred_weight * cred_fake_prob, 4),
    }
