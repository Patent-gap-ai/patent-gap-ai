from urllib.parse import urljoin

import requests


class WebScraper:
    def __init__(
        self, session: requests.Session | None = None, timeout: float = 10
    ) -> None:
        self.session = session or requests.Session()
        self.timeout = timeout

    def get(self, url: str) -> str:
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.text

    @staticmethod
    def resolve_url(base_url: str, url: str | None) -> str | None:
        if not url:
            return None
        return urljoin(base_url, url)
