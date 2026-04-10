"""
Outil WebSearch pour le Brain ReAct
Utilise l'API WebSearch pour rechercher des informations en temps réel
"""
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass
import os


@dataclass
class SearchResult:
    """Résultat de recherche web"""
    title: str
    url: str
    snippet: Optional[str] = None


class WebSearchTool:
    """Outil de recherche web pour le Brain ReAct"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialise l'outil WebSearch

        Args:
            api_key: Clé API pour le service de recherche (optionnel)
        """
        self.api_key = api_key or os.getenv("WEBSEARCH_API_KEY") or os.getenv("TAVILY_API_KEY")
        self.search_count = 0

    def search(
        self,
        query: str,
        max_results: int = 5,
        allowed_domains: Optional[List[str]] = None,
        blocked_domains: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Effectue une recherche web

        Args:
            query: La requête de recherche
            max_results: Nombre maximum de résultats (défaut: 5)
            allowed_domains: Domaines autorisés uniquement
            blocked_domains: Domaines à bloquer

        Returns:
            Liste de SearchResult
        """
        self.search_count += 1

        # Si nous avons une clé API Tavily, utilisons-la
        if self.api_key:
            return self._search_with_tavily(query, max_results, allowed_domains, blocked_domains)

        # Sinon, utilisation d'une méthode alternative (DuckDuckGo, etc.)
        return self._search_with_duckduckgo(query, max_results, allowed_domains, blocked_domains)

    def _search_with_tavily(
        self,
        query: str,
        max_results: int,
        allowed_domains: Optional[List[str]],
        blocked_domains: Optional[List[str]]
    ) -> List[SearchResult]:
        """Recherche via l'API Tavily"""
        try:
            url = "https://api.tavily.com/search"
            headers = {"Content-Type": "application/json"}

            payload = {
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
                "include_raw_content": False,
            }

            if allowed_domains:
                payload["include_domains"] = allowed_domains
            if blocked_domains:
                payload["exclude_domains"] = blocked_domains

            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []
            for result in data.get("results", []):
                results.append(SearchResult(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    snippet=result.get("content", "")
                ))

            return results

        except Exception as e:
            print(f"⚠️ Erreur Tavily API: {str(e)}")
            return self._search_with_duckduckgo(query, max_results, allowed_domains, blocked_domains)

    def _search_with_duckduckgo(
        self,
        query: str,
        max_results: int,
        allowed_domains: Optional[List[str]],
        blocked_domains: Optional[List[str]]
    ) -> List[SearchResult]:
        """Recherche via DuckDuckGo (fallback gratuit)"""
        try:
            from duckduckgo_search import DDGS

            ddgs = DDGS()
            search_results = ddgs.text(query, max_results=max_results)

            results = []
            for result in search_results:
                url = result.get("link", "")
                title = result.get("title", "")
                snippet = result.get("body", "")

                # Filtrage par domaines
                if allowed_domains:
                    if not any(domain in url for domain in allowed_domains):
                        continue

                if blocked_domains:
                    if any(domain in url for domain in blocked_domains):
                        continue

                results.append(SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet
                ))

            return results

        except ImportError:
            # Si duckduckgo_search n'est pas installé, retourne un résultat simulé
            print("⚠️ duckduckgo_search non installé. Installation: pip install duckduckgo-search")
            return [SearchResult(
                title="Installation requise",
                url="https://pypi.org/project/duckduckgo-search/",
                snippet="Installez duckduckgo-search pour utiliser la recherche web: pip install duckduckgo-search"
            )]
        except Exception as e:
            print(f"⚠️ Erreur recherche DuckDuckGo: {str(e)}")
            return []

    def format_results_for_llm(self, results: List[SearchResult], query: str) -> str:
        """
        Formate les résultats de recherche pour être passés au LLM

        Args:
            results: Liste des résultats de recherche
            query: La requête originale

        Returns:
            Chaîne formatée pour le LLM
        """
        if not results:
            return f"Aucun résultat trouvé pour la requête : {query}"

        formatted = f"Résultats de recherche pour '{query}':\n\n"

        for i, result in enumerate(results, 1):
            formatted += f"{i}. {result.title}\n"
            formatted += f"   URL: {result.url}\n"
            if result.snippet:
                formatted += f"   Extrait: {result.snippet[:200]}...\n"
            formatted += "\n"

        return formatted

    def get_stats(self) -> Dict[str, int]:
        """Retourne les statistiques d'utilisation"""
        return {
            "total_searches": self.search_count
        }


# Fonction helper pour utilisation directe
def web_search(
    query: str,
    max_results: int = 5,
    api_key: Optional[str] = None
) -> str:
    """
    Fonction helper pour recherche web rapide

    Args:
        query: La requête de recherche
        max_results: Nombre max de résultats
        api_key: Clé API optionnelle

    Returns:
        Résultats formatés pour le LLM
    """
    tool = WebSearchTool(api_key)
    results = tool.search(query, max_results)
    return tool.format_results_for_llm(results, query)


if __name__ == "__main__":
    import sys

    # Configuration UTF-8
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except:
            pass

    # Test rapide
    print("Test de WebSearchTool\n")

    tool = WebSearchTool()

    # Test avec une requête simple
    query = "Python langage de programmation"
    print(f"Recherche : {query}\n")
    print("-" * 60)

    results = tool.search(query, max_results=3)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.title}")
        print(f"   URL: {result.url}")
        if result.snippet:
            print(f"   {result.snippet[:150]}...")

    print("\n" + "-" * 60)
    print(f"\n📊 Statistiques: {tool.get_stats()}")
