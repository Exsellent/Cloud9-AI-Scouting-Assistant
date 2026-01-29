import logging
import os

import httpx

logger = logging.getLogger("grid_client")


class GRIDClient:
    """
    Hackathon-optimized GRID Esports API Client

    Features:
    - Demo mode for stable hackathon demo
    - Graceful fallback to mock data
    - Clean logging
    - Simple configuration
    """

    def __init__(self):
        self.api_key = os.getenv("GRID_API_KEY")
        self.base_url = "https://api-op.grid.gg/central-data/graphql"

        # Demo mode by default for hackathon stability
        self.demo_mode = os.getenv("DEMO_MODE", "true").lower() == "true"

        # Configurable timeout
        self.timeout = float(os.getenv("GRID_TIMEOUT", "30"))

        if not self.api_key and not self.demo_mode:
            logger.warning("GRID_API_KEY not set → forcing demo mode")
            self.demo_mode = True

        logger.info(
            "GRID Client initialized",
            extra={
                "demo_mode": self.demo_mode,
                "timeout": self.timeout,
                "has_api_key": bool(self.api_key)
            }
        )

    async def _query(self, query: str, variables: dict):
        """Execute GraphQL query with fallback"""
        if self.demo_mode:
            logger.info("Demo mode active — skipping real API call")
            return {"data": {}}

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.base_url,
                    json={"query": query, "variables": variables},
                    headers=headers
                )
                response.raise_for_status()
                logger.info("Successfully queried GRID API")
                return response.json()

        except Exception as e:
            logger.error(f"GRID API error: {e} → falling back to demo data")
            return {"data": {}}

    async def get_team_id(self, team_name: str, game: str = "valorant"):
        """Get team ID by name (real mode only)"""
        if self.demo_mode:
            return None

        query = """
        query SearchTeams($query: String!) {
          searchTeams(query: $query, first: 5) {
            edges {
              node {
                id
                name
                game {
                  name
                }
              }
            }
          }
        }
        """

        result = await self._query(query, {"query": team_name})
        teams = result.get("data", {}).get("searchTeams", {}).get("edges", [])

        for team in teams:
            if team["node"]["name"].lower() == team_name.lower():
                logger.info(f"Found team ID for {team_name}: {team['node']['id']}")
                return team["node"]["id"]

        logger.warning(f"Team {team_name} not found in GRID")
        return None

    async def get_recent_matches(
            self,
            team_name: str,
            game: str = "valorant",
            limit: int = 5
    ):
        """
        Get recent matches for a team

        Returns:
            List[Dict] with match data (real or demo)
        """
        logger.info(
            f"Fetching recent matches for {team_name}",
            extra={"game": game, "limit": limit, "demo_mode": self.demo_mode}
        )

        # Demo mode - immediate return with mock data
        if self.demo_mode:
            logger.info("Using demo mock data (DEMO_MODE=true)")
            return self._get_demo_data(team_name, game, limit)

        # Real API mode
        logger.info("Using real GRID API data (DEMO_MODE=false)")

        try:
            team_id = await self.get_team_id(team_name, game)
            if not team_id:
                logger.warning(f"Team {team_name} not found → fallback to demo")
                return self._get_demo_data(team_name, game, limit)

            query = """
            query GetTeamMatches($teamId: ID!, $limit: Int!) {
              team(id: $teamId) {
                matches(last: $limit) {
                  edges {
                    node {
                      id
                      startTime
                      winner { 
                        name 
                      }
                      games {
                        id
                        map { 
                          name 
                        }
                        teams {
                          side
                          score
                          roster {
                            players {
                              player {
                                name
                              }
                              agent {
                                name
                              }
                              stats {
                                kills
                                deaths
                                assists
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
            """

            result = await self._query(query, {"teamId": team_id, "limit": limit})
            matches_node = result.get("data", {}).get("team", {}).get("matches", {}).get("edges", [])

            matches = []
            for edge in matches_node:
                node = edge["node"]
                games = node.get("games", [])
                if not games:
                    continue

                # Take first game
                game_data = games[0]
                teams_data = game_data.get("teams", [])
                winner_name = node.get("winner", {}).get("name", "")

                # Determine result
                result = "win" if winner_name and winner_name.lower() == team_name.lower() else "loss"

                # Extract agents
                agents = []
                for team in teams_data:
                    for player in team.get("roster", {}).get("players", []):
                        agent_name = player.get("agent", {}).get("name")
                        if agent_name:
                            agents.append(agent_name)
                    if len(agents) >= 5:
                        break

                match = {
                    "match_id": node["id"],
                    "date": node.get("startTime", "2026-01-01")[:10],
                    "map": game_data.get("map", {}).get("name", "Unknown"),
                    "result": result,
                    "agents": agents[:5],
                    "key_stats": {
                        "rounds": self._extract_score(teams_data)
                    }
                }
                matches.append(match)

            if not matches:
                logger.warning("No matches parsed from GRID → fallback to demo")
                return self._get_demo_data(team_name, game, limit)

            logger.info(f"Successfully fetched {len(matches)} real matches from GRID")
            return matches

        except Exception as e:
            logger.error(f"Error processing GRID data: {e} → fallback to demo")
            return self._get_demo_data(team_name, game, limit)

    def _extract_score(self, teams_data):
        """Extract rounds score"""
        if len(teams_data) >= 2:
            score1 = teams_data[0].get("score", 0)
            score2 = teams_data[1].get("score", 0)
            return f"{score1}-{score2}"
        return "13-0"

    def _get_demo_data(self, team_name: str, game: str = "valorant", limit: int = 5):
        """
        Enhanced demo data with variety for better LLM analysis

        Hackathon-optimized: always works, always looks good
        """
        logger.info(f"Generating {limit} mock matches for {team_name} ({game})")

        # Varied mock matches for realistic demo
        mock_matches = [
            {
                "date": "2026-01-08",
                "map": "Bind",
                "result": "win",
                "agents": ["Jett", "Omen", "Sova", "KAY/O", "Cypher"],
                "key_stats": {"rounds": "13-8"}
            },
            {
                "date": "2026-01-07",
                "map": "Ascent",
                "result": "win",
                "agents": ["Jett", "Omen", "Sova", "Breach", "Killjoy"],
                "key_stats": {"rounds": "13-10"}
            },
            {
                "date": "2026-01-06",
                "map": "Haven",
                "result": "loss",
                "agents": ["Jett", "Astra", "Sova", "Skye", "Cypher"],
                "key_stats": {"rounds": "11-13"}
            },
            {
                "date": "2026-01-05",
                "map": "Icebox",
                "result": "win",
                "agents": ["Jett", "Viper", "Sova", "Sage", "Chamber"],
                "key_stats": {"rounds": "13-7"}
            },
            {
                "date": "2026-01-04",
                "map": "Split",
                "result": "loss",
                "agents": ["Raze", "Omen", "Breach", "Sage", "Cypher"],
                "key_stats": {"rounds": "10-13"}
            },
            {
                "date": "2026-01-03",
                "map": "Bind",
                "result": "win",
                "agents": ["Jett", "Omen", "Fade", "KAY/O", "Killjoy"],
                "key_stats": {"rounds": "13-9"}
            },
            {
                "date": "2026-01-02",
                "map": "Ascent",
                "result": "win",
                "agents": ["Jett", "Astra", "Sova", "Skye", "Killjoy"],
                "key_stats": {"rounds": "13-6"}
            },
            {
                "date": "2026-01-01",
                "map": "Haven",
                "result": "loss",
                "agents": ["Jett", "Omen", "Breach", "Skye", "Chamber"],
                "key_stats": {"rounds": "12-14"}
            },
        ]

        # LoL mock data
        if game == "lol":
            for m in mock_matches:
                m["agents"] = ["KSante", "Vi", "Azir", "Kalista", "Rell"]

        return mock_matches[:limit]

    def get_config(self) -> dict:
        """Get current configuration for debugging"""
        return {
            "provider": "grid",
            "demo_mode": self.demo_mode,
            "timeout": self.timeout,
            "has_api_key": bool(self.api_key)
        }
