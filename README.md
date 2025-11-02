# League of Legends Matchup Stats — Personal Performance Analyzer

This application retrieves a player's League of Legends match history from the Riot Games API and analyzes personal performance in specific champion matchups. The program stores match data in a **SQLite relational database** and provides an interactive **PyQt6 GUI** for searching best counter picks against any enemy champion.

**Core Features**
- Syncs newest match history while respecting Riot API rate limits
- Stores matches and detailed participant stats in SQLite
- Shows **Top 5 counter champions** filtered by lane (Top/Jungle/Mid/ADC/Support)
- Displays champion icons via Data Dragon
- Supports searching **any Riot ID**, not just the author’s
- Queue filtering: Only Summoner’s Rift modes (Ranked/Normal/Swiftplay)

**Demo Video:** https://youtu.be/MB_yg7G2g1k

---

## Relational Database

The SQLite backend contains two primary tables:

### `matches`
| Column | Type | Description |
|--------|------|-------------|
| match_id | TEXT (PK) | Unique ID of match |
| platform | TEXT | Platform routing (e.g., NA1) |
| game_start_ts | INTEGER | Match timestamp |
| queue_id | INTEGER | Summoner’s Rift queue type |
| game_duration | INTEGER | Length in seconds |
| patch | TEXT | Game version |

### `participants`
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER (PK) | Unique participant entry |
| match_id | TEXT (FK) | Links to match |
| puuid | TEXT | Player global identifier |
| summoner_name | TEXT | Summoner name at match time |
| champion_id | INTEGER | Champ ID |
| champion_name | TEXT | Champ name |
| team_id | INTEGER | Blue/Red side |
| is_me | INTEGER | 1 if user, otherwise 0 |
| position | TEXT | Lane/role played |
| win | INTEGER | 1 = win / 0 = loss |
| kills | INTEGER | Kills |
| deaths | INTEGER | Deaths |
| assists | INTEGER | Assists |
| cs | INTEGER | Total minions killed |
| gold | INTEGER | Gold earned |
| vision_score | INTEGER | Vision stat |
| damage_dealt | INTEGER | Damage to champions |

**SQL demonstrated**
- INSERT new matches and player stats
- SELECT with aggregates (`COUNT`, `SUM`, `AVG`)
- JOIN across players in matches
- Optional UPDATE and DELETE supported

---

##Development Environment

**Language:** Python 3.11  
**Frameworks & Libraries**
- PyQt6 — UI rendering
- requests — API calls
- sqlite3 — Database operations
- threading — Async syncing + UI responsiveness

**External Services**
- Riot Games Match V5 & Account V1 APIs
- Data Dragon for champion icons and patch versions

Developed in **Visual Studio Code** on Windows.

---

##How to Run

### Install dependencies
install the following: [{PyQt6}{requests}]
aqcuire riot dev key
- go to https://developer.riotgames.com/
- sign into a Riot Games Account
- go to your API key page (mine opened to this once signed in)
- copy API key and in your enviroment run either
        - Windows
    - $Env:RIOT_API_KEY = "RGAPI-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
        - Mac
    - export RIOT_API_KEY="RGAPI-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
- run the program

##Things to add

- items to build in the match
- playstyle recommendations