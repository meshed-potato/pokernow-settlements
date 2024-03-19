# PokerNow Settlement

Finds an optimal settlement from a list of cashouts from pokernow. Player name to payment info should be provided in `payment_info.csv`. payment_info.csv should have 'PN/ClubGG Alias' (nickname) and 'Venmo / other' (payment) columns.

This tool identifies the most efficient settlement strategy for a series of cashouts recorded in PokerNow, it is used internally in a small poker group.

The script outputs will be available in the ./output/ directory. 

### Steps:
1. If applicable, download all ledgers from pokernow based on game ids automatically
2. Read all ledgers, generate player cashout summary based on payment info
3. Use or-tools or networkx library to compute the best settlement among players (with SimpleMinCostFlow solver)
4. Convert the settlement to html format

## Requirements after downloading this directory:

1. Install necessary packages by running `pip install -r requirements.txt`
2. Ensure `./payment_info.csv` is up to date

## Example Usage:

### Auto mode:

Run the script with game IDs to automatically download ledger files:

```bash
python pokernow_settlement.py --game_ids pglLK9zwhOEqRCe43t_VDbAkO pglOqySSgqfH4edFGUvhmUKYt
```

Note: 
1. Use <b>Space</b> between different game ids
2. A game id is the suffix in the pokernow link.
    For example, the game id for https://www.pokernow.club/games/ABCxyz123 is ABCxyz123

### Manual mode:
Manual mode is useful when recaptcha is required for ledge downloading.

Download ledger file, and run the following script:

```bash
python pokernow_settlement.py --ledgers_path_manual ./data/[dir_with_ledge]/
```
