'''
Finds an optimal settlement from a list of cashouts.
Example Usage:
    Auto mode:
        python pokernow_settlement.py --game_ids pglLK9zwhOEqRCe43t_VDbAkO pglOqySSgqfH4edFGUvhmUKYt
    Manual mode:
        python pokernow_settlement.py --ledgers_path_manual ./data/[dir_with_ledge]/
        python pokernow_settlement.py --ledgers_path_manual ./data/tmp_ledgers/

'''
import argparse
import collections
import csv
import datetime
import functools
import math
import multiprocessing
import os
import pandas as pd
import random
import requests
import sys
import time
from glob import glob

TMP_LEDGERS_DIR = "./data/tmp_ledgers/"
# This input file should have POKERNOW_NICKNAME_COL and PAYMENT_COL columns.
INPUT_PAYMENT_INFO_FILE = "./payment_info.csv"

OUTPUT_DIR = "./output/"
OUTPUT_SETTLEMENT_FILE = f"{OUTPUT_DIR}settlement.html"

POKERNOW_NICKNAME_COL = 'PN/ClubGG Alias'
PAYMENT_COL = 'Venmo / other'

random.seed(123)


def combine_and_save_ledgers(source_dir, output_dir):
    """
    Combines all CSV files in the source directory into a single DataFrame and saves it
    in the output directory with a YYYYMMDD suffix.

    :param source_dir: Directory where the individual ledger CSV files are stored.
    :param output_dir: Directory where the combined CSV file will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine all downloaded ledgers
    all_ledgers = pd.DataFrame()
    for csv_file in glob(os.path.join(source_dir, '*.csv')):
        df = pd.read_csv(csv_file)
        all_ledgers = pd.concat([all_ledgers, df], ignore_index=True)

    # Get current date in YYYYMMDD format for the filename
    today_str = datetime.datetime.now().strftime('%Y%m%d')
    combined_file_path = os.path.join(output_dir, f'downloaded_ledgers_{today_str}.csv')
    
    try:
        all_ledgers.to_csv(combined_file_path, index=False)
        print(f"\nCombined ledger saved to {combined_file_path}")
    except Exception as e:
        print(f"\nError saving combined ledger: {e}.")
        return False

    return True

def download_ledgers(game_ids, ledger_dir):
    """
    Downloads ledger files for a list of game IDs and saves them to the specified directory.
    Returns False if the game_ids list is empty or if any of the HTTP GET requests fail, and True otherwise.


    :param game_ids: List of game IDs for which to download ledgers.
    :param ledger_dir: Directory to save the ledger files.
    """
    if not game_ids:
        return False
    # Ensure the directory exists
    os.makedirs(ledger_dir, exist_ok=True)
    # Clear the ledger directory before downloading
    csv_files = glob(os.path.join(ledger_dir, '*.csv'))
    for csv_file in csv_files:
        os.remove(csv_file)
    print(f"\nSuccessfully Cleaned Up dir {ledger_dir}\n")

    # Base URL pattern for downloading ledgers
    base_url = "https://www.pokernow.club/games/{}/ledger_{}.csv"

    for game_id in game_ids:
        ledger_url = base_url.format(game_id, game_id)
        print(f"processing url: {ledger_url}")

        file_path = os.path.join(ledger_dir, f"ledger_{game_id}.csv")

        # Download the ledger
        try:
            response = requests.get(ledger_url)
            response.raise_for_status()
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded {file_path}")
        except Exception as e:
            # Handle errors in downloading or writing the file
            print(f"Error downloading ledger for game ID {game_id}: {e}. Manual downloading is suggested.")
            return False
        # Sleep a few secs for each download
        time.sleep(3)

    # Combine and save the downloaded ledgers
    combine_and_save_ledgers(ledger_dir, OUTPUT_DIR)

    return True

def read_cashouts(path, read_all_csv=True):
    '''Read (player, cashout) pairs from an input CSV file or all CSV files in a directory,
    using Venmo information from a separate mapping file.

    Args:
        path: Path to a CSV file or a directory containing CSV files.
        read_all_csv: Flag to read all CSV files in the directory if True; read a single file if False.
    
    Required columns for ledger file:
        'player_nickname' -- player nickname
        'net' -- cashout

    Outputs:
        a list of tuples, with each tuple consisting of a player's Venmo account identifier and the 
        aggregated net cashout amount for that player.
        Example: [("JohnDoeVenmo", 250), ("JaneSmithVenmo", -200), ("Unknown Player", -50)]
    '''
    players = collections.defaultdict(int)
    venmo_mapping = {}

    # Load Venmo information mapping
    with open(INPUT_PAYMENT_INFO_FILE, newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        headers = next(reader)  # Assumes first row is header
        for row in reader:
            mapped_row = dict(zip(headers, row))
            venmo_mapping[mapped_row[POKERNOW_NICKNAME_COL]] = mapped_row[PAYMENT_COL]

    # Convert mapping key to lower cases.
    venmo_mapping_lower = {k.lower(): v for k, v in venmo_mapping.items()}

    # Function to process each file
    def process_file(filename):
        with open(filename, newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            headers = next(reader)
            for row in reader:
                mapped_row = dict(zip(headers, row))
                try:
                    cashout = int(mapped_row['net'].replace('.', ''))
                    # Lookup PN/ClubGG Alias to get the correct Venmo information. Using lower case to lookup.
                    player_nickname = mapped_row['player_nickname'].lower()
                    if player_nickname == "" or player_nickname not in venmo_mapping_lower:
                        print(f"\nError: Player with nickname '{player_nickname}' not found in Payment mapping or nickname is empty.\n")
                        sys.exit(1)  # Terminate program
                    else:
                        player_id = venmo_mapping_lower[player_nickname]
                except ValueError:
                    continue
                players[player_id] += cashout

    if read_all_csv:
        # Ensure path ends with slash to correctly build glob pattern
        directory = os.path.join(path, '')  # Adding a trailing slash if it's not present
        for filename in glob(directory + '*.csv'):
            process_file(filename)
    else:
        process_file(path)

    return list(players.items())


def edge_weight():
    '''
        Generate a weight for a payment graph edge.
    '''
    return random.randint(-1_000_000_000, 1_000_000_000)

def find_settlement_or(cashouts, _):
    '''Finds a feasible settlement in a payment graph using OR-tools library.'''
    from ortools.graph.python import min_cost_flow
    smcf = min_cost_flow.SimpleMinCostFlow()

    for player_from, cashout1 in enumerate(cashouts):
        if cashout1 <= 0:
            continue
        for player_to, cashout2 in enumerate(cashouts):
            if cashout2 >= 0:
                continue
            cap = min(cashout1, -cashout2)
            smcf.add_arcs_with_capacity_and_unit_cost(player_from, player_to, cap, edge_weight())

    for player_id, cashout in enumerate(cashouts):
        smcf.set_nodes_supplies(player_id, cashout)

    status = smcf.solve()
    assert status == smcf.OPTIMAL, status

    settlements = collections.defaultdict(list)
    for i in range(smcf.num_arcs()):
        if smcf.flow(i) <= 0:
            continue
        settlements[smcf.tail(i)].append((smcf.head(i), smcf.flow(i)))

    return settlements


def find_settlement_nx(cashouts, _):
    '''Finds a feasible settlement in a payment graph using networkx library.'''
    import networkx as nx
    graph = nx.DiGraph()

    source, sink = len(cashouts), len(cashouts) + 1
    for player, cashout in enumerate(cashouts):
        if cashout < 0:
            graph.add_edge(source, player, capacity=-cashout)
        elif cashout > 0:
            graph.add_edge(player, sink, capacity=cashout)

    for player_from, cashout1 in enumerate(cashouts):
        if cashout1 >= 0:
            continue
        for player_to, cashout2 in enumerate(cashouts):
            if cashout2 <= 0:
                continue
            cap = min(-cashout1, cashout2)
            graph.add_edge(player_from, player_to, capacity=cap, weight=edge_weight())

    smcf = nx.max_flow_min_cost(graph, source, sink)

    settlements = collections.defaultdict(list)
    for player_from, cashout in enumerate(cashouts):
        if cashout >= 0:
            continue
        for player_to, payment in smcf[player_from].items():
            if payment <= 0:
                continue
            settlements[player_to].append((player_from, payment))

    return settlements


def find_best_settlement(settlement_finder, cashouts, num_trials):
    '''Finds a settlement with a minimum number of payments.'''
    best_cost, best_settlement = len(cashouts) * len(cashouts) + 1, None

    find_settlement = functools.partial(settlement_finder, cashouts)

    chunk = math.ceil(num_trials / multiprocessing.cpu_count())
    with multiprocessing.Pool() as pool:
        for settlement in pool.imap_unordered(find_settlement, range(num_trials), chunksize=chunk):
            cost = sum(len(al) for _, al in settlement.items())
            if cost < best_cost:
                best_cost, best_settlement = cost, settlement

    return best_settlement


def get_printable_dollar_amount(num):
    '''Gets a printable_dollar amount.'''
    quot, rem = divmod(num, 100)
    return f'${quot}.{rem:02}'


def href(name):
    '''Wrap a name in a hyperlink.'''
    return '<a href="https://venmo.com/u/' + name[1:] + '">' + name + '</a>'


def get_venmo_profile_link(username):
    '''Gets a hyperlink to a Venmo profile. Username starts with @'''
    return '<a href="https://venmo.com/u/' + username[1:] + '">' + username + '</a>'

def get_venmo_amount_only_link(username, amount, description=None):
    '''Gets a hyperlink to a Venmo request. Username starts with @'''
    quot, rem = divmod(amount, 100)
    note = ''
    if description:
        note = '&note=' + description
    return f'<a href="https://venmo.com/?txn=charge&audience=private&recipients={username[1:]}&amount={quot}.{rem:02}{note}">' + get_printable_dollar_amount(amount) + '<a>'

def get_venmo_amount_and_username_link(username, amount, description=None):
    '''Gets a hyperlink to a Venmo request. Username starts with @'''
    quot, rem = divmod(amount, 100)
    note = ''
    if description:
        note = '&note=' + description
    printable_amount = get_printable_dollar_amount(amount);
    return f'{printable_amount} <a href="https://venmo.com/?txn=charge&audience=private&recipients={username[1:]}&amount={quot}.{rem:02}{note}">{username}<a>'

def generate_settlement_html(settlement, players, description=None):
    '''Returns a settlement as a HTML string.'''
    result = '<pre>'
    for player_id in sorted(settlement, key=lambda player_id: players[player_id][0].lower()):
        name, cashout = players[player_id]
        result += f'\n{get_venmo_profile_link(name)} requests {get_printable_dollar_amount(cashout)} from:'
        for payer in sorted(settlement[player_id], key=lambda p: (-p[1], players[p[0]][0].lower())):
            amount = payer[1]
            payer_username = players[payer[0]][0]
            # 1 link with amount and username
            result += f'\n\t{get_venmo_amount_and_username_link(payer_username, amount, description)}'
        result += '\n'
    result += '</pre>'
    return result


def main():
    '''Prints an optimal settlement for a list of cashouts provided in a CSV file.'''
    parser = argparse.ArgumentParser(description="Read player cashouts from CSV file(s).")

    exclusive_group = parser.add_mutually_exclusive_group()
    # Add arguments to the mutually exclusive group and make them optional
    exclusive_group.add_argument('--game_ids', metavar='N', type=str, nargs='+',
                                 help='A list of game IDs for which to download ledgers. If provided, the ledgers are downloaded automatically.')
    exclusive_group.add_argument("--ledgers_path_manual", type=str, 
                                 help="Path to the CSV file or directory containing manually downloaded ledger CSV files.")

    # This argument remains outside the exclusive group as it does not conflict with the others
    parser.add_argument("--description", type=str, default=None, help="Description for the payment.")
    parser.add_argument("--num_trials", type=int, default=1_001, help="Number of trials for the settlement.")


    # Parse the arguments
    args = parser.parse_args()

    try:
        import ortools.graph
        settlement_finder = find_settlement_or
        print('Using or-tools library to find settlement.')
    except ModuleNotFoundError:
        settlement_finder = find_settlement_nx
        print('Using networkx library to find settlement.')

    # Step 1: Automatically download ledgers locally
    if args.game_ids:
        print(f"{len(args.game_ids)} Game ids are: {args.game_ids}")
        download_success = download_ledgers(args.game_ids, TMP_LEDGERS_DIR)
    else:
        download_success = False

    # Step 2: Read payouts and do settlement
    if not download_success and not args.ledgers_path_manual:
        return
    ledgers_path = TMP_LEDGERS_DIR if download_success else args.ledgers_path_manual
    mode = "Auto" if download_success else "Manual"
    print(f"\n{mode} Mode: Using {mode} downloaded ledgers to compute settlement: {ledgers_path}\n")

    player_cashouts = read_cashouts(path=ledgers_path, read_all_csv=True)
    cashouts = [pc[1] for pc in player_cashouts]

    # New validation check for sum in player_cashouts
    if sum(cashouts) != 0:
        print(f"Error: The total sum of cashouts is {sum(cashouts)}, but it should be 0.")
        return

    settlement = find_best_settlement(settlement_finder, cashouts, args.num_trials)

    settlement_html = generate_settlement_html(settlement, player_cashouts, args.description)
    with open(OUTPUT_SETTLEMENT_FILE, 'w') as file:
            file.write(settlement_html)
    print(f"\nSettlement Completed! Output is ready in {OUTPUT_SETTLEMENT_FILE}\n")

if __name__ == '__main__':
    main()

