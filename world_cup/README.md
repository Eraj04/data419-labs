# FIFA World Cup Data Project

This project contains the reusable data acquisition and loading code for my DATA 419 semester project.

## Data Source

The original data comes from the Football - FIFA World Cup, 1930-2022 dataset on Kaggle.

Dataset:
piterfm/fifa-football-world-cup

## Generated Tables

### matches

One row represents one FIFA World Cup match from 1930 through 2022. It includes teams, scores, attendance, tournament round, venue, host country, date, managers, captains, goals, cards, substitutions, and expected goals where available.

### tournaments

One row represents one FIFA World Cup tournament. It includes information such as the year, host country, number of teams, champion, runner-up, attendance, and number of matches.

### rankings

One row represents one national team in the FIFA rankings from October 2022. It includes the team, association, rank, ranking points, previous rank, and previous ranking points.

## Installation

From the data419-labs directory:

    pip install -e ./world_cup

## Recreating the Data

Run:

    python -m world_cup.pull_all

This downloads the original CSV files, performs basic cleaning and preparation, and writes the analysis-ready Parquet files into the world_cup/data directory.

## Loading the Data

In Python:

    import world_cup

    tables = world_cup.load()

The returned dictionary contains:

    tables["matches"]
    tables["tournaments"]
    tables["rankings"]

## Requirements

Python 3.11 or newer is required.

The project uses pandas, pyarrow, and kagglehub.

## Known Limitations

Expected goals data is only available for more recent World Cup matches, so older matches contain missing values for those variables. The FIFA ranking table is from October 2022 rather than containing rankings for every historical World Cup.
