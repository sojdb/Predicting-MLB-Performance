from urllib.request import Request,urlopen
import requests
from bs4 import BeautifulSoup as soup
from time import sleep
from io import StringIO
import re
from unidecode import unidecode
import os
import pybaseball as bb
import pandas as pd
import numpy as np
import datetime as dt

import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier

import xgboost as xgb

import random
import warnings
warnings.filterwarnings("ignore", message="The input looks more like a filename than markup. You may want to open this file and pass the filehandle into Beautiful Soup.")
warnings.filterwarnings("ignore", message="invalid value encountered in cast")
pd.set_option('display.max_rows',20)
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None

"""
Script acts as a module to create our baseball scraping class. After initializing the object, one can utilize it to scrape a season schedule,
team's 40 man roster for a season, the current day lineups, the betting odds, etc.
"""

class BaseballScraper:

    def __init__(self):
        self.abr_convert = self.teams_abbr = {"AZ" : 'ARI',
                                            "ARI": 'ARI',
                                            "LAD" : 'LAD',
                                            "SD": 'SDP',
                                            "SF": 'SFG',
                                            "COL": 'COL',
                                            "HOU" : 'HOU',
                                            "TEX" : 'TEX',
                                            "SEA": 'SEA',
                                            "LAA": 'LAA',
                                            "OAK": 'OAK',
                                            "MIL": 'MIL',
                                            "CHC": 'CHC',
                                            'Chi. Cubs': 'CHC',
                                            "CIN": 'CIN',
                                            "PIT": 'PIT',
                                            "STL": 'STL',
                                            "MIN": 'MIN',
                                            "DET": 'DET',
                                            'Detroit': 'DET',
                                            "CLE": 'CLE',
                                            "CWS": 'CHW',
                                            'CHW':'CHW',
                                            'Chi. White Sox' : 'CHW',
                                            "KC": 'KCR',
                                            "ATL": 'ATL',
                                            "PHI": 'PHI',
                                            "MIA": 'MIA',
                                            'Miami': 'MIA',
                                            "NYM": 'NYM',
                                            "WSH": 'WSN',
                                            'Washington':'WSN',
                                            'WAS':'WSN',
                                            "BAL": 'BAL',
                                            "TB": 'TBR',
                                            "TOR": 'TOR',
                                            "NYY": 'NYY',
                                            "BOS": 'BOS',
                                            'Boston':'BOS'}

    def getSoup(self, url):
        req = Request(url=url,headers={'User-Agent': 'Mozilla/6.0'})
        uClient = urlopen(req)
        #grabs everything from page
        html = uClient.read()
        #close connection
        uClient.close()
        #does HTML parsing
        parse = soup(html, "html.parser")

        return parse
    
    ### Lineups ###

    def getSchedule(self, season: int):
        ### ONLY CARE ABOUT DATES RN, WILL NEED TO BE CHANGED IF WANT FULL SCHEDULE ###
        text = self.getSoup("https://www.baseball-reference.com/leagues/majors/{0}-schedule.shtml".format(season))
        games = text.find('div', {'class':'section_content'})
        dates = games.find_all('h3')
        dates = [pd.to_datetime(','.join(date.text.split(',')[1:]).strip()).strftime('%Y-%m-%d') for date in dates]
        matchups = games.find_all('p', {'class':'game'})
        home_team, away_team = [], []
        for match in matchups:
            away_team.append(match.text.split('\n')[1].strip())
            home_team.append(match.text.split('\n')[4].strip())
        return dates
    
    def get40man(self, teams: list, seasons: list):
        all_roster = pd.DataFrame()
        for season in seasons:
            for team in teams:
                print(season, team)
                text = self.getSoup("https://www.baseball-reference.com/teams/{0}/{1}-roster.shtml#all_the40man".format(team, season))
                sleep(2)
                roster = StringIO(unidecode(str(text.find('table', {'id':'appearances'}))))
                bb_ref_id = [str(x).split('data-append-csv="')[1][0:9].replace('"', '') for x in text.findAll('th', {'class':'left'}) if 'data-append-csv="' in str(x)]
                bb_ref_id = [x for x in bb_ref_id if x != 'player.fc']
                df = pd.read_html(roster)[0]

                df.insert(3, 'Team', team)
                df.insert(3, 'Season', season)
                df = df[:-1]
                df.insert(3, 'bbref_id', bb_ref_id[0:len(df)])

                df = df[['Name', 'bbref_id', 'Season', 'Team']]

                all_roster = pd.concat([all_roster, df])
        
        return all_roster
    
    def get40ManAllTeams(self, years: list):

        return self.get_40man(set(self.teams_abbr.values()), years)
        
    def getCurrentDayLineup(self, date: str):

        text = self.getSoup(f'https://www.mlb.com/starting-lineups/{date}')
        away_team = text.findAll('div', {'class':"starting-lineups__team-logo-image starting-lineups__team-logo-image--away"})
        away_team = [str(x).split('data-tri-code="')[1][0:3].replace('"','') for x in away_team]
        away_team = [self.abr_convert.get(x) for x in away_team]

        home_team = text.findAll('div', {'class':"starting-lineups__team-logo-image starting-lineups__team-logo-image--home"})
        home_team = [str(x).split('data-tri-code="')[1][0:3].replace('"', '') for x in home_team]
        home_team = [self.abr_convert.get(x) for x in home_team]

        away_pitchers = text.findAll('div', {'class':'starting-lineups__pitcher-name'})
        away_pitchers = [re.search('">(.*)</a>', unidecode(str(x))).group(1) for x in away_pitchers if 'TBD' not in str(x)]

        home_pitchers = away_pitchers[1::2]
        del away_pitchers[1::2]

        away_hitters = text.findAll('ol', {'class':'starting-lineups__team starting-lineups__team--away'})
        del away_hitters[1::2]
        away_hitters = [re.findall('target="">(.*)</a>', unidecode(str(x))) for x in away_hitters if 'TBD' not in str(x)]

        home_hitters = text.findAll('ol', {'class':'starting-lineups__team starting-lineups__team--home'})
        del home_hitters[1::2]
        home_hitters = [re.findall('target="">(.*)</a>', unidecode(str(x))) for x in home_hitters if 'TBD' not in str(x)]

        return away_hitters, home_hitters, away_pitchers, home_pitchers, away_team, home_team

    ### Odds Data ###
    def _convertOdds(self, x: str):
        if len(x) == 1:
            return np.nan
        elif '+' in x:
            return round((int(x[1:])/100) + 1, 2)
        elif '-' in x:
            return round((100/int(x[1:])) + 1, 2)

    def getOddsForDate(self, dates: list):
        odds_table = pd.DataFrame()
        for date in dates:
            print(date)
            away_teams, home_teams = [], []
            text = self.getSoup("https://www.sportsbookreview.com/betting-odds/mlb-baseball/?date={0}".format(date))
            teams = text.find_all('div', {'class':"d-flex w-100 GameRows_participantContainer__6Rpfq"})
            
            for i, item in enumerate(teams):
                team = item.find('a', {'class':"d-flex align-items-center overflow-hidden fs-9 GameRows_gradientContainer__ZajIf"}).text.split('-')[0]
                if i % 2 == 0:
                    away_teams.append(team.replace(u'\xa0', u''))
                else:
                    home_teams.append(team.replace(u'\xa0', u''))
            
            away_odds, home_odds = [], []
            games = text.find_all('div', {'class':"d-flex flex-column align-items-end justify-content-center"})
            for team in games:
                away_odds.append(team.find_all('span', {'class':'fs-9 undefined'})[0].text)
                home_odds.append(team.find_all('span', {'class':'fs-9 undefined'})[1].text)
            
            temp = pd.DataFrame({'Date': date, 'Home Team':home_teams, 'Home W Odds':home_odds, 'Away Team':away_teams, 'Away W Odds':away_odds})
            temp['Home W Odds'], temp['Away W Odds'] = temp['Home W Odds'].apply(self._convertOdds), temp['Away W Odds'].apply(self._convertOdds)
            odds_table = pd.concat([odds_table, temp])
        return odds_table
    
    def getOddsForYears(self, years: list):
        all_odds = pd.DataFrame()
        for season in years:
            season_odds = self.getHistoricalOdds(self.getSchedule(season))
            all_odds = pd.concat([all_odds, season_odds])
        
        return all_odds
    
    ### Bullpen Ranking ###
    def _get_csv_data_with_selenium(self, season, game_date):
        """
        Uses Selenium to load the FanGraphs CSV URL for relief pitchers for the specified season.
        The URL provided is:
        https://www.fangraphs.com/leaders/major-league?pos=RP&lg=all&qual=0&type=8&
        season={season}&month=0&season1={season}&ind=0&csv=1&stats=rel&team=0%2Cts
        Returns the CSV content as a string.
        """
        # Construct the URL using the specified season.
        url = (
            # f"https://www.fangraphs.com/leaders/major-league?"
            # f"pos=RP&lg=all&qual=0&type=8&season={season}&month=0&season1={season}"
            # f"&ind=0&csv=1&stats=rel&team=0%2Cts&startdate={dt.date(season,1,1).strftime('%Y-%m-%d')}&enddate={game_date.strftime('%Y-%m-%d')}"
            f"https://www.fangraphs.com/leaders/major-league?pos=RP&lg=all&qual=0&type=8&month=1000&ind=0&csv=1&stats=rel&team=0%2Cts&startdate={dt.date(season,1,1).strftime('%Y-%m-%d')}&enddate={game_date.strftime('%Y-%m-%d')}&season={season}&season1={season}"
        )
        
        # Set up Selenium with headless Chrome.
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        driver = webdriver.Chrome(options=chrome_options)
        
        try:
            driver.get(url)
            # Give the page time to load the CSV text.
            time.sleep(3)
            
            # Many browsers render CSV data inside a <pre> tag.
            try:
                csv_text = driver.find_element(By.CLASS_NAME, "table-scroll").text
            except Exception:
                # Fallback: grab the entire body text.
                csv_text = driver.find_element(By.TAG_NAME, "body").text

            return csv_text
        except Exception as e:
            print("Error retrieving CSV data:", e)
            return None
        finally:
            driver.quit()

    def _load_data_from_csv(self, csv_text):
        """
        Reads the CSV data from a string into a pandas DataFrame.
        """
        try:
            csv_text = csv_text.replace('xERA', '')\
                                .replace('\n',',',18)\
                                .replace(',,',',')\
                                .replace('vFA (pi)','vFA(pi)')\
                                .replace(' ',',')
            df = pd.read_csv(StringIO(csv_text))
            return df
        except Exception as e:
            print("Error reading CSV data:", e)
            return None
        
    def _normalize_series(self, series, invert=False):
        """
        Normalizes a pandas Series to a 0-1 range.
        If invert=True, then lower values (which are better) receive higher normalized scores.
        """
        # Avoid division by zero in case the range is 0:
        range_val = series.max() - series.min()
        if range_val == 0:
            return pd.Series([1.0] * len(series), index=series.index)
        if invert:
            return (series.max() - series) / range_val
        else:
            return (series - series.min()) / range_val
        
    def rankBullpens(self, season, game_date):    
        print(f"Fetching CSV data for season {season} from FanGraphs...")
        csv_text = self._get_csv_data_with_selenium(season, game_date)
        if not csv_text:
            print("Failed to retrieve CSV data.")
            return
        
        print("Loading CSV data into a DataFrame...")
        df = self._load_data_from_csv(csv_text)
        if df is None or df.empty:
            print("No data loaded from CSV.")
            return

        # #Only keep needed columns
        required_columns = ['Team', 'ERA', 'WAR', 'K/9', 'SV']
        df = df[required_columns]
        for col in required_columns:
            if col not in df.columns:
                print(f"Error: Column '{col}' not found in the CSV. Please check the data source.")
                return
        df = df.dropna(subset=required_columns)
        # --- Step 4. Normalize each metric ---
        # For ERA and WHIP: lower is better so we invert the normalization.
        df['ERA_norm'] = self._normalize_series(df['ERA'], invert=True)
        # For K/9 and SV (Saves): higher is better so use direct normalization.
        df['K9_norm'] = self._normalize_series(df['K/9'], invert=False)
        df['SV_norm'] = self._normalize_series(df['SV'], invert=False)
        df['WAR_norm'] = self._normalize_series(df['WAR'], invert=False)

        # --- Step 5. Compute the composite bullpen rating ---
        # Here, we simply take the average of the normalized scores.
        df['Bullpen_Rating'] = round(df[['ERA_norm', 'WAR_norm', 'K9_norm', 'SV_norm']].mean(axis=1),4)
        df.insert(0, 'Season', season)
        df.insert(1, 'End Date', game_date)
        
        # Sort teams by the composite rating (highest rating first)
        df_sorted = df.sort_values(by='Bullpen_Rating', ascending=False)

        return df_sorted

if __name__ == '__main__':
    data_path = r"C:\Users\joshm\OneDrive\Documents\Side stuff\Baseball\Data"
    # season = 2023

    dates = pd.read_csv(os.path.join(data_path, 'team_gamelogs.csv'),usecols=['Date'])
    dates = list(set(pd.to_datetime(dates['Date']).to_list()))
    already_have_dates = pd.read_csv(os.path.join(data_path, 'bullpens.csv'),usecols=['End Date'])
    already_have_dates = list(set(pd.to_datetime(already_have_dates['End Date']).to_list()))
    dates = list(set(dates).difference(set(already_have_dates)))
    data_pull = BaseballScraper()
    # game_date = dt.date(season, 12, 1)
    df = pd.DataFrame()
    for game_date in dates:
        temp = data_pull.rankBullpens(game_date.year, game_date)
        df = pd.concat([df, temp])
    if 'bullpens.csv' in os.listdir(data_path):
        old_df = pd.read_csv(os.path.join(data_path, 'bullpens.csv'))
        df = pd.concat([old_df, df]).drop_duplicates(subset=['Team', 'Season','End Date'], keep='last')
    df['End Date'] = pd.to_datetime(df['End Date'])
    df.to_csv(os.path.join(data_path, 'bullpens.csv'), index=False)
    #edit to remove, getting to github