import pandas as pd
import argparse
import os


def parse_args():
    """
    Parse the arguments
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--tagged_file', type=str,
                        help='Path to the geo-tagged csv files')
    parser.add_argument('--column', type=str,
                        help='Column for which distribution has to be calculated.')
    parser.add_argument('--noun', type=str,
                        help='Noun for which you want distribution for')
    parser.add_argument('--svm', action='store_true',
                    help='enter False if you want distributions for the whole dataset, True if you want for positively classified images')
    parser.add_argument('--copy', action='store_true',
                        help='Want to copy values from one column to another column?')
    parser.add_argument('--source', type=str, default=None,
                        help='Source column to copy from')
    args = parser.parse_args()
    return args


def get_distribution(df, col, noun):
    """
    Get the country-wise distribution as per mixtral's outputs.
    """
    # if os.path.exists(f'data/distribution_{noun}_{col}.csv'):
    #     distribution = pd.read_csv(f'data/distribution_{noun}_{col}.csv')
    # else:
    distribution = df.groupby([col]).size().reset_index(name='counts')
    distribution.to_csv(f'data/distribution_{noun}_{col}.csv', index=False)
    return distribution


def preprocess_column(text):
    if type(text) == float:
        return 'no'
    text = text.lower()
    if text == 'czech republic (czechia)':
        return text
    text = text.replace('(', '')
    text = text.replace(')', '')
    text = text.replace('\"', '')
    text = text.replace('\'', '')
    text = text.split(':')[-1]
    text = text.strip()
    text = text.replace('_', ' ')
    text = text.split('/')[0]
    text = text.replace('-', ' ')
    text = text.replace('\&', 'and')
    if len(text) == 0:
        text = 'no'
    if text in ['us', 'usa', 'the united states', 'the united states of america',
                'u.s.', 'u.s.a', 'u.s.a.', 'united states of america', 'united states.']:
        text = 'united states'
    elif text in ['britain', 'great britain', 'the united kingdom', 'uk', 'england', 'wales', 'northern ireland', 'scotland',
                    'u.k.', 'u.k']:
        text = 'united kingdom'
    elif text == 'state of palestine':
        text = 'palestine'
    elif text == "democratic people's republic of korea":
        text = 'north korea'
    elif text == 'republic of korea':
        text = 'south korea'
    elif text == 'holland':
        text = 'netherlands'
    elif text == 'burma':
        text = 'myanmar'
    elif text == 'russian federation':
        text = 'russia'
    elif text in ['czechia', 'czech republic', 'czech']:
        text = 'czech republic (czechia)'
    elif text == 'swaziland':
        text = 'eswatini'
    elif text in ['uae', 'u.a.e', 'u.a.e.', 'abu dhabi']:
        text = 'united arab emirates'
    elif text in ['east timor', 'timor leste']:
        text = 'timor-leste'
    elif text in ['congo', 'republic of congo']:
        text = 'republic of the congo'
    
    elif text == 'lao peoples democratic republic':
        text = 'laos'
    elif text in ['gaza', 'gaza city', 'gaza strip']:
        text = 'egypt'
    elif text in ['turkiet', 'türkiye']:
        text = 'turkey'
    elif text == 'méxico':
        text = 'mexico'
    if len(text.split()) > 5 or ',' in text:
        text = 'NULL'
    return text


df_countries = pd.read_csv('data/gdp_population.csv')
print(df_countries.columns)
all_countries = list(df_countries['Country'])
all_countries = [x.lower() for x in all_countries]
all_countries = [preprocess_column(x) for x in all_countries] + ['no']

def check_country(text):
    if text not in all_countries:
        if 'cz' in text:
            print(text)
        return 'NULL'
    return text
    

def find_distribution(args=None):
    df = pd.read_csv(args.tagged_file, low_memory=False)
    df.reset_index(drop=True, inplace=True)
    
    if args.svm:
        print(set(list(df['svm_1'])))
        df = df[df['svm_1']==1]
        
    if args.source:
        df[args.source] = df[args.source].str.lower()
    df[args.column] = df[args.column].str.lower()
    df[args.column] = df[args.column].apply(preprocess_column)
    df[args.column] = df[args.column].apply(check_country)
    df = df[df[args.column]!='NULL']
    if args.copy:
        for idx, row in df.iterrows():
            if len(row[args.source].split('_')) <= 1:
                df.at[idx, args.column] = df.at[idx, args.source]
    df = get_distribution(df, args.column, args.noun)
    return df
    #df.to_csv(f'data/distribution_{args.noun}.csv')

if __name__ == '__main__':
    args = parse_args()
    find_distribution(args)