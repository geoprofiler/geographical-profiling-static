import streamlit as st
import pandas as pd
import spacy
import os
from transformers import AutoTokenizer, CLIPModel, AutoProcessor, CLIPProcessor
from fuzzywuzzy import process
import torch
import requests
import torch.nn.functional as F
from transformers import CLIPProcessor
from PIL import Image
from io import BytesIO
import joblib
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import re
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool, ColumnDataSource
from collections import namedtuple
from find_distribution import find_distribution, preprocess_column

# Load the English language model
nlp = spacy.load("en_core_web_sm")

st.set_page_config(layout="wide")

st.title('Geographical Analysis')

width = 5000

countries = ['India', 'Japan', 'South Korea', 'Israel', 'Finland', 'United States', 
            'United Kingdom', 'Germany', 'France', 'Italy', 'Spain', 'Greece',
            'Czech Republic', 'Poland', 'Chile', 'Brazil', 'Mexico', 'Canada',
            'Australia', 'New Zealand', 'Portugal', 'Hungaria', 'South Africa', 'Slovenia',
            'Estonia', 'Latvia', 'Belgium', 'No GPE', 'Guinea', 'Nigeria', 'Philippines', 'Kenya', 'Thailand']

nouns = ['house', 'car', 'kitchen', 'flag', 'road', 'bedroom', 'beach', 'hotel', 'toilet', 'apartment']

@st.cache_data
def get_dataset(country, s, n):
    data_path = f'data/{n}_data.csv'
    df = pd.read_csv(data_path, low_memory=False)
    cap_dic = []
    captions = df['text'].tolist()
    ids = df['id'].tolist()
    urls = df['url'].tolist()
    column = 'mixtral'
    country = preprocess_column(country)
    # df[column] = df[column].fillna('no')
    # df[column] = df[column].str.lower()
    # df[column] = df[column].apply(preprocess_column)
    # mixtral = df[column].tolist()
    
    # Create a list to store images and captions
    images = []
    for i, c in enumerate(df[column].tolist()):
        c = preprocess_column(c)
        url = urls[i]
        caption = captions[i]
        local_path = f'data/image_folders/{n}_region_balanced_images/'+ids[i]+'.jpg'
        if country != 'No GPE' and c != country:
            continue  
        flag = False
        predicted_label = df.loc[i, 'svm_1']
        if predicted_label != 1:
            continue
        else:
            flag = True            
        if flag is True:
            try:
                # Load image
                image = Image.open(local_path)
                # Resize image to maintain aspect ratio
                aspect_ratio = 1
                new_width = 350
                new_height = int(new_width / aspect_ratio)
                image = image.resize((new_width, new_height))
                # Append image and caption to the list
                images.append((url, image, country))
            except Exception as e:
                print(f"Exception: {str(e)}")
    
    # Display images in a grid layout with multiple images per row
    cols = 5  # Number of images per row
    rows = len(images) // cols + (len(images) % cols > 0)  # Calculate number of rows needed
    for i in range(rows):
        row_images = images[i * cols: (i + 1) * cols]  # Get images for current row
        cols_container = st.columns(cols)  # Create columns for images
        col_index=0
        for col, (url, img, caption) in zip(cols_container, row_images):
            with col:
                st.image(img, caption=str.title(caption))
                # display_image_on_hover(url, i, col_index, '')
                # col_index += 1

# Create a sidebar menu
st.sidebar.header('Menu')

if 'section' not in st.session_state:
    st.session_state.section = 'Visualisations'

# Add a dropdown menu for different sections
section = st.sidebar.selectbox('Select Section', ['Visualisations', 'World-Wide Frequencies', 'Top-K countries'])
st.session_state.section = section

if 'section' not in st.session_state:
    st.session_state.section = ''

selected_noun = st.selectbox('Select noun:', nouns)

# Define the options buttons
if 'visualisations' not in st.session_state:
    st.session_state.visualisations = False

if st.session_state.visualisations or st.session_state.section == 'Visualisations':
    st.title('Visualisations')
    with st.form(key='visualisations_form'):
        st.session_state.visualisations = True
        selected_country = st.selectbox('Select country:', countries)
        if 'counter' not in st.session_state:
            st.session_state.counter = 0
        if st.form_submit_button('Let\'s see!'):
            get_dataset(selected_country, st.session_state.counter, selected_noun)
            st.session_state.counter += width

elif st.session_state.section != 'Visualisations':
    st.session_state.visualisations = False
    st.session_state.section = section

# if st.session_state.section == 'World-Wide Frequencies':
#     st.title('World-Wide Frequencies')
#     noun = selected_noun
#     data_path = f'/data2/abhipsa/datasets/laion_processed/laion_{noun}1M_1.csv'
#     column = 'mixtral_spacy'
#     if os.path.exists(f'data/distribution_{noun}_{column}.csv'):
#         df = pd.read_csv(f'data/distribution_{noun}_{column}.csv')
#     else:
#         df = pd.read_csv(data_path, low_memory=False)
#         parse_args = namedtuple('parse_args', ['tagged_file', 'column', 'noun', 'svm', 'source', 'copy'])
#         args = parse_args(tagged_file=data_path, column=column, noun=noun, svm=True, source=False, copy=False)
#         df = find_distribution(args)
#     world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
#     world['name'] = world['name'].str.lower()
#     world['name'] = world['name'].str.replace(r'\bw\.', 'west', regex=True)
#     world['name'] = world['name'].str.replace(r'\be\.', 'east', regex=True)
#     world['name'] = world['name'].str.replace(r'\bs\.', 'south', regex=True)
#     world['name'] = world['name'].str.replace(r'\bn\.', 'north', regex=True)
#     # file_path = 'data/world_modified.shp'
#     # world = gpd.read_file(file_path)
#     world['name'] = world['name'].apply(preprocess_column)
#     # Merge the world map dataframe with the country counts
#     world = world.merge(df.groupby('mixtral_spacy')['counts'].sum().reset_index(),
#                         left_on='name', right_on='mixtral_spacy', how='left')
#     # Fill missing values with 0
#     world['counts'] = world['counts'].fillna(0)
#     # Apply logarithmic transformation to the counts
#     world['count_log'] = np.log10(world['counts'] + 1)  # Adding 1 to avoid log(0)
#     # Plot the world map with country counts on logarithmic scale
#     levels = [0, 1, 2, 3, 4, 5]
#     labels = [f'$10^{level}$' for level in levels]
#     fig, ax = plt.subplots(1, 1, figsize=(10, 6))
#     ax.axis('off')
#     world.plot(column='count_log', ax=ax, legend=True,
#                legend_kwds={'orientation': "vertical"}, cmap='Blues')
#     # Customize the legend labels
#     colorbar = ax.get_figure().get_axes()[1]
#     colorbar.set_aspect(2)
#     colorbar.set_frame_on(False)
#     # Set the position of the ticks manually
#     num_ticks = len(levels)
#     colorbar.set_yticks(range(num_ticks))
#     colorbar.set_yticklabels(labels, fontsize=6)
#     plt.title(f'Country-wise distribution of {noun} (Log Scale)')
#     # plt.savefig(f'Plots/world_map_{noun}.png', bbox_inches='tight')
#     st.pyplot(fig)
#     plt.close()

# if 'topk' not in st.session_state:
#     st.session_state.topk = False

# if st.session_state.topk or st.session_state.section == 'Top-K countries':
#     st.title('Top-K countries')
#     with st.form(key='topk_form'):
#         st.session_state.topk = True
#         selected_k = st.selectbox('Select K:', [10, 100])
#         if st.form_submit_button('Generate graphs'):
#             noun = selected_noun
#             data_path = f'/data2/abhipsa/datasets/laion_processed/laion_{noun}1M_1.csv'
#             column = 'mixtral'
#             if os.path.exists(f'data/distribution_{noun}_{column}.csv'):
#                 df = pd.read_csv(f'data/distribution_{noun}_{column}.csv')
#             else:
#                 df = pd.read_csv(data_path, low_memory=False)
#                 total_data_len = df.shape[0]
#                 actual_noun_percentage = round(df[df['svm_1'] == 1].shape[0] / df.shape[0] * 100, 2)
#                 parse_args = namedtuple('parse_args', ['tagged_file', 'column', 'noun', 'svm', 'source', 'copy'])
#                 args = parse_args(tagged_file=data_path, column=column, noun=noun, svm=True, source=False, copy=False)
#                 df = find_distribution(args)
#             # Calculate percentage of occurrences of "No"
#             no_percentage = round((df[df[column] == 'no']['counts'].sum() / df['counts'].sum()) * 100, 2)
#             # st.write(f'Out of the {total_data_len} rows, {actual_noun_percentage}% of the data is {noun}s')
#             st.write(f'{no_percentage}% of the {noun} data have no country tags.')
#             # Exclude "No" from the DataFrame
#             df_without_no = df[df[column] != 'no']
#             df_without_no.reset_index(drop=True, inplace=True)
#             try:
#                 df_without_no = df_without_no.sort_values(by='counts', ascending=False)
#             except KeyError:
#                 print("Error: 'counts' column not found in the DataFrame.")
#                 exit()
#             # Calculate total number of countries (handle empty DataFrame)
#             if df_without_no.empty:
#                 print("Error: DataFrame is empty.")
#                 exit()
#             total_countries = len(df_without_no)

#             # Sort the DataFrame by 'counts' in descending order
#             df_sorted = df_without_no.sort_values(by='counts', ascending=False)

#             # Top k countries
#             top_k_countries = df_sorted.iloc[:selected_k]

#             # Filter to get countries with counts >= 10 for bottom selection
#             df_counts_ge_10 = df_sorted[df_sorted['counts'] >= 10]

#             # Bottom k countries (which have counts >= 10)
#             bottom_k_countries = df_counts_ge_10.iloc[-selected_k:]

#             # Ensure middle_k selection excludes top and bottom k countries
#             remaining_countries = df_sorted[~df_sorted.index.isin(top_k_countries.index) & ~df_sorted.index.isin(bottom_k_countries.index)]

#             # Randomly select middle k countries
#             middle_k_countries = remaining_countries.sample(n=selected_k, random_state=1)

#             middle_k_countries = middle_k_countries.sort_values(by='counts', ascending=False)
#             df_gdp = pd.read_csv('data/gdp_population.csv')
#             # Merge the dataframes and filter out rows with missing values in specified columns
#             df_merged = pd.merge(df_without_no, df_gdp, on=column, how='inner').dropna(subset=['counts', 'GDP(nominal, 2022)', 'Population(2022)', 'GDP per capita'])

#             # Filter merged dataframe for top, middle, and bottom countries
#             df_top_k = df_merged[df_merged[column].isin(top_k_countries[column])]
#             df_middle_k = df_merged[df_merged[column].isin(middle_k_countries[column])]
#             df_bottom_k = df_merged[df_merged[column].isin(bottom_k_countries[column])]

#             # Calculate correlations using the filtered DataFrame
#             correlation_gdp_top = round(df_top_k['counts'].corr(df_top_k['GDP(nominal, 2022)']), 2)
#             correlation_population_top = round(df_top_k['counts'].corr(df_top_k['Population(2022)']), 2)
#             correlation_gdp_capita_top = round(df_top_k['counts'].corr(df_top_k['GDP per capita']), 2)
#             correlation_gdp_middle = round(df_middle_k['counts'].corr(df_middle_k['GDP(nominal, 2022)']), 2)
#             correlation_population_middle = round(df_middle_k['counts'].corr(df_middle_k['Population(2022)']), 2)
#             correlation_gdp_capita_middle = round(df_middle_k['counts'].corr(df_middle_k['GDP per capita']), 2)
#             correlation_gdp_bottom = round(df_bottom_k['counts'].corr(df_bottom_k['GDP(nominal, 2022)']), 2)
#             correlation_population_bottom = round(df_bottom_k['counts'].corr(df_bottom_k['Population(2022)']), 2)
#             correlation_gdp_capita_bottom = round(df_bottom_k['counts'].corr(df_bottom_k['GDP per capita']), 2)
#             # Display the correlations
#             st.write(f"GDP X top {selected_k}: {correlation_gdp_top}")
#             st.write(f"GDP X middle {selected_k}: {correlation_gdp_middle}")
#             st.write(f"GDP X bottom {selected_k}: {correlation_gdp_bottom}")
#             st.write(f"Population X top {selected_k}: {correlation_population_top}")
#             st.write(f"Population X middle {selected_k}: {correlation_population_middle}")
#             st.write(f"Population X bottom {selected_k}: {correlation_population_bottom}")
#             st.write(f"GDP per capita X top {selected_k}: {correlation_gdp_capita_top}")
#             st.write(f"GDP per capita X middle {selected_k}: {correlation_gdp_capita_middle}")
#             st.write(f"GDP per capita X bottom {selected_k}: {correlation_gdp_capita_bottom}")
#             # Plot bar graph of top K countries and "No" separately
#             fig, ax = plt.subplots(figsize=(20, 10))
#             top_k_countries.plot(kind='bar', x=column, y='counts', ax=ax, legend=False)
#             plt.xlabel('Country')
#             plt.ylabel('Counts')
#             plt.title(f'Top {selected_k} countries with highest counts of {noun}')
#             plt.xticks(rotation=90)
#             st.pyplot(fig)
#             plt.close()
#             fig, ax = plt.subplots(figsize=(20, 10))
#             middle_k_countries.plot(kind='bar', x=column, y='counts', ax=ax, legend=False)
#             plt.xlabel('Country')
#             plt.ylabel('Counts')
#             plt.title(f'Middle {selected_k} countries with highest counts of {noun}')
#             plt.xticks(rotation=90)
#             st.pyplot(fig)
#             plt.close()
#             fig, ax = plt.subplots(figsize=(20, 10))
#             bottom_k_countries.plot(kind='bar', x=columnb, y='counts', ax=ax, legend=False)
#             plt.xlabel('Country')
#             plt.ylabel('Counts')
#             plt.title(f'Bottom {selected_k} countries with highest counts of {noun}')
#             plt.xticks(rotation=90)
#             st.pyplot(fig)
#             plt.close()
# elif st.session_state.section != 'Top-K countries':
#     st.session_state.topk = False
#     st.session_state.section = section