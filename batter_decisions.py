import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.image as mpimg
import streamlit as st

data = pd.read_csv('restrictedcounty copy.csv')
player_names = data['Batter'].unique()

@st.cache_data
def swingdecisions(batter):
    # Load data
    data = pd.read_csv('restrictedcounty copy.csv')
    
    # Remove outliers
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2.5 * IQR
        upper_bound = Q3 + 2.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    data = remove_outliers(data, 'PastY')
    data = remove_outliers(data, 'PastZ')
    
    # Define coordinate ranges
    boxy = 0.25
    boxz = 1.3 / 8
    
    coordsy = [-1]
    coordsz = [0]
    
    # Generate coordinate ranges
    for i in coordsy:  
        nextlength = i + boxy
        if nextlength <= 1:
            coordsy.append(nextlength)
        else:
            break
        
    for i in coordsz:  
        nextlength = i + boxz
        if nextlength <= 1.3:
            coordsz.append(nextlength)
        else:
            break
        
    coordsy = [round(value, 2) for value in coordsy]  
    coordsz = [round(value, 2) for value in coordsz]
    
    # Initialize the DataFrame for swings
    def initialize_swings():
        swings = pd.DataFrame(index=coordsy, columns=coordsz)
        swings = swings.apply(pd.to_numeric, errors='coerce') 
        return swings

    player_data = data[data['Batter'] == batter]

    # Create a function to calculate swing percentages based on graph type
    def calculate_percentages(player_data, coordsy, coordsz, boxy, boxz, mode):
        swings = initialize_swings()

        for i in coordsy:
            for j in coordsz:
                filtered_data = player_data[(player_data['PastY'] >= i) & (player_data['PastY'] < i + boxy) &
                                            (player_data['PastZ'] >= j) & (player_data['PastZ'] < j + boxz)]
                
                count1 = 0
                count2 = 0

                for _, row in filtered_data.iterrows():
                    if mode == 'Swing':
                        if row['Shot'] in ['Back Defence', 'No Shot', 'Forward Defence', 'Padded Away', 'Drop and Run']:
                            count2 += 1
                        elif not pd.isna(row['Shot']):
                            count1 += 1
                    elif mode == 'Middle':
                        if row['Shot'] not in ['No Shot', 'Padded Away', 'Left'] and row['Connection'] == 'Middled':
                            count1 += 1
                        elif not pd.isna(row['Connection']):
                            count2 += 1
                    elif mode == 'Edge':
                        if row['Shot'] not in ['No Shot', 'Padded Away', 'Left'] and row['Connection'] in [
                            'Inside Edge', 'Think Edge', 'Outside Edge', 'Leading Edge', 'Top Edge', 'Bottom Edge']:
                            count1 += 1
                        elif not pd.isna(row['Connection']):
                            count2 += 1

                total = count1 + count2
                if total > 0:
                    swings.loc[i, j] = count1 / total
                else:
                    swings.loc[i, j] = 0
        return swings

    # Calculate Swing, Middle, and Edge percentages
    swing_data = calculate_percentages(player_data, coordsy, coordsz, boxy, boxz, 'Swing')
    middle_data = calculate_percentages(player_data, coordsy, coordsz, boxy, boxz, 'Middle')
    edge_data = calculate_percentages(player_data, coordsy, coordsz, boxy, boxz, 'Edge')

    # Plot the data on a single figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)

    def create_heatmap(ax, data, title):
        cax = ax.matshow(data, cmap='coolwarm', interpolation='nearest')
        ax.set_xticks(np.arange(len(data.columns)))
        ax.set_yticks(np.arange(len(data.index)))
        ax.set_xticklabels(data.columns, rotation=90)
        ax.set_yticklabels(data.index)
        ax.set_title(title)

        for i in range(len(data.index)):
            for j in range(len(data.columns)):
                ax.text(j, i, f'{data.iloc[i, j]:.2f}', ha='center', va='center', color='white')

        ax.invert_yaxis()
        ax.set_xticks([])  # Remove X-axis ticks
        ax.set_yticks([])

    # Create heatmaps for Swing, Middle, and Edge
    create_heatmap(axes[0], swing_data, f'{batter} - Swing')
    create_heatmap(axes[1], middle_data, f'{batter} - Middle')
    create_heatmap(axes[2], edge_data, f'{batter} - Edge')

    # Display stumps on all heatmaps
    stumpspath = 'stumps copy.png'
    stumps = mpimg.imread(stumpspath)
    for ax in axes:
        ax.imshow(stumps, extent=[3.55, 4.45, -0.6, 4.2], alpha=0.7, zorder=2)
        ax.set_xlim(-0.5, 8.5)
        ax.set_ylim(-0.5, 8.5)

    plt.tight_layout()
    st.pyplot(fig)

# Streamlit App
st.title("Batter Heatmaps")
batter = st.selectbox("Select a Batsman", player_names)

if st.button("Generate Batter Graph"):
    swingdecisions(batter)
