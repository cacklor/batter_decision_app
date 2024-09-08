import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.image as mpimg
import streamlit as st

data = pd.read_csv('restrictedcounty copy.csv')
player_names = data['Batter'].unique()


@st.cache_data
def swingdecisions(batter,graphtype):
    # Load data
    data = pd.read_csv('restrictedcounty copy.csv')
    #print(data['Connection'].unique().tolist())
    # Remove outliers
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 2 * IQR
        upper_bound = Q3 + 2 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    data = remove_outliers(data, 'PastY')
    data = remove_outliers(data, 'PastZ')
    
    # Define coordinate ranges
    boxy = 0.25
    boxz = 1.3 / 8
    
    coordsy = [-1]
    coordsz = [0]
    
    
    
    # Generate coordinate ranges
    for i in coordsy:  # Create a copy of coordsy to iterate over
        nextlength = i + boxy
        if nextlength <= 1:
            coordsy.append(nextlength)
        else:
            break
        
    for i in coordsz:  # Create a copy of coordsz to iterate over
        nextlength = i + boxz
        if nextlength <= 1.3:
            coordsz.append(nextlength)
        else:
            break
        
    coordsy = [round(value, 2) for value in coordsy]  # Ensure initial values are rounded to 2dp
    coordsz = [round(value, 2) for value in coordsz]
    
    # Initialize the DataFrame for swings
    swings = pd.DataFrame(index=coordsy, columns=coordsz)
    swings = swings.apply(pd.to_numeric, errors='coerce')  # Ensure all entries are numeric or NaN
    
    # Process each batter
    player_names = data['Batter'].unique().tolist()
    
    
    
    player_data = data[data['Batter'] == batter]
    
    if graphtype == 'Swing':
        for i in coordsy:
           for j in coordsz:
                filtered_data = player_data[(player_data['PastY'] >= i) & (player_data['PastY'] < i + boxy) &
                                                (player_data['PastZ'] >= j) & (player_data['PastZ'] < j + boxz)]
        
                swing = 0
                non_swing = 0
        
                for index, row in filtered_data.iterrows():
                    if row['Shot'] in ['Back Defence', 'No Shot', 'Forward Defence', 'Padded Away', 'Drop and Run']:
                        non_swing += 1
                    elif pd.isna(row['Shot']):
                        continue
                    else:
                        swing += 1
        
                    # Calculate swing percentage and update the swings DataFrame
                if swing + non_swing > 0:
                    swing_percentage = swing / (swing + non_swing)
                else:
                    swing_percentage = 0
                    
                swings.loc[i, j] = swing_percentage
    
    elif graphtype == 'Middle':
            for i in coordsy:
               for j in coordsz:
                    filtered_data = player_data[(player_data['PastY'] >= i) & (player_data['PastY'] < i + boxy) &
                                                    (player_data['PastZ'] >= j) & (player_data['PastZ'] < j + boxz)]
            
                    middled = 0
                    not_middled = 0
            
                    for index, row in filtered_data.iterrows():
                        if row['Shot'] in ['No Shot', 'Padded Away','Left']:
                            continue
                        elif row['Connection'] in ['Middled']:
                            middled += 1
                        elif pd.isna(row['Connection']):
                            continue
                        else:
                            not_middled += 1
            
                        # Calculate swing percentage and update the swings DataFrame
                    if middled + not_middled > 0:
                        swing_percentage = middled / (middled + not_middled)
                    else:
                        swing_percentage = 0
                    
                    swings.loc[i, j] = swing_percentage
    
    elif graphtype == 'Edge':
            for i in coordsy:
               for j in coordsz:
                    filtered_data = player_data[(player_data['PastY'] >= i) & (player_data['PastY'] < i + boxy) &
                                                    (player_data['PastZ'] >= j) & (player_data['PastZ'] < j + boxz)]
            
                    edged = 0
                    not_edged = 0
            
                    for index, row in filtered_data.iterrows():
                        if row['Shot'] in ['No Shot', 'Padded Away','Left']:
                            continue
                        elif row['Connection'] in ['Inside Edge', 'Think Edge', 'Outside Edge', 'Leading Edge'
                                                   ,'Top Edge','Bottom Edge']:
                            edged += 1
                        elif pd.isna(row['Connection']):
                            continue
                        else:
                            not_edged += 1
            
                        # Calculate swing percentage and update the swings DataFrame
                    if edged + not_edged > 0:
                        swing_percentage = edged / (edged + not_edged)
                    else:
                        swing_percentage = 0
                        
                    swings.loc[i, j] = swing_percentage
    #print(f"Swings DataFrame for batter {batter}:")
    #print(swings)
    
    fig, ax = plt.subplots(figsize=(12, 9))
    stumpspath = 'stumps copy.png'
    stumps = mpimg.imread(stumpspath)
    
    # Create a heatmap
    cax = ax.matshow(swings, cmap='coolwarm', interpolation='nearest')
    
    # Add color bar
    fig.colorbar(cax)
    
    # Set up the grid labels
    ax.set_xticks(np.arange(len(swings.index)))
    ax.set_yticks(np.arange(len(swings.columns)))
    
    # Label the ticks
    ax.set_xticklabels(swings.index)
    ax.set_yticklabels(swings.columns)
    
    
    # Rotate the x labels
    plt.xticks(rotation=90)
    
    # Add values inside each grid cell
    for i in range(len(swings.columns)):
        for j in range(len(swings.index)):
            ax.text(j, i, f'{swings.iloc[i, j]:.2f}', ha='center', va='center', color='white')
    
    ax.invert_yaxis()
    

    ax.imshow(stumps,extent=[3.55, 4.45, -0.6, 4.2], alpha=0.7, zorder=2)
    ax.set_xlim(-0.5, 8.5)  # X-axis limits to cover all 9 columns
    ax.set_ylim(-0.5, 8.5)
    ax.set_xticks([])  # Remove X-axis ticks
    ax.set_yticks([])
    #ax.set_aspect(aspect=(2 / 1.3))
    plt.title('Grid Heatmap with Values for '+batter+' ('+graphtype+')', fontsize=13, pad=10)
    #plt.savefig('heatmap_plot.png', dpi=300)
    st.pyplot(fig)

st.title("Batter Heatmaps")
batter = st.selectbox("Select a Batsman", player_names)
graphtype = st.selectbox("Graph Type", ['Swing', 'Middle', 'Edge'])

if st.button("Generate Batter Graph"):
    fig = swingdecisions(batter, graphtype)
    