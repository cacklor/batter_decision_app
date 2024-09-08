import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import streamlit as st

# Load the CSV file once outside the function (best for caching and performance)
data = pd.read_csv('restrictedcounty copy.csv')
player_names = data['Batter'].unique()


@st.cache_data
def swingdecisions(batter, graphtype):
    # Load data inside the cached function
    data = pd.read_csv('restrictedcounty copy.csv')

    # Function to remove outliers
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
    while coordsy[-1] + boxy <= 1:
        coordsy.append(round(coordsy[-1] + boxy, 2))
        
    while coordsz[-1] + boxz <= 1.3:
        coordsz.append(round(coordsz[-1] + boxz, 2))

    # Initialize DataFrame for swings
    swings = pd.DataFrame(index=coordsy, columns=coordsz).apply(pd.to_numeric, errors='coerce')

    player_data = data[data['Batter'] == batter]
    
    if player_data.empty:
        st.error(f"No data available for {batter}.")
        return None

    # Swing decision logic
    if graphtype == 'Swing':
        for i in coordsy:
            for j in coordsz:
                filtered_data = player_data[(player_data['PastY'] >= i) & (player_data['PastY'] < i + boxy) &
                                            (player_data['PastZ'] >= j) & (player_data['PastZ'] < j + boxz)]
                swing = filtered_data[filtered_data['Shot'].notna() & 
                                      ~filtered_data['Shot'].isin(['Back Defence', 'No Shot', 
                                                                    'Forward Defence', 'Padded Away', 'Drop and Run'])].shape[0]
                non_swing = filtered_data.shape[0] - swing
                
                swing_percentage = swing / (swing + non_swing) if swing + non_swing > 0 else 0
                swings.loc[i, j] = swing_percentage
    
    # Middle logic
    elif graphtype == 'Middle':
        for i in coordsy:
            for j in coordsz:
                filtered_data = player_data[(player_data['PastY'] >= i) & (player_data['PastY'] < i + boxy) &
                                            (player_data['PastZ'] >= j) & (player_data['PastZ'] < j + boxz)]
                middled = filtered_data[filtered_data['Connection'] == 'Middled'].shape[0]
                not_middled = filtered_data.shape[0] - middled
                
                swing_percentage = middled / (middled + not_middled) if middled + not_middled > 0 else 0
                swings.loc[i, j] = swing_percentage

    # Edge logic
    elif graphtype == 'Edge':
        for i in coordsy:
            for j in coordsz:
                filtered_data = player_data[(player_data['PastY'] >= i) & (player_data['PastY'] < i + boxy) &
                                            (player_data['PastZ'] >= j) & (player_data['PastZ'] < j + boxz)]
                edged = filtered_data[filtered_data['Connection'].isin(['Inside Edge', 'Thin Edge', 
                                                                       'Outside Edge', 'Leading Edge', 
                                                                       'Top Edge', 'Bottom Edge'])].shape[0]
                not_edged = filtered_data.shape[0] - edged
                
                swing_percentage = edged / (edged + not_edged) if edged + not_edged > 0 else 0
                swings.loc[i, j] = swing_percentage

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 9))
    stumpspath = 'stumps copy.png'
    stumps = mpimg.imread(stumpspath)

    # Create a heatmap
    cax = ax.matshow(swings, cmap='coolwarm', interpolation='nearest')

    # Add color bar
    fig.colorbar(cax)

    # Label setup
    ax.set_xticks(np.arange(len(swings.columns)))
    ax.set_yticks(np.arange(len(swings.index)))
    ax.set_xticklabels(swings.columns)
    ax.set_yticklabels(swings.index)

    # Rotate the x labels
    plt.xticks(rotation=90)

    # Add values inside each grid cell
    for i in range(len(swings.columns)):
        for j in range(len(swings.index)):
            ax.text(j, i, f'{swings.iloc[i, j]:.2f}', ha='center', va='center', color='white')

    ax.invert_yaxis()

    # Add stumps image
    ax.imshow(stumps, extent=[3.55, 4.45, -0.6, 4.2], alpha=0.7, zorder=2)
    ax.set_xlim(-0.5, 8.5)  # X-axis limits to cover all 9 columns
    ax.set_ylim(-0.5, 8.5)
    ax.set_xticks([])  # Remove X-axis ticks
    ax.set_yticks([])
    
    plt.title(f'Grid Heatmap with Values for {batter} ({graphtype})', fontsize=13, pad=10)
    return fig


# Streamlit app title
st.title("Batter Heatmaps")

# Select a Batsman
batter = st.selectbox("Select a Batsman", player_names)

# Select the Graph Type
graphtype = st.selectbox("Graph Type", ['Swing', 'Middle', 'Edge'])

# Generate the graph on button click
if st.button("Generate Batter Graph"):
    fig = swingdecisions(batter, graphtype)
    if fig:
        st.pyplot(fig)
