
#used Chatgpt 4.0 to help set up streamlit homepage on OCt15th at 10am
#chatgpt was used ot convert the plots i used in my notebook(uploaded) to what is in the streamlit app
# Title of the app


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hiplot as hip

def load_data():
    try:
        non_cancelled_flights = pd.read_csv('non_cancelled_flights_2022_sampled.csv')
        canceled_flights = pd.read_csv('canceled_flights_2022_sampled.csv')
        weather_data = pd.read_csv('weather_data_2022_sampled.csv')
        return non_cancelled_flights, canceled_flights, weather_data
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

non_cancelled_flights, canceled_flights, weather_data = load_data()

if non_cancelled_flights is not None and canceled_flights is not None and weather_data is not None:
    # Title of the app
    st.title("Airline Timeliness Analysis and Uncovering Delay Causes")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page:", ["Home", "EDA", "Interactive Plots"])

    # Home Page
    if page == "Home":

        st.header("Project Description")
        project_description = (
            "The goal of this project is to analyze airline timeliness, focusing on flight delays and cancellations across various airlines, "
            "routes, and airports. By the end of the project, I aim to identify patterns in delays, "
            "determine the most common causes of delays, and explore trends based on different factors "
            "(e.g., day of the week, season, airline, airport). "
            "The user will develop insights that could help both airlines and consumers better understand the factors affecting flight punctuality."
        )
        st.text_area("Description:", project_description, height=200)

        st.header("Imputation Techniques")
        imputation_techniques = (
            "For the missing values in the flights dataset, I used KNN for imputation. "
            "This was useful because often flights have direct correlations with distance, meaning that certain flights, "
            "such as transatlantic or transpacific routes, often have similar distances. "
            "Furthermore, these flights typically have similar departure times, allowing me to use KNN in the distance-ordered "
            "dataset to impute the missing values."
        )
        st.text_area("Techniques Used:", imputation_techniques, height=100)

        st.header("Dataset Description")
        dataset_description = (
            "This work involved two datasets: one from the US Department of Transportation (DOT) and a weather events dataset downloaded from Kaggle. "
            "Both datasets had missing values, which were imputed using the techniques mentioned above."
        )
        st.text_area("Dataset Overview:", dataset_description, height=100)

        st.header("Next Steps in Modeling")
        next_steps = (
            "Going forward in the course, as we develop more skills in the modeling domain, I hope to develop a model that can predict "
            "some of the features that I have explored manually in the EDA section. "
            "This will help increase the usefulness of this app and make it valuable for users when picking their next plane ticket!"
        )
        st.text_area("Future Directions:", next_steps, height=100)

    # EDA Page
    elif page == "EDA":
        st.title("Exploratory Data Analysis (EDA)")

        if st.checkbox("Show Non-Canceled Flights Data"):
            st.write(non_cancelled_flights.head())

        if st.checkbox("Show Canceled Flights Data"):
            st.write(canceled_flights.head())

        if st.checkbox("Show Weather Data"):
            st.write(weather_data.head())

        if st.checkbox("Show Summary Statistics for Non-Canceled Flights"):
            st.write(non_cancelled_flights.describe())

        # Fixed element: Distribution of Flight Distance (Box Plot & Histogram)
        st.subheader('Distribution of Flight Distance')

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=[14.70, 8.27])

        # Box plot
        sns.boxplot(data=non_cancelled_flights, x='DISTANCE', ax=ax1, color='#6a9bc8')
        ax1.set(xlim=(-10, 3000))
        ax1.set_xlabel("Distance (miles)", size=10, weight="bold")

        # Histogram
        sns.histplot(data=non_cancelled_flights, x='DISTANCE', ax=ax2, kde=True, bins=50, color='#ff6f61')
        ax2.set(xlim=(-10, 3000))
        ax2.set_xlabel("Distance (miles)", size=10, weight="bold")

        # Draw mean distance
        mean_distance = non_cancelled_flights['DISTANCE'].mean()
        ax2.axvline(x=mean_distance, color='r', linestyle='--', label=f'Avg. Distance: {mean_distance:.2f}')
        plt.suptitle('Distribution of the "Distance" Variable for Non-Canceled Flights', weight='bold', size=14)
        ax2.legend()
        st.pyplot(fig)

        # Fixed element: Canceled and Diverted Flights Ratio
        st.subheader('Canceled and Diverted Flights Ratio')

        flights_2022_filtered = pd.concat([non_cancelled_flights, canceled_flights])

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=[14.7, 8.27])
        colors_canceled = ['#4C72B0', '#55A868']
        colors_diverted = ['#C44E52', '#8172B3']

        # Canceled flights pie chart
        labels_cancelled = flights_2022_filtered['CANCELLED'].value_counts().index.to_list()
        ax1.pie(flights_2022_filtered['CANCELLED'].value_counts(), explode=(0.2, 0), autopct='%.2f%%',
                startangle=180, counterclock=True, colors=colors_canceled)
        ax1.title.set(text="Canceled Flights Ratio", weight='bold', size=14)

        # Diverted flights pie chart
        labels_diverted = flights_2022_filtered['DIVERTED'].value_counts().index.to_list()
        ax2.pie(flights_2022_filtered['DIVERTED'].value_counts(), explode=(0.2, 0), autopct='%.2f%%',
                startangle=180, counterclock=True, colors=colors_diverted)
        ax2.title.set(text="Diverted Flights Ratio", weight='bold', size=14)

        ax1.legend(labels_cancelled, loc='lower center', ncol=2)
        ax2.legend(labels_diverted, loc='lower center', ncol=2)
        st.pyplot(fig)
        # Streamlit Title
        st.title("Cancellation Reasons Counts and Ratios (2022)")

        # Create the plot
        plt.figure(figsize=[14.70, 8.27])

        new_base_color = '#FF6F61'

        # Get the order for the plot based on the value counts of 'CANCELLATION_CODE'
        order = canceled_flights['CANCELLATION_CODE'].value_counts().index

        g = sns.countplot(data=canceled_flights, x='CANCELLATION_CODE', color=new_base_color, order=order)

        # Annotate the bars with counts and percentages using iloc for position-based access
        for bar in range(canceled_flights['CANCELLATION_CODE'].value_counts().shape[0]):
            count = canceled_flights['CANCELLATION_CODE'].value_counts().iloc[bar]
            pct = (count / canceled_flights['CANCELLATION_CODE'].shape[0]) * 100
            plt.text(x=bar, y=count, s=f"{count} ({pct:.2f}%)", va='top', ha='center')

        # Set a logarithmic scale for the y-axis and adjust the y-ticks
        plt.yscale('log')
        plt.yticks([100, 500, 5e3, 2e4], [100, 500, '5k', '20k'])

        g.set_xticklabels(['Carrier', 'Weather', 'National Air System', 'Security'])

        plt.xlabel('Cancellation Reason', fontsize=10, weight="bold")
        plt.ylabel('Count', fontsize=10, weight="bold")

        plt.title('Cancellation Reasons Counts and Ratio (2022)', weight='bold', size=14)

        # Show the plot in Streamlit
        st.pyplot(plt)
        # Convert date columns to datetime
        canceled_flights['FL_DATE'] = pd.to_datetime(canceled_flights['FL_DATE'], errors='coerce')

        # Ensure 'StartTime(UTC)' is in datetime format for snow events
        weather_data['StartTime(UTC)'] = pd.to_datetime(weather_data['StartTime(UTC)'], errors='coerce')

        # Snow Events
        snow_events = weather_data[weather_data['Type'] == 'Snow']
        snow_events['Month'] = snow_events['StartTime(UTC)'].dt.month_name()

        # Group by Month and Severity
        monthly_snow_counts = snow_events.groupby(['Month', 'Severity']).size().unstack(fill_value=0)

        # Monthly Canceled Flights
        month_counts = canceled_flights['FL_DATE'].dt.month_name().value_counts()
        sorted_month_counts = month_counts.sort_values(ascending=True)

        # Create side-by-side plots
        st.title("Monthly Cancellations and Snow Events Analysis")

        # Create a figure with 2 subplots
        fig, axes = plt.subplots(ncols=2, figsize=(14.70, 7))

        # Plot Canceled Flights per Month
        colors = sns.color_palette("coolwarm", len(sorted_month_counts))
        sns.barplot(
            x=sorted_month_counts.index,
            y=sorted_month_counts.values,
            palette=colors,
            ax=axes[0]
        )

        axes[0].set_xlabel('Month', fontsize=10, weight="bold")
        axes[0].set_ylabel('Count', fontsize=10, weight="bold")
        axes[0].set_title('Number of Canceled Flights per Month', fontsize=14, weight="bold")

        # Annotate bars
        for bar in range(len(sorted_month_counts)):
            count = sorted_month_counts.values[bar]
            axes[0].text(x=bar, y=count + 100, s=f'{count}', va='top', ha='center')

        # Plot Snow Occurrences per Month
        monthly_snow_counts['Total'] = monthly_snow_counts.sum(axis=1)
        sorted_month_order = monthly_snow_counts.index.tolist()
        colors = sns.color_palette("RdYlBu", len(monthly_snow_counts.columns) - 1)

        monthly_snow_counts.drop(columns='Total').reindex(sorted_month_order).plot(
            kind='bar', stacked=True, color=colors, ax=axes[1]
        )

        axes[1].set_xlabel('Month', fontsize=12, weight='bold')
        axes[1].set_ylabel('Number of Snow Occurrences', fontsize=12, weight='bold')
        axes[1].set_title('Number of Snow Occurrences per Month', fontsize=14, weight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].legend(title='Severity', fontsize=10)

        plt.tight_layout()
        st.pyplot(fig)

    # Interactive Plots Page
    elif page == "Interactive Plots":
        # Ensure 'FL_DATE' and 'Month' columns are properly formatted (assuming non_cancelled_flights and canceled_flights are loaded)
        non_cancelled_flights['FL_DATE'] = pd.to_datetime(non_cancelled_flights['FL_DATE'])
        non_cancelled_flights['Month'] = non_cancelled_flights['FL_DATE'].dt.month_name()

        canceled_flights['FL_DATE'] = pd.to_datetime(canceled_flights['FL_DATE'])
        canceled_flights['Month'] = canceled_flights['FL_DATE'].dt.month_name()

        # Group data
        total_flights = non_cancelled_flights.groupby(['Month', 'AIRLINE_CODE']).size().reset_index(name='TotalFlights')
        canceled_counts = canceled_flights.groupby(['Month', 'AIRLINE_CODE']).size().reset_index(name='CanceledCount')
        merged_counts = pd.merge(total_flights, canceled_counts, on=['Month', 'AIRLINE_CODE'], how='left').fillna(0)

        # Calculate percentage of canceled flights
        merged_counts['CanceledCount'] = merged_counts['CanceledCount'].astype(int)
        merged_counts['PercentageCanceled'] = (merged_counts['CanceledCount'] / merged_counts['TotalFlights']) * 100

        # Sort by cancellations
        monthly_cancellations = merged_counts.groupby('Month')['CanceledCount'].sum().reset_index()
        monthly_cancellations = monthly_cancellations.sort_values(by='CanceledCount')
        ordered_months = monthly_cancellations['Month'].tolist()
        merged_counts['Month'] = pd.Categorical(merged_counts['Month'], categories=ordered_months, ordered=True)
        merged_counts = merged_counts.sort_values(by=['Month', 'PercentageCanceled'])

        # Streamlit UI for interaction
        st.title("Interactive Canceled Flights Plot")
        st.sidebar.header("Filter Options")

        # Dropdown to select month
        selected_month = st.sidebar.selectbox("Select Month", options=ordered_months)

        # Multi-select for airline codes
        airline_codes = merged_counts['AIRLINE_CODE'].unique()
        selected_airlines = st.sidebar.multiselect("Select Airline Codes", options=airline_codes, default=airline_codes)

        # Filter data based on user selection
        filtered_data = merged_counts[
            (merged_counts['Month'] == selected_month) & (merged_counts['AIRLINE_CODE'].isin(selected_airlines))]

        # Create the interactive bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        color_palette = sns.color_palette("husl", len(filtered_data['AIRLINE_CODE'].unique()))

        sns.barplot(data=filtered_data, x='AIRLINE_CODE', y='PercentageCanceled', ax=ax, palette=color_palette,
                    order=filtered_data['AIRLINE_CODE'])
        ax.set_title(f'Percentage of Canceled Flights for {selected_month}', fontsize=16, weight='bold')
        ax.set_ylabel('Percentage of Cancellations (%)', fontsize=14, weight='bold')
        ax.set_xlabel('Airline Code', fontsize=14, weight='bold')
        ax.tick_params(axis='x', rotation=45)

        # Display plot
        st.pyplot(fig)

        # Ensure date columns are in datetime format
        canceled_flights['FL_DATE'] = pd.to_datetime(canceled_flights['FL_DATE'], errors='coerce')
        weather_data['StartTime(UTC)'] = pd.to_datetime(weather_data['StartTime(UTC)'], errors='coerce')

        # Major airlines and their colors
        major_airlines = ['DL', 'UA', 'AA']  # Delta, United, American Airlines
        color_palette = {'DL': 'blue', 'UA': 'green', 'AA': 'red'}  # Assign colors

        # Filter dataset for major airlines
        major_airlines_data = non_cancelled_flights[non_cancelled_flights['AIRLINE_CODE'].isin(major_airlines)]

        # Define the relevant delay columns
        delay_columns = ['DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY',
                         'DELAY_DUE_LATE_AIRCRAFT']

        # Melt the DataFrame to get the delay type as an x-axis category
        melted_df = major_airlines_data.melt(id_vars=['AIRLINE_CODE'], value_vars=delay_columns,
                                             var_name='DelayType', value_name='DelayMinutes')

        # Remove rows with NaN delay minutes
        melted_df = melted_df.dropna(subset=['DelayMinutes'])

        # Convert DelayType to a categorical variable with the correct order
        delay_categories = ['DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS', 'DELAY_DUE_SECURITY',
                            'DELAY_DUE_LATE_AIRCRAFT']
        melted_df['DelayType'] = pd.Categorical(melted_df['DelayType'], categories=delay_categories, ordered=True)

        # Streamlit layout
        st.title("Delay Distribution by Major Airlines")

        # Selectbox for choosing the airline
        selected_airline = st.selectbox("Select Airline:", major_airlines)

        # Filter data for the selected airline
        filtered_data = melted_df[melted_df['AIRLINE_CODE'] == selected_airline]

        # Create the strip plot
        plt.figure(figsize=(14.70, 8.27))

        g = sns.stripplot(data=filtered_data, x='DelayType', y='DelayMinutes', hue='AIRLINE_CODE',
                          dodge=True, linewidth=0.5, palette=color_palette)

        # Set the custom x-axis tick labels to match the delay types
        g.set_xticklabels(['Carrier', 'Weather', 'Nat. Air Sys.', 'Security', 'Late Aircraft'])

        # Title, labels, and legend
        plt.title(f"Distribution of Delay by Type for {selected_airline}", fontsize=14, weight="bold")
        plt.ylabel('Delay (Minutes)', fontsize=10, weight="bold")
        plt.xlabel('Delay Type', fontsize=10, weight="bold")

        # Display the legend and plot
        plt.legend(title='Airline', title_fontsize='13', fontsize='11')
        plt.xticks(rotation=45)  # Rotate x-ticks for better readability

        # Show plot in Streamlit
        st.pyplot(plt)
