import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import plotly.express as px
import lightgbm as lgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from catboost import CatBoostClassifier
from io import StringIO


df = pd.read_csv("/Users/_fangkhai/Downloads/Test Assessment/application.csv")

df1 = pd.read_csv("/Users/_fangkhai/Downloads/Test Assessment/offers.csv")

combined_df = pd.merge(df, df1, on='appId', how='left')
combined_df.fillna("N/A", inplace = True)

closed_app_df = combined_df[combined_df['applic_status_name'] == 'Closed App']
expired_app_df = combined_df[combined_df['applic_status_name'] == 'Expired App']
loan_disbursed_app_df = combined_df[combined_df['applic_status_name'] == 'Loan Disbursed App']
bank_declined_app_df = combined_df[combined_df['applic_status_name'] == 'Bank Declined App']

columns_to_drop = ['applicationTime', 'pendingOfferTime', 'offerGivenTime', 'offerChosenTime', 'disburseTime', 'offerId']
closed_app_df = closed_app_df.drop(columns=columns_to_drop)
expired_app_df = expired_app_df.drop(columns=columns_to_drop)
loan_disbursed_app_df = loan_disbursed_app_df.drop(columns=columns_to_drop)
bank_declined_app_df = bank_declined_app_df.drop(columns=columns_to_drop)

def nationality_count(dataset_name):
    nationality_counts = dataset_name['nationality'].value_counts()
    nationality_counts_df = nationality_counts.reset_index()
    nationality_counts_df.columns = ['Nationality', 'Count']
    nationality_options = nationality_counts_df['Nationality'].tolist()
    
    country_mapping = {
        'MY': 'Malaysia',
        'PH': 'Philippines',
        'IN': 'India',
        'CN': 'China',
        'ID': 'Indonesia',
        'MM': 'Myanmar',
        'TH': 'Thailand',
        'NO': 'Norway',
        'GB': 'United Kingdom',
        'NZ': 'New Zealand',
        'TW': 'Taiwan',
        'JP': 'Japan',
        'NL': 'Netherlands',
        'PE': 'Peru',
        'VN': 'Vietnam',
        'FR': 'France',
        'HK': 'Hong Kong',
        'KR': 'South Korea',
        'LK': 'Sri Lanka',
        'NP': 'Nepal'
    }
    nationality_counts_df['Country'] = nationality_counts_df['Nationality'].map(country_mapping)
    plot_data = nationality_counts_df.groupby('Country').sum().reset_index()

    fig = px.choropleth(plot_data,
                        locations='Country',
                        locationmode='country names',
                        color='Count',
                        hover_name='Country',
                        title='Overview of Closed Apps by Nationality',
                        color_continuous_scale=px.colors.sequential.Viridis)
    st.plotly_chart(fig)
    selected_nationality = st.selectbox('**Choose a nationality:**', nationality_options)
    if selected_nationality:
        count = nationality_counts_df[nationality_counts_df['Nationality'] == selected_nationality]['Count'].values[0]
        st.write(f"###### The count for {selected_nationality} is: {count}")

def age_distribution(dataset_name):
    st.title('Age Distribution of Applicants')
    plt.figure(figsize=(10, 5))
    sns.kdeplot(dataset_name['age'], fill=True, color='skyblue', alpha=0.6)
    plt.xlabel('Age')
    plt.ylabel('Density')
    plt.grid(True)
    st.pyplot(plt)
    
    st.write("")
    st.write("")
    selected_age = st.selectbox('**Select an Age to View Frequency:**', sorted(dataset_name['age'].unique()))
    age_frequency = dataset_name[dataset_name['age'] == selected_age].shape[0]
    st.write(f"###### Frequency of Applicants at Age {selected_age}: {age_frequency}")

def civil_status(dataset_name):
    filtered_statuses_1 = ['Queued', 'Bank Approved IPA', 'Bank Declined IPA', 'PR Fail IPA/LG']
    filtered_df_1 = dataset_name[dataset_name['offer_status_name'].isin(filtered_statuses_1)]
    status_counts_1 = filtered_df_1.groupby(['civilStatus', 'offer_status_name']).size().unstack(fill_value=0)

    st.title('Civil Status Impact on Applications')
    plt.figure(figsize=(10, 6))
    status_counts_1.plot(kind='bar', ax=plt.gca(), width=0.8, color=['lightblue', 'lightgreen', 'salmon', 'red'])
    plt.xlabel('Civil Status', fontsize=14)
    plt.ylabel('Count of Applications', fontsize=14)
    plt.xticks(rotation=0)
    plt.legend(title='Approval Status', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(plt)

    filtered_statuses_2 = [
        'Customer Reject IPA/LG', 'Loan Rejected - Post', 
        'Pending IPA', 'Loan Rejected - Pre', 'Acknowledged', 'Chosen - Manual Review'
    ]
    filtered_df_2 = dataset_name[dataset_name['offer_status_name'].isin(filtered_statuses_2)]
    status_counts_2 = filtered_df_2.groupby(['civilStatus', 'offer_status_name']).size().unstack(fill_value=0)
    plt.figure(figsize=(10, 6))
    status_counts_2.plot(kind='bar', ax=plt.gca(), width=0.8, color=plt.cm.tab20.colors)
    plt.xlabel('Civil Status', fontsize=14)
    plt.ylabel('Count of Applications', fontsize=14)
    plt.xticks(rotation=0)
    plt.legend(title='Approval Status', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(plt)
    st.write("")
    st.write("")
    combined_filtered_df = pd.concat([filtered_df_1, filtered_df_2])
    selected_civil_status = st.selectbox('**Select a Civil Status to View Combined Status Counts:**', combined_filtered_df['civilStatus'].unique())
    selected_status_counts = combined_filtered_df[combined_filtered_df['civilStatus'] == selected_civil_status].groupby('offer_status_name').size()
    st.write(f"###### Status Counts for {selected_civil_status}:")
    st.dataframe(selected_status_counts.reset_index(name='Count'))

def residency_count(dataset_name):
    residency_counts = dataset_name['residency'].value_counts().reset_index()
    residency_counts.columns = ['Residency Status', 'Number of Applicants']
    fig = px.treemap(residency_counts,
                    path=['Residency Status'],
                    values='Number of Applicants',
                    color='Number of Applicants',
                    color_continuous_scale='Viridis')
    fig.update_traces(textinfo='label+value')
    st.title('Residency Distribution of Applicants')
    st.plotly_chart(fig, use_container_width=True)

def applied_amount(dataset_name):
    st.title('Distribution of Applied Amounts')
    plt.figure(figsize=(10, 6))
    sns.histplot(dataset_name['appliedAmount'], bins=30)
    plt.xlabel('Applied Amount')
    plt.ylabel('Frequency')
    st.pyplot(plt)

def monthly_income(dataset_name):
    st.title('Applied Amount VS Monthly Income')
    grouped_data = dataset_name.groupby(['monthlyIncome', 'loanPurpose']).agg(
        total_applied_amount=('appliedAmount', 'sum'),
        average_applied_amount=('appliedAmount', 'mean'),
        count=('appliedAmount', 'size')
    ).reset_index()

    min_count = int(grouped_data['count'].min())
    max_count = int(grouped_data['count'].max())
    selected_count_range = st.slider(
        "**Select Count Range (Number of Applications)**",
        min_value=min_count,
        max_value=max_count,
        value=(min_count, min_count + 10)
    )
    lower_count, upper_count = selected_count_range
    filtered_grouped_data = grouped_data[(grouped_data['count'] >= lower_count) & 
                                        (grouped_data['count'] <= upper_count)]

    st.write(f"###### Filtered Grouped Data (Count: {lower_count} - {upper_count} applications)", filtered_grouped_data)
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=filtered_grouped_data, x='monthlyIncome', y='total_applied_amount', hue='loanPurpose', alpha=0.6)
    plt.title(f'Applied Amount vs. Monthly Income by Loan Purpose (Count: {lower_count} - {upper_count})')
    plt.xlabel('Monthly Income')
    plt.ylabel('Total Applied Amount')
    plt.legend(title='Loan Purpose', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(plt)

def income_category(dataset_name):
    income_bins = [0, 3000, 5000, 7000, 10000, 15000]
    income_labels = ['<3k', '3k-5k', '5k-7k', '7k-10k', '>10k']
    dataset_name['incomeCategory'] = pd.cut(dataset_name['monthlyIncome'], bins=income_bins, labels=income_labels)
    st.title('Loan Amounts by Monthly Income Categories')
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=dataset_name, x='incomeCategory', y='appliedAmount')
    plt.xlabel('Income Category')
    plt.ylabel('Applied Amount')
    st.pyplot(plt) 

    selected_category = st.selectbox(
        "**Select Monthly Income Category**",
        income_labels
    )
    
    filtered_data = dataset_name[dataset_name['incomeCategory'] == selected_category]
    iqr = filtered_data['appliedAmount'].quantile(0.75) - filtered_data['appliedAmount'].quantile(0.25)
    q1 = filtered_data['appliedAmount'].quantile(0.25)
    q3 = filtered_data['appliedAmount'].quantile(0.75)
    median = filtered_data['appliedAmount'].median()
    mean = filtered_data['appliedAmount'].mean()
    std_dev = filtered_data['appliedAmount'].std()

    st.write(f"**Interquartile Range (IQR):** {iqr:.2f}")
    st.write(f"**25th Percentile (Q1):** {q1:.2f}")
    st.write(f"**75th Percentile (Q3):** {q3:.2f}")
    st.write(f"**Median:** {median:.2f}")
    st.write(f"**Mean:** {mean:.2f}")
    st.write(f"**Standard Deviation:** {std_dev:.2f}")

def loan_purpose(dataset_name):
    loan_purpose_counts = dataset_name['loanPurpose'].value_counts()
    st.title('Frequency of Loan Purposes')
    plt.figure(figsize=(10, 6))
    loan_purpose_counts.plot(kind='barh', color='lightblue')
    plt.xlabel('Number of Applications')
    plt.ylabel('Loan Purpose')
    st.pyplot(plt)
    st.write("")
    st.write("")
    selected_loan_purpose = st.selectbox(
        "**Select a Loan Purpose to See Its Frequency**",
        loan_purpose_counts.index 
    )
    selected_purpose_count = loan_purpose_counts[selected_loan_purpose]
    st.write(f"**Number of Applications for** '{selected_loan_purpose}': {selected_purpose_count}")

def loan_approval_analysis(dataset_name):
    approval_counts = dataset_name.groupby('loanPurpose').agg(
        total_apps=('offer_status_name', 'count'),
        approved_apps=('offer_status_name', lambda x: (x.str.contains('Bank Approved')).sum())
    ).reset_index()
    
    approval_counts['approval_rate'] = (approval_counts['approved_apps'] / approval_counts['total_apps']) * 100
    approval_counts = approval_counts.sort_values(by='approval_rate', ascending=False)

    st.title('Approval Rates by Loan Purpose')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=approval_counts, x='approval_rate', y='loanPurpose', palette='viridis')

    plt.xlabel('Approval Rate (%)', fontsize=12)
    plt.ylabel('Loan Purpose', fontsize=12)
    st.pyplot(plt)

    selected_loan_purposes = st.multiselect(
        "**Select Two Loan Purposes to Compare Their Approval Statistics**",
        approval_counts['loanPurpose'].unique(), 
        default=approval_counts['loanPurpose'].unique()[:2]
    )
    if len(selected_loan_purposes) == 2:
        selected_data = approval_counts[approval_counts['loanPurpose'].isin(selected_loan_purposes)]
        st.write(selected_data[['loanPurpose', 'total_apps', 'approved_apps', 'approval_rate']])
    else:
        st.warning("Please select exactly two loan purposes for comparison.")

def declined_approval_analysis(dataset_name):
    declined_counts = dataset_name.groupby('loanPurpose').agg(
        total_apps=('offer_status_name', 'count'),
        declined_apps=('offer_status_name', lambda x: (x.str.contains('Bank Declined')).sum())
    ).reset_index()

    declined_counts['declined_rate'] = (declined_counts['declined_apps'] / declined_counts['total_apps']) * 100
    declined_counts = declined_counts.sort_values(by='declined_rate', ascending=False)

    st.title('Declined Rates by Loan Purpose')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=declined_counts, x='declined_rate', y='loanPurpose', palette='magma')
    plt.xlabel('Declined Rate (%)', fontsize=12)
    plt.ylabel('Loan Purpose', fontsize=12)
    st.pyplot(plt)

    selected_loan_purposes = st.multiselect(
        "**Select Two Loan Purposes to Compare Their Declined Rates**",
        declined_counts['loanPurpose'].unique(),
        default=declined_counts['loanPurpose'].unique()[:2] 
    )
    if len(selected_loan_purposes) == 2:
        selected_data = declined_counts[declined_counts['loanPurpose'].isin(selected_loan_purposes)]
        st.write(selected_data[['loanPurpose', 'total_apps', 'declined_apps', 'declined_rate']])
    else:
        st.warning("Please select exactly two loan purposes for comparison.")

def approval_rate_by_lender(dataset_name):
    total_applications = len(dataset_name)
    approved_applications = len(dataset_name[dataset_name['offer_status_name'] == 'Bank Approved IPA'])
    approval_rate = (approved_applications / total_applications) * 100

    approval_by_lender = dataset_name.groupby('lenderName')['offer_status_name'].value_counts().unstack(fill_value=0)
    approval_by_lender_rate = (approval_by_lender['Bank Approved IPA'] / approval_by_lender.sum(axis=1)) * 100

    st.title('Approval Rates by Lender')
    plt.figure(figsize=(10, 10))
    approval_by_lender_rate.sort_values().plot(kind='barh', color='lightgreen', ax=plt.gca())
    plt.xlabel('Approval Rate (%)')
    plt.ylabel('Lender')
    st.pyplot(plt)

    default_lenders = approval_by_lender_rate.nlargest(3).index.tolist()

    selected_lenders = st.multiselect(
        "**Select up to 3 Lenders to Compare Approval Rates**",
        approval_by_lender_rate.index.tolist(),
        default=default_lenders,
        max_selections=4
    )

    if len(selected_lenders) > 0:
        selected_data = approval_by_lender_rate[selected_lenders]
        selected_data_df = selected_data.reset_index(name='Approval Rate')
        st.write(selected_data_df)
    else:
        st.warning("Please select up to 3 lenders to compare.")

def decline_rate_by_lender(dataset_name):
    total_applications = len(dataset_name)
    declined_applications = len(dataset_name[dataset_name['offer_status_name'] == 'Bank Declined IPA'])
    decline_rate = (declined_applications / total_applications) * 100

    decline_by_lender = dataset_name.groupby('lenderName')['offer_status_name'].value_counts().unstack(fill_value=0)
    decline_by_lender_rate = (decline_by_lender['Bank Declined IPA'] / decline_by_lender.sum(axis=1)) * 100

    st.title('Decline Rates by Lender')
    plt.figure(figsize=(10, 10))
    decline_by_lender_rate.sort_values().plot(kind='barh', color='salmon', ax=plt.gca())
    plt.xlabel('Decline Rate (%)')
    plt.ylabel('Lender')
    st.pyplot(plt)
    default_lenders = decline_by_lender_rate.nlargest(3).index.tolist()

    selected_lenders = st.multiselect(
        "**Select up to 3 Lenders to Compare Decline Rates**",
        decline_by_lender_rate.index.tolist(),
        default=default_lenders,
        max_selections=4
    )

    if len(selected_lenders) > 0:
        selected_data = decline_by_lender_rate[selected_lenders]
        selected_data_df = selected_data.reset_index(name='Decline Rate')
        st.write(selected_data_df)
    else:
        st.warning("Please select up to 3 lenders to compare.")

def conversion_rate_by_lead_source(dataset_name):
    lead_source_counts = dataset_name.groupby('leadSource')['offer_status_name'].value_counts().unstack(fill_value=0)
    conversion_rate = (lead_source_counts['Bank Approved IPA'] / lead_source_counts.sum(axis=1)) * 100

    st.title('Conversion Rates by Lead Source')
    plt.figure(figsize=(10, 8))
    conversion_rate.sort_values().plot(kind='barh', color='blue')
    plt.ylabel('Lead Source')
    plt.xlabel('Conversion Rate (%)') 
    plt.axvline(x=0, color='gray', linestyle='--')
    st.pyplot(plt)

def loan_purpose_by_lender(dataset_name):
    lender_purpose = dataset_name.groupby(['lenderName', 'loanPurpose']).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 10))
    lender_purpose.plot(kind='barh', stacked=True, ax=ax)
    ax.set_title('Loan Purpose by Lender')
    ax.set_xlabel('Number of Applications')
    ax.set_ylabel('Lender')
    ax.legend(title='Loan Purpose', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.title("Loan Purpose by Lender")
    st.pyplot(fig)

def work_status(dataset_name):
    employment_approval_rate = dataset_name.groupby('workStatus')['offer_status_name'].value_counts(normalize=True).unstack().fillna(0)
    employment_approval_rate = employment_approval_rate.reset_index()
    employment_approval_rate['Approval Rate'] = employment_approval_rate.get('Bank Approved IPA', 0) * 100

    st.title('Approval Rates by Employment Status')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(employment_approval_rate['workStatus'], employment_approval_rate['Approval Rate'], color=plt.cm.Paired.colors)
    ax.set_xlabel('Employment Status')
    ax.set_ylabel('Approval Rate (%)')
    ax.set_xticklabels(employment_approval_rate['workStatus'], rotation=45)
    ax.set_ylim(0, 100)
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)
    
    approval_rate_table = employment_approval_rate[['workStatus', 'Approval Rate']]
    approval_rate_table.columns = ['Employment Status', 'Approval Rate (%)']  # Rename columns for display
    st.table(approval_rate_table)

def industry_analysis(dataset_name):
    industry_analysis = dataset_name.groupby('jobIndustry')['appliedAmount'].agg(['count', 'mean'])
    approval_counts = dataset_name[dataset_name['offer_status_name'] == 'Bank Approved IPA'].groupby('jobIndustry')['offer_status_name'].count()
    industry_analysis['Approval Rate'] = approval_counts / industry_analysis['count'] * 100
    fig, ax1 = plt.subplots(figsize=(12, 6))
    industry_analysis[['count', 'mean']].plot(kind='bar', ax=ax1, color=['lightblue', 'lightgreen'])

    ax2 = ax1.twinx()
    industry_analysis['Approval Rate'].plot(kind='line', marker='o', color='salmon', ax=ax2, linewidth=2)
    ax1.set_xlabel('Job Industry')
    ax1.set_ylabel('Number of Applications / Average Amount')
    ax2.set_ylabel('Approval Rate (%)')
    ax1.set_xticks(range(len(industry_analysis)))
    ax1.set_xticklabels(industry_analysis.index, rotation=45)
    plt.tight_layout()
    st.title('Loan Requests VS Approval Rates by Job Industry')
    st.pyplot(fig)

def correlation_analysis(dataset_name):
    correlation_matrix = dataset_name[['age', 'monthlyIncome', 'appliedAmount']].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, ax=ax)
    st.title('Correlation Matrix')
    st.pyplot(fig)

def page_home():
    st.title("Lendela's Loan Landscape 📊")
    st.write("Name: Foo Fang Khai")
    st.write("Field Of Study: Bachelor of Computer Science (HONS) specialise in Data Science")
    st.write("")
    st.header("Objective 🚀")
    st.write("This project aims to analyze loan application and offer datasets to derive insights that can inform strategic decisions at Lendela. The analysis encompasses various dimensions:")
    st.write(" • Demographic Analysis")
    st.write(" • Financial Analysis")
    st.write(" • Loan Purpose Analysis")
    st.write(" • Approval Status Analysis")
    st.write(" • Source of Leads")
    st.write(" • Lender Performance Analysis")
    st.write(" • Work Status and Industry Analysis")
    st.write(" • Correlation Analysis")
    st.write(" • Predictive Modeling")

    st.image("get_started.gif", use_column_width = True)

def data_engineering():
    st.title("Data Engineering")
    st.write("This page focuses on data processing and cleaning.")

    st.header("Application Dataset 📂")
    st.write(df)
    st.write("This dataset consists of ",df.shape[0], " rows and ", df.shape[1], " columns")

    duplicates = df[df.duplicated()]
    if duplicates.empty:
        st.write("There're no duplicates")
    else:
        st.write("Duplicate Rows:" + duplicates)

    col1, col2 = st.columns(2)

    with col1:
        st.header("Descriptive Statistics 🔍")
        st.write(df.describe())
    with col2:
        st.header("Summary of DataFrame 📝")
        buffer = StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    col1, col2 = st.columns(2)

    with col1: 
        st.header("Null Values ❌")
        st.write(df.isnull().sum())
        st.write("This dataset consists of ",df.shape[0], " rows and ", df.shape[1], " columns")
    with col2:
        st.header("Cleaned Data ✅")
        df.fillna("N/A", inplace = True)
        st.write(df.isnull().sum())
        st.write("This dataset consists of ",df.shape[0], " rows and ", df.shape[1], " columns")

    st.markdown("---") 

    st.header("Offer Dataset 📂")
    st.write(df1)
    st.write("This dataset consists of ",df1.shape[0], " rows and ", df1.shape[1], " columns")

    duplicates = df1[df1.duplicated()]
    if duplicates.empty:
        st.write("There're no duplicates")
    else:
        st.write("Duplicate Rows:" + duplicates)

    col1, col2 = st.columns(2)

    with col1:
        st.header("Descriptive Statistics 🔍")
        st.write(df1.describe())
    with col2:
        st.header("Summary of DataFrame 📝")
        buffer = StringIO()
        df1.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    col1, col2 = st.columns(2)

    with col1: 
        st.header("Null Values ❌")
        st.write(df1.isnull().sum())
        st.write("This dataset consists of ",df1.shape[0], " rows and ", df1.shape[1], " columns")
    with col2:
        st.header("Cleaned Data ✅")
        df1.fillna("N/A", inplace = True)
        st.write(df1.isnull().sum())
        st.write("This dataset consists of ",df1.shape[0], " rows and ", df1.shape[1], " columns")

    st.markdown("---") 
    
    st.title("URGHHHH")
    st.write("However, there’s one action we can take regarding these two datasets. What could that be?")
    st.image("question-marks-why.gif", use_column_width = False)

def data_concatenate():
    st.title("Dataset Concatenate")
    st.write("This page focuses on merging Application and Offer dataset.")

    st.header("Combined Dataset 📂")
    st.write(combined_df)
    st.write("This dataset consists of ",combined_df.shape[0], " rows and ", combined_df.shape[1], " columns")

    duplicates = combined_df[combined_df.duplicated()]
    if duplicates.empty:
        st.write("There're no duplicates")
    else:
        st.write("Duplicate Rows:" + duplicates)

    col1, col2 = st.columns(2)

    with col1:
        st.header("Descriptive Statistics 🔍")
        st.write(combined_df.describe())
    with col2:
        st.header("Summary of DataFrame 📝")
        buffer = StringIO()
        combined_df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

    st.markdown("---") 

    st.header("Segmenting Data by Application Status")
    st.write("I grouped applic_status_name into four categories: Closed App, Expired App, Loan Disbursed App, and Bank Declined App. This helps us analyze trends in application status, which is crucial for understanding loan disbursement likelihood. By refining our processes based on these insights, we can reduce declines and expirations, ultimately improving customer satisfaction.")

    st.header("Closed App")
    st.write(closed_app_df)
    st.write("This dataset consists of ",closed_app_df.shape[0], " rows and ", closed_app_df.shape[1], " columns")
    st.write("")
    st.header("Expired App")
    st.write(expired_app_df)
    st.write("This dataset consists of ",expired_app_df.shape[0], " rows and ", expired_app_df.shape[1], " columns")
    st.write("")
    st.header("Loan Disbursed App")
    st.write(loan_disbursed_app_df)
    st.write("This dataset consists of ",loan_disbursed_app_df.shape[0], " rows and ", loan_disbursed_app_df.shape[1], " columns")
    st.write("")
    st.header("Bank Declined App")
    st.write(bank_declined_app_df)
    st.write("This dataset consists of ",bank_declined_app_df.shape[0], " rows and ", bank_declined_app_df.shape[1], " columns")

    st.markdown("---") 
    st.title('Application Status Distribution')
    status_counts = combined_df['applic_status_name'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    st.markdown("---") 
    st.header("HRMMMM, DATA STILL NOT CLEANED")
    st.write("I have decided to drop the columns 'applicationTime', 'pendingOfferTime', 'offerGivenTime', 'offerChosenTime', 'disburseTime', and 'offerId' as they do not contribute valuable insights into applicant statuses or loan approval factors. Removing these columns will streamline the dataset and improve interpretability.")

    st.image("datacleaned.gif", use_column_width = True)

def app_analysis(dataset_name):
    st.markdown("---")
    st.title("Overview of Dataset") 
    st.write(dataset_name)
    st.write("This dataset consists of ",dataset_name.shape[0], " rows and ", dataset_name.shape[1], " columns")
    st.markdown("---") 
    st.title('Count of Applications by Nationality')
    nationality_count(dataset_name)
    st.markdown("---") 
    age_distribution(dataset_name)
    st.markdown("---") 
    civil_status(dataset_name)
    st.markdown("---")
    residency_count(dataset_name)
    st.markdown("---")  
    applied_amount(dataset_name)
    st.markdown("---")
    monthly_income(dataset_name)
    st.markdown("---")
    income_category(dataset_name)
    st.markdown("---") 
    loan_purpose(dataset_name)
    st.markdown("---") 
    loan_approval_analysis(dataset_name)
    st.markdown("---")
    declined_approval_analysis(dataset_name)
    st.markdown("---")
    approval_rate_by_lender(dataset_name)
    st.markdown("---")    
    decline_rate_by_lender(dataset_name)
    st.markdown("---")  
    conversion_rate_by_lead_source(dataset_name)
    st.markdown("---")      
    loan_purpose_by_lender(dataset_name)
    st.markdown("---") 
    work_status(dataset_name)
    st.markdown("---") 
    industry_analysis(dataset_name)
    st.markdown("---")   
    correlation_analysis(dataset_name)
    st.markdown("---")     

def model_evaluation():
    st.title("Model Evaluation")
    st.write("This page focuses on evaluating several machine learning algorithms that gives us the best accuracy in predicting the outcome.")  

    le_civil_status = LabelEncoder()
    combined_df['civilStatus'] = le_civil_status.fit_transform(combined_df['civilStatus'])
    le_loan_purpose = LabelEncoder()
    combined_df['loanPurpose'] = le_loan_purpose.fit_transform(combined_df['loanPurpose'])
    le_nationality = LabelEncoder()
    combined_df['nationality'] = le_nationality.fit_transform(combined_df['nationality'])
    le_work_status = LabelEncoder()
    combined_df['workStatus'] = le_work_status.fit_transform(combined_df['workStatus'])    

    scaler = StandardScaler()
    combined_df[['age', 'appliedAmount', 'monthlyIncome']] = scaler.fit_transform(combined_df[['age', 'appliedAmount', 'monthlyIncome']])

    X = combined_df[['age', 'appliedAmount', 'civilStatus', 'loanPurpose', 'monthlyIncome', 'nationality', 'workStatus']]
    y = combined_df['applic_status_name']

    split_ratio = st.selectbox("**Select Train-Test Split Ratio**", options=["70 : 30", "80 : 20", "90 : 10"])
    if split_ratio == "70 : 30":
        test_size = 0.3
    elif split_ratio == "80 : 20":
        test_size = 0.2
    else:
        test_size = 0.1
    st.markdown("---")  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    st.header("**Random Forest Classifier**")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    st.write(f'Random Forest Classifier Accuracy: {accuracy:.2f}')
    st.text(report)
    st.markdown("---")  

    st.header("**LightGBM Classifier**")
    lgb_model = lgb.LGBMClassifier(random_state=42)
    lgb_model.fit(X_train, y_train)
    y_pred_lgb = lgb_model.predict(X_test)
    accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
    report_lgb = classification_report(y_test, y_pred_lgb)
    st.write(f'LightGBM Accuracy: {accuracy_lgb:.2f}')
    st.text(report_lgb)
    st.markdown("---")  

    st.header("**CatBoost Classifier**")
    catboost_model = CatBoostClassifier(silent=True, random_state=42)
    catboost_model.fit(X_train, y_train)
    y_pred_cat = catboost_model.predict(X_test)
    accuracy_cat = accuracy_score(y_test, y_pred_cat)
    report_cat = classification_report(y_test, y_pred_cat)
    st.write(f'CatBoost Accuracy: {accuracy_cat:.2f}')
    st.text(report_cat)
    st.markdown("---")  
        
    
def prediction():
    st.title("Prediction ⚙️")
    st.write("This page focuses on utilising several machine learning algorithms that gives us the best accuracy in predicting the outcome.")  
    st.markdown("---")  
    st.subheader("**Counter Check for Prediction**")

    dataset_selection = st.selectbox("**Dataset Selection**", options=["Expired Application", "Closed Application", "Bank Declined Application", "Loan Disbursed Application"])
    if dataset_selection == "Expired Application":
        st.write(expired_app_df)
    elif dataset_selection == "Closed Application":
        st.write(closed_app_df)
    elif dataset_selection == "Bank Declined Application":
        st.write(bank_declined_app_df)
    elif dataset_selection == "Loan Disbursed Application":
        st.write(loan_disbursed_app_df)   
    st.markdown("---") 
    
    st.subheader("**Prediction for Loan Status**") 
    le_civil_status = LabelEncoder()
    combined_df['civilStatus'] = le_civil_status.fit_transform(combined_df['civilStatus'])
    le_loan_purpose = LabelEncoder()
    combined_df['loanPurpose'] = le_loan_purpose.fit_transform(combined_df['loanPurpose'])
    le_nationality = LabelEncoder()
    combined_df['nationality'] = le_nationality.fit_transform(combined_df['nationality'])
    le_work_status = LabelEncoder()
    combined_df['workStatus'] = le_work_status.fit_transform(combined_df['workStatus'])    
    scaler = StandardScaler()
    combined_df[['age', 'appliedAmount', 'monthlyIncome']] = scaler.fit_transform(combined_df[['age', 'appliedAmount', 'monthlyIncome']])
    X = combined_df[['age', 'appliedAmount', 'civilStatus', 'loanPurpose', 'monthlyIncome', 'nationality', 'workStatus']]
    y = combined_df['applic_status_name']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    age = st.number_input("Age", min_value=18, max_value=80, value=18)
    applied_amount = st.number_input("Applied Amount", min_value=1000, max_value=250000, value=1000)
    monthly_income = st.number_input("Monthly Income", min_value=0, max_value=2382323, value=0)

    civil_status = st.selectbox("Civil Status", options=["single", "married", "divorced", "widow"])
    loan_purpose = st.selectbox("Loan Purpose", options=['debt-consolidation', 'business-expansion', 'paying-bills', 'credit-card-debt', 'debt-consolidation-ff', 'other', 'home', 'renovation', 'wedding', 'medical', 'investment', 'car', 'line-of-credit', 'special-occasion', 'education', 'hobbies', 'vacation'])
    nationality = st.selectbox("Nationality", options=['SG', 'LB', 'MY', 'IN', 'PH', 'ID', 'CN', 'AU', 'HK', 'MM', 'FR', 'VN', 'NP', 'GB', 'TH', 'BD', 'PE', 'AR', 'CA', 'NO', 'JP', 'US', 'IT', 'NL', 'AX', 'TW', 'NZ', 'KR', 'LK', 'RU'])
    work_status = st.selectbox("Work Status", options=['employed-salary', 'self-employed', 'unemployed', 'other', 'employed-contract', 'national-serviceman', 'retired', 'student'])

    new_data = pd.DataFrame({
        'age': [age],
        'appliedAmount': [applied_amount],
        'civilStatus': [le_civil_status.transform([civil_status])[0]], 
        'loanPurpose': [le_loan_purpose.transform([loan_purpose])[0]], 
        'monthlyIncome': [monthly_income],
        'nationality': [le_nationality.transform([nationality])[0]], 
        'workStatus': [le_work_status.transform([work_status])[0]]
    })
    new_data[['age', 'appliedAmount', 'monthlyIncome']] = scaler.transform(new_data[['age', 'appliedAmount', 'monthlyIncome']])
    st.write("")
    if st.button("Predict"):
        predicted_status = model.predict(new_data)
        st.write(f"**Predicted Application Status: {predicted_status}**")

def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("Loan Status Analysis")
    page_options = {
        "Home": "🏠",
        "Data Engineering": "🛠️",
        "Dataset Concatenation": "✏️",
        "Application Analysis": "🔔",
        "Model Evaluation": "🔮",
        "Prediction": "💻",
    }

    selected_page = st.sidebar.selectbox("", list(page_options.keys()), format_func=lambda x: f"{page_options[x]} {x}")

    if selected_page == "Home":
        st.sidebar.image("logo.gif", use_column_width=True)
        page_home()
    elif selected_page == "Data Engineering":
        st.sidebar.image("datacleaning.gif", use_column_width=True)
        data_engineering()
    elif selected_page == "Dataset Concatenation":
        st.sidebar.image("dataconcat.gif", use_column_width=True)
        data_concatenate()
    elif selected_page == "Application Analysis":
        st.sidebar.image("closedapp.gif", use_column_width=True)
        st.title("Application Analysis 📈")
        st.write("This page focuses on analysing different application status trends.")
        dataset_option = st.selectbox(
            "Select a dataset to view:",
            options=["Expired Application", "Closed Application", "Loan Disbursed Application", "Bank Declined Application"]
        )
        if dataset_option == "Expired Application":
            app_analysis(expired_app_df)
        elif dataset_option == "Closed Application":
            app_analysis(closed_app_df)
        elif dataset_option == "Loan Disbursed Application":
            app_analysis(loan_disbursed_app_df)
        elif dataset_option == "Bank Declined Application":
            app_analysis(bank_declined_app_df)
    elif selected_page == "Model Evaluation":
        st.sidebar.image("conclusion.gif", use_column_width=True)
        model_evaluation()
    elif selected_page == "Prediction":
        st.sidebar.image("prediction.gif", use_column_width=True)
        prediction()
            
if __name__ == "__main__":
    main()
