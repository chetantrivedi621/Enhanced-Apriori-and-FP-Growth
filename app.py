import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE SETUP ---------------- #
st.set_page_config(page_title="Apriori Product Analysis", layout="wide")
st.title("🛒 Product Associativity Analysis (Apriori Algorithm)")

# ---------------- LOAD DATA ---------------- #
@st.cache_data
def load_data():
    df = pd.read_excel('Online_Retail_Cleaned.xlsx', engine='openpyxl')
    
    # Cleaning
    df.dropna(subset=['Description', 'InvoiceNo'], inplace=True)
    df['Description'] = df['Description'].str.strip()
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df = df[df['Quantity'] > 0]
    
    return df

df = load_data()

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("📊 Parameters")

countries = sorted(df['Country'].unique())
selected_country = st.sidebar.selectbox("Select Country", countries)

min_support = st.sidebar.slider("Minimum Support", 0.01, 0.2, 0.07)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 1.0, 0.5)
min_lift = st.sidebar.slider("Minimum Lift", 1.0, 5.0, 1.2)

# ---------------- OPTIMIZED BASKET ---------------- #
def create_basket(data, country):
    
    # Filter country
    data = data[data['Country'] == country]
    
    # SAMPLE DATA (speed boost)
    data = data.sample(min(len(data), 10000))
    
    # KEEP TOP PRODUCTS ONLY (speed boost)
    top_items = data['Description'].value_counts().head(100).index
    data = data[data['Description'].isin(top_items)]
    
    # Create basket
    basket = (data.groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().fillna(0))
    
    # Convert to binary
    return basket.applymap(lambda x: 1 if x > 0 else 0)

# ---------------- RUN MODEL ---------------- #
if st.button("🚀 Run Apriori Analysis"):
    
    with st.spinner("Processing... Please wait ⏳"):
        
        basket = create_basket(df, selected_country)
        
        # APRIORI (OPTIMIZED)
        frequent_items = apriori(
            basket,
            min_support=min_support,
            use_colnames=True,
            max_len=2   # IMPORTANT FOR SPEED
        )
        
        if frequent_items.empty:
            st.warning("No frequent itemsets found. Increase data or reduce support.")
        
        else:
            rules = association_rules(
                frequent_items,
                metric="lift",
                min_threshold=min_lift
            )
            
            rules = rules[rules['confidence'] >= min_confidence]
            
            if rules.empty:
                st.warning("No strong rules found. Try lowering thresholds.")
            
            else:
                st.success(f"✅ Found {len(rules)} strong rules")
                
                # ---------------- TABLE ---------------- #
                st.subheader("📋 Association Rules")
                st.dataframe(
                    rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
                    .sort_values(by='lift', ascending=False)
                    .head(20)
                )
                
                # ---------------- GRAPHS ---------------- #
                col1, col2 = st.columns(2)

                # 1. Support vs Confidence
                with col1:
                    st.subheader("Support vs Confidence")
                    fig1, ax1 = plt.subplots()
                    sns.scatterplot(data=rules, x='support', y='confidence', hue='lift', ax=ax1)
                    st.pyplot(fig1)

                # 2. Support vs Lift
                with col2:
                    st.subheader("Support vs Lift")
                    fig2, ax2 = plt.subplots()
                    sns.scatterplot(data=rules, x='support', y='lift', hue='confidence', ax=ax2)
                    st.pyplot(fig2)

                # 3. Top Products
                st.subheader("Top 10 Products")
                top_items = basket.sum().sort_values(ascending=False).head(10)
                fig3, ax3 = plt.subplots()
                top_items.plot(kind='bar', ax=ax3)
                st.pyplot(fig3)

                # 4. Top Rules by Lift
                st.subheader("Top Rules by Lift")
                top_rules = rules.sort_values(by='lift', ascending=False).head(10)
                fig4, ax4 = plt.subplots()
                sns.barplot(x=top_rules['lift'], y=[str(i) for i in top_rules['antecedents']], ax=ax4)
                st.pyplot(fig4)

                # 5. Confidence Distribution
                st.subheader("Confidence Distribution")
                fig5, ax5 = plt.subplots()
                sns.histplot(rules['confidence'], bins=20, ax=ax5)
                st.pyplot(fig5)

# ---------------- RAW DATA ---------------- #
with st.expander("🔍 View Raw Data"):
    st.write(df.head())