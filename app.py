
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

DATA_FILE = "NR_dataset.xlsx"

st.set_page_config(
    page_title="NovaRetail Customer Intelligence Dashboard",
    page_icon="🛍️",
    layout="wide",
)


def clean_column_name(col: str) -> str:
    return (
        str(col)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def format_currency(value: float) -> str:
    if pd.isna(value):
        return "$0"
    return f"${value:,.0f}"


def format_number(value: float) -> str:
    if pd.isna(value):
        return "0"
    return f"{value:,.0f}"


def safe_mode(series: pd.Series):
    series = series.dropna()
    if series.empty:
        return np.nan
    mode = series.mode()
    return mode.iloc[0] if not mode.empty else series.iloc[0]


def normalize_category(category: str) -> str:
    if pd.isna(category):
        return "Other"

    category = str(category).strip()

    mapping = {
        "Electronics": "Electronics",
        "Gaming": "Electronics",
        "Clothing": "Clothing",
        "Fashion": "Clothing",
        "Fashion & Apparel": "Clothing",
        "Fashion Accessories": "Clothing",
        "Children's Clothing": "Clothing",
        "Sportswear": "Clothing",
        "Groceries": "Groceries",
        "Grocery": "Groceries",
        "Grocery Items": "Groceries",
        "Food & Beverages": "Groceries",
        "Books": "Books",
        "Books & Magazines": "Books",
        "Home Appliances": "Home Products",
        "Home Decor": "Home Products",
        "Furniture": "Home Products",
        "Furniture & Decor": "Home Products",
        "Home & Garden": "Home Products",
        "Home Improvement": "Home Products",
        "Gardening Tools": "Home Products",
        "Beauty Products": "Beauty & Health",
        "Beauty & Personal Care": "Beauty & Health",
        "Cosmetics": "Beauty & Health",
        "Health & Beauty": "Beauty & Health",
        "Health & Wellness": "Beauty & Health",
        "Health Supplements": "Beauty & Health",
        "Sporting Goods": "Sports & Outdoors",
        "Sports & Outdoors": "Sports & Outdoors",
        "Sports Equipment": "Sports & Outdoors",
        "Outdoor Equipment": "Sports & Outdoors",
        "Toys": "Toys & Games",
        "Toys & Games": "Toys & Games",
        "Office Supplies": "Office Supplies",
        "Automotive": "Automotive",
    }
    return mapping.get(category, "Other")


@st.cache_data
def load_data() -> pd.DataFrame:
    file_path = Path(DATA_FILE)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find '{DATA_FILE}'. Place the dataset in the same folder as app.py."
        )

    df = pd.read_excel(file_path)
    df.columns = [clean_column_name(c) for c in df.columns]

    rename_map = {
        "label": "customer_segment",
    }
    df = df.rename(columns=rename_map)

    if "idx" in df.columns:
        df = df.drop(columns=["idx"])

    required_columns = {
        "customer_segment",
        "customer_id",
        "transaction_id",
        "transaction_date",
        "product_category",
        "purchase_amount",
        "customer_age_group",
        "customer_gender",
        "customer_region",
        "customer_satisfaction",
        "retail_channel",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    df["customer_segment"] = df["customer_segment"].fillna("Unknown")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["purchase_amount"] = pd.to_numeric(df["purchase_amount"], errors="coerce").fillna(0)
    df["customer_satisfaction"] = pd.to_numeric(
        df["customer_satisfaction"], errors="coerce"
    )
    df["category_group"] = df["product_category"].apply(normalize_category)
    df["order_month"] = df["transaction_date"].dt.to_period("M").dt.to_timestamp()
    df["year"] = df["transaction_date"].dt.year
    df["month_name"] = df["transaction_date"].dt.strftime("%b %Y")

    segment_order = ["Promising", "Growth", "Stable", "Decline", "Unknown"]
    df["customer_segment"] = pd.Categorical(
        df["customer_segment"], categories=segment_order, ordered=True
    )

    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    min_date = df["transaction_date"].min().date()
    max_date = df["transaction_date"].max().date()

    selected_dates = st.sidebar.date_input(
        "Transaction Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date, end_date = min_date, max_date

    segment_options = sorted(
        [str(x) for x in df["customer_segment"].dropna().astype(str).unique()],
        key=lambda x: ["Promising", "Growth", "Stable", "Decline", "Unknown"].index(x)
        if x in ["Promising", "Growth", "Stable", "Decline", "Unknown"]
        else 999,
    )
    selected_segments = st.sidebar.multiselect(
        "Customer Segment", options=segment_options, default=segment_options
    )

    region_options = sorted(df["customer_region"].dropna().astype(str).unique())
    selected_regions = st.sidebar.multiselect(
        "Region", options=region_options, default=region_options
    )

    category_options = sorted(df["category_group"].dropna().astype(str).unique())
    selected_categories = st.sidebar.multiselect(
        "Category Group", options=category_options, default=category_options
    )

    channel_options = sorted(df["retail_channel"].dropna().astype(str).unique())
    selected_channels = st.sidebar.multiselect(
        "Retail Channel", options=channel_options, default=channel_options
    )

    gender_options = sorted(df["customer_gender"].dropna().astype(str).unique())
    selected_genders = st.sidebar.multiselect(
        "Customer Gender", options=gender_options, default=gender_options
    )

    age_options = sorted(df["customer_age_group"].dropna().astype(str).unique())
    selected_ages = st.sidebar.multiselect(
        "Customer Age Group", options=age_options, default=age_options
    )

    filtered = df[
        (df["transaction_date"].dt.date >= start_date)
        & (df["transaction_date"].dt.date <= end_date)
        & (df["customer_segment"].astype(str).isin(selected_segments))
        & (df["customer_region"].astype(str).isin(selected_regions))
        & (df["category_group"].astype(str).isin(selected_categories))
        & (df["retail_channel"].astype(str).isin(selected_channels))
        & (df["customer_gender"].astype(str).isin(selected_genders))
        & (df["customer_age_group"].astype(str).isin(selected_ages))
    ].copy()

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Filtered rows: {len(filtered):,}")

    return filtered


def build_customer_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    latest_segment = (
        df.sort_values(["customer_id", "transaction_date"])
        .groupby("customer_id", as_index=False)
        .tail(1)[["customer_id", "customer_segment"]]
        .rename(columns={"customer_segment": "latest_segment"})
    )

    summary = (
        df.groupby("customer_id", as_index=False)
        .agg(
            total_revenue=("purchase_amount", "sum"),
            total_orders=("transaction_id", pd.Series.nunique),
            avg_satisfaction=("customer_satisfaction", "mean"),
            first_purchase=("transaction_date", "min"),
            last_purchase=("transaction_date", "max"),
            primary_region=("customer_region", safe_mode),
            primary_channel=("retail_channel", safe_mode),
            primary_age_group=("customer_age_group", safe_mode),
            primary_gender=("customer_gender", safe_mode),
        )
        .merge(latest_segment, on="customer_id", how="left")
    )

    summary["days_since_last_purchase"] = (
        df["transaction_date"].max() - summary["last_purchase"]
    ).dt.days
    return summary


def metric_row(df: pd.DataFrame):
    total_revenue = df["purchase_amount"].sum()
    unique_customers = df["customer_id"].nunique()
    unique_orders = df["transaction_id"].nunique()
    aov = total_revenue / unique_orders if unique_orders else 0
    avg_satisfaction = df["customer_satisfaction"].mean()
    revenue_per_customer = total_revenue / unique_customers if unique_customers else 0

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Revenue", format_currency(total_revenue))
    col2.metric("Unique Customers", format_number(unique_customers))
    col3.metric("Unique Orders", format_number(unique_orders))
    col4.metric("Avg Order Value", format_currency(aov))
    col5.metric(
        "Avg Satisfaction",
        f"{avg_satisfaction:.2f}/5" if pd.notna(avg_satisfaction) else "N/A",
    )
    col6.metric("Revenue per Customer", format_currency(revenue_per_customer))


def show_empty_state():
    st.warning("No data matches the current filter selection. Adjust the filters to continue.")
    st.stop()


def create_bar(data, x, y, title, text_auto=".2s", horizontal=False):
    if data.empty:
        st.info(f"No data available for {title.lower()}.")
        return
    if horizontal:
        fig = px.bar(data, x=y, y=x, orientation="h", title=title, text_auto=text_auto)
    else:
        fig = px.bar(data, x=x, y=y, title=title, text_auto=text_auto)
    fig.update_layout(height=430, xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)


def create_line(data, x, y, title):
    if data.empty:
        st.info(f"No data available for {title.lower()}.")
        return
    fig = px.line(data, x=x, y=y, markers=True, title=title)
    fig.update_layout(height=430, xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)


def create_scatter(data, x, y, size, color, hover_name, title):
    if data.empty:
        st.info(f"No data available for {title.lower()}.")
        return
    fig = px.scatter(
        data,
        x=x,
        y=y,
        size=size,
        color=color,
        hover_name=hover_name,
        text=hover_name,
        title=title,
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def create_heatmap(data, x, y, z, title):
    if data.empty:
        st.info(f"No data available for {title.lower()}.")
        return
    fig = px.imshow(
        data,
        labels={"x": x, "y": y, "color": z},
        aspect="auto",
        title=title,
        text_auto=True,
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


try:
    df = load_data()
except Exception as e:
    st.error(f"Unable to load the dataset. {e}")
    st.stop()

filtered_df = apply_filters(df)

st.title("NovaRetail Customer Intelligence Dashboard")
st.markdown(
    """
    Explore customer behavior, revenue patterns, segment risk, and growth opportunities across
    regions, categories, channels, and customer demographics. Use the filters in the sidebar to
    dynamically slice the data and uncover actionable business insights.
    """
)

if filtered_df.empty:
    show_empty_state()

metric_row(filtered_df)

tabs = st.tabs(["Executive Overview", "Segment Health & Risk", "Growth Opportunities"])

with tabs[0]:
    st.subheader("Executive Overview")

    overview_left, overview_right = st.columns(2)

    segment_revenue = (
        filtered_df.groupby("customer_segment", observed=False, as_index=False)["purchase_amount"]
        .sum()
        .sort_values("purchase_amount", ascending=False)
    )
    with overview_left:
        create_bar(segment_revenue, "customer_segment", "purchase_amount", "Revenue by Segment")
        if not segment_revenue.empty:
            top_seg = segment_revenue.iloc[0]
            st.caption(
                f"Highest revenue segment in the current view: **{top_seg['customer_segment']}** "
                f"with **{format_currency(top_seg['purchase_amount'])}**."
            )

    monthly_revenue = (
        filtered_df.groupby("order_month", as_index=False)["purchase_amount"].sum()
        .sort_values("order_month")
    )
    with overview_right:
        create_line(monthly_revenue, "order_month", "purchase_amount", "Revenue Trend Over Time")
        if not monthly_revenue.empty:
            best_month = monthly_revenue.sort_values("purchase_amount", ascending=False).iloc[0]
            st.caption(
                f"Peak revenue month: **{best_month['order_month'].strftime('%b %Y')}** "
                f"at **{format_currency(best_month['purchase_amount'])}**."
            )

    row2_col1, row2_col2 = st.columns(2)

    region_revenue = (
        filtered_df.groupby("customer_region", as_index=False)["purchase_amount"]
        .sum()
        .sort_values("purchase_amount", ascending=False)
    )
    with row2_col1:
        create_bar(region_revenue, "customer_region", "purchase_amount", "Revenue by Region")
        if not region_revenue.empty:
            top_region = region_revenue.iloc[0]
            st.caption(
                f"Top-performing region: **{top_region['customer_region']}** "
                f"with **{format_currency(top_region['purchase_amount'])}** in revenue."
            )

    channel_revenue = (
        filtered_df.groupby("retail_channel", as_index=False)["purchase_amount"]
        .sum()
        .sort_values("purchase_amount", ascending=False)
    )
    with row2_col2:
        create_bar(channel_revenue, "retail_channel", "purchase_amount", "Revenue by Retail Channel")
        if not channel_revenue.empty:
            top_channel = channel_revenue.iloc[0]
            st.caption(
                f"Best-performing channel: **{top_channel['retail_channel']}** "
                f"with **{format_currency(top_channel['purchase_amount'])}**."
            )

    st.markdown("### Top Customers")
    customer_summary = build_customer_summary(filtered_df)
    top_customers = (
        customer_summary.sort_values("total_revenue", ascending=False)
        .head(10)
        .rename(
            columns={
                "customer_id": "Customer ID",
                "latest_segment": "Latest Segment",
                "total_revenue": "Total Revenue",
                "total_orders": "Orders",
                "avg_satisfaction": "Avg Satisfaction",
                "primary_region": "Primary Region",
                "primary_channel": "Primary Channel",
                "days_since_last_purchase": "Days Since Last Purchase",
            }
        )
    )

    if top_customers.empty:
        st.info("No customers available for the current filters.")
    else:
        display_top = top_customers.copy()
        display_top["Total Revenue"] = display_top["Total Revenue"].map(lambda x: f"${x:,.2f}")
        display_top["Avg Satisfaction"] = display_top["Avg Satisfaction"].map(lambda x: f"{x:.2f}")
        st.dataframe(
            display_top[
                [
                    "Customer ID",
                    "Latest Segment",
                    "Total Revenue",
                    "Orders",
                    "Avg Satisfaction",
                    "Primary Region",
                    "Primary Channel",
                    "Days Since Last Purchase",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

with tabs[1]:
    st.subheader("Segment Health & Risk")

    segment_health = (
        filtered_df.groupby("customer_segment", observed=False, as_index=False)
        .agg(
            revenue=("purchase_amount", "sum"),
            customers=("customer_id", pd.Series.nunique),
            orders=("transaction_id", pd.Series.nunique),
            avg_satisfaction=("customer_satisfaction", "mean"),
        )
        .sort_values("revenue", ascending=False)
    )

    risk_col1, risk_col2 = st.columns(2)

    with risk_col1:
        create_scatter(
            segment_health.dropna(subset=["avg_satisfaction"]),
            x="avg_satisfaction",
            y="revenue",
            size="customers",
            color="customer_segment",
            hover_name="customer_segment",
            title="Revenue vs Satisfaction by Segment",
        )
        if not segment_health.empty:
            lowest_sat = segment_health.dropna(subset=["avg_satisfaction"]).sort_values(
                "avg_satisfaction", ascending=True
            )
            if not lowest_sat.empty:
                risk_seg = lowest_sat.iloc[0]
                st.caption(
                    f"Lowest satisfaction segment: **{risk_seg['customer_segment']}** "
                    f"at **{risk_seg['avg_satisfaction']:.2f}/5**."
                )

    with risk_col2:
        customer_count = segment_health[["customer_segment", "customers"]].sort_values(
            "customers", ascending=False
        )
        create_bar(customer_count, "customer_segment", "customers", "Customer Count by Segment")
        sat_data = segment_health[["customer_segment", "avg_satisfaction"]].sort_values(
            "avg_satisfaction", ascending=True
        )
        create_bar(sat_data, "customer_segment", "avg_satisfaction", "Average Satisfaction by Segment")

    st.markdown("### Decline Segment Diagnostics")
    decline_df = filtered_df[filtered_df["customer_segment"].astype(str) == "Decline"]

    decline_col1, decline_col2 = st.columns(2)
    with decline_col1:
        decline_region = (
            decline_df.groupby("customer_region", as_index=False)["purchase_amount"]
            .sum()
            .sort_values("purchase_amount", ascending=False)
        )
        create_bar(
            decline_region,
            "customer_region",
            "purchase_amount",
            "Decline Segment Revenue by Region",
        )

    with decline_col2:
        decline_category = (
            decline_df.groupby("category_group", as_index=False)["purchase_amount"]
            .sum()
            .sort_values("purchase_amount", ascending=False)
        )
        create_bar(
            decline_category,
            "category_group",
            "purchase_amount",
            "Decline Segment Revenue by Category Group",
            horizontal=True,
        )

    st.markdown("### At-Risk Customers")
    risk_customers = build_customer_summary(filtered_df)

    if not risk_customers.empty:
        risk_customers["risk_flag"] = np.where(
            (risk_customers["latest_segment"].astype(str) == "Decline")
            & (risk_customers["avg_satisfaction"].fillna(0) <= 3),
            "Decline + Low Satisfaction",
            np.where(
                risk_customers["latest_segment"].astype(str) == "Decline",
                "Decline Segment",
                np.where(
                    risk_customers["avg_satisfaction"].fillna(999) <= 2.5,
                    "Low Satisfaction",
                    "Monitor",
                ),
            ),
        )

        risk_view = risk_customers[
            (risk_customers["latest_segment"].astype(str) == "Decline")
            | (risk_customers["avg_satisfaction"].fillna(999) <= 2.5)
            | (risk_customers["days_since_last_purchase"].fillna(0) > 20)
        ].copy()

        risk_view = risk_view.sort_values(
            ["risk_flag", "days_since_last_purchase", "total_revenue"],
            ascending=[True, False, False],
        )

        if risk_view.empty:
            st.info("No at-risk customers identified under the current filters.")
        else:
            risk_display = risk_view.rename(
                columns={
                    "customer_id": "Customer ID",
                    "latest_segment": "Latest Segment",
                    "total_revenue": "Total Revenue",
                    "total_orders": "Orders",
                    "avg_satisfaction": "Avg Satisfaction",
                    "primary_region": "Primary Region",
                    "primary_channel": "Primary Channel",
                    "days_since_last_purchase": "Days Since Last Purchase",
                    "risk_flag": "Risk Flag",
                }
            ).copy()

            risk_display["Total Revenue"] = risk_display["Total Revenue"].map(lambda x: f"${x:,.2f}")
            risk_display["Avg Satisfaction"] = risk_display["Avg Satisfaction"].map(
                lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
            )

            st.dataframe(
                risk_display[
                    [
                        "Customer ID",
                        "Latest Segment",
                        "Risk Flag",
                        "Total Revenue",
                        "Orders",
                        "Avg Satisfaction",
                        "Days Since Last Purchase",
                        "Primary Region",
                        "Primary Channel",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

with tabs[2]:
    st.subheader("Growth Opportunities")

    growth_df = filtered_df[
        filtered_df["customer_segment"].astype(str).isin(["Promising", "Growth"])
    ].copy()

    if growth_df.empty:
        st.info("No Promising or Growth segment data is available for the selected filters.")
    else:
        growth_col1, growth_col2 = st.columns(2)

        growth_region = (
            growth_df.groupby("customer_region", as_index=False)["purchase_amount"]
            .sum()
            .sort_values("purchase_amount", ascending=False)
        )
        with growth_col1:
            create_bar(
                growth_region,
                "customer_region",
                "purchase_amount",
                "Promising + Growth Revenue by Region",
            )

        growth_category = (
            growth_df.groupby("category_group", as_index=False)["purchase_amount"]
            .sum()
            .sort_values("purchase_amount", ascending=False)
        )
        with growth_col2:
            create_bar(
                growth_category,
                "category_group",
                "purchase_amount",
                "Promising + Growth Revenue by Category Group",
                horizontal=True,
            )

        growth_col3, growth_col4 = st.columns(2)

        channel_by_segment = (
            growth_df.groupby(["retail_channel", "customer_segment"], observed=False, as_index=False)[
                "purchase_amount"
            ]
            .sum()
            .sort_values("purchase_amount", ascending=False)
        )
        with growth_col3:
            if channel_by_segment.empty:
                st.info("No data available for channel performance by segment.")
            else:
                fig = px.bar(
                    channel_by_segment,
                    x="retail_channel",
                    y="purchase_amount",
                    color="customer_segment",
                    barmode="group",
                    title="Channel Performance by Segment",
                    text_auto=".2s",
                )
                fig.update_layout(height=430, xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig, use_container_width=True)

        age_by_segment = (
            growth_df.groupby(["customer_age_group", "customer_segment"], observed=False, as_index=False)[
                "purchase_amount"
            ]
            .sum()
            .sort_values("purchase_amount", ascending=False)
        )
        with growth_col4:
            if age_by_segment.empty:
                st.info("No data available for age-group performance by segment.")
            else:
                fig = px.bar(
                    age_by_segment,
                    x="customer_age_group",
                    y="purchase_amount",
                    color="customer_segment",
                    barmode="group",
                    title="Age-Group Performance by Segment",
                    text_auto=".2s",
                )
                fig.update_layout(height=430, xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Region × Category Opportunity Heatmap")
        heatmap_data = growth_df.pivot_table(
            index="customer_region",
            columns="category_group",
            values="purchase_amount",
            aggfunc="sum",
            fill_value=0,
        )
        create_heatmap(
            heatmap_data,
            x="Category Group",
            y="Region",
            z="Revenue",
            title="Revenue Heatmap: Region by Category Group",
        )

        st.markdown("### Opportunity Summary")
        opportunity = (
            growth_df.groupby(["customer_region", "category_group"], as_index=False)
            .agg(
                revenue=("purchase_amount", "sum"),
                customers=("customer_id", pd.Series.nunique),
                avg_satisfaction=("customer_satisfaction", "mean"),
                orders=("transaction_id", pd.Series.nunique),
            )
            .sort_values("revenue", ascending=False)
        )

        if opportunity.empty:
            st.info("No opportunity summary data is available.")
        else:
            revenue_cutoff = opportunity["revenue"].quantile(0.75)

            def recommend_action(row):
                if row["revenue"] >= revenue_cutoff and row["avg_satisfaction"] >= 4:
                    return "Scale marketing and inventory"
                if row["revenue"] >= revenue_cutoff and row["avg_satisfaction"] < 4:
                    return "Protect CX before expanding"
                if row["avg_satisfaction"] >= 4:
                    return "Test targeted upsell campaign"
                return "Monitor and optimize locally"

            opportunity["recommended_action"] = opportunity.apply(recommend_action, axis=1)

            best_opportunity = opportunity.iloc[0]
            st.caption(
                f"Best immediate growth pocket: **{best_opportunity['customer_region']} / "
                f"{best_opportunity['category_group']}** with **{format_currency(best_opportunity['revenue'])}** "
                f"in revenue and **{best_opportunity['avg_satisfaction']:.2f}/5** satisfaction."
            )

            opp_display = opportunity.rename(
                columns={
                    "customer_region": "Region",
                    "category_group": "Category Group",
                    "revenue": "Revenue",
                    "customers": "Customers",
                    "avg_satisfaction": "Avg Satisfaction",
                    "orders": "Orders",
                    "recommended_action": "Recommended Action",
                }
            ).copy()

            opp_display["Revenue"] = opp_display["Revenue"].map(lambda x: f"${x:,.2f}")
            opp_display["Avg Satisfaction"] = opp_display["Avg Satisfaction"].map(
                lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
            )

            st.dataframe(
                opp_display[
                    [
                        "Region",
                        "Category Group",
                        "Revenue",
                        "Customers",
                        "Orders",
                        "Avg Satisfaction",
                        "Recommended Action",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

st.markdown("---")
st.caption(
    "Dashboard built for NovaRetail to support customer intelligence, revenue optimization, "
    "risk detection, and growth prioritization."
)
