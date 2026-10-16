
import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


# 1. Page configuration
st.set_page_config(
    page_title="SkillMatch AI",
    page_icon="🎯",
    layout="wide"
)


# 2. Locate the dataset
DATA_PATH = Path(__file__).parent / "data" / "job_roles.csv"


# 3. Load career data
@st.cache_data
def load_roles():
    return pd.read_csv(DATA_PATH)


# 4. Machine learning recommendation function
@st.cache_data(ttl=3600, max_entries=128, show_spinner=False)
def recommend_roles(user_skills, roles, top_n=6):
    if roles.empty:
        return pd.DataFrame()

    user_text = " ".join(user_skills).strip().lower()

    if not user_text:
        return pd.DataFrame()

    role_text = (
        roles["required_skills"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    documents = [user_text] + role_text
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        norm="l2"
    )

    try:
        vectors = vectorizer.fit_transform(documents)
        scores = (vectors[1:] @ vectors[0:1].T).toarray().ravel()
    except ValueError:
        scores = [0.0] * len(roles)

    results = roles.copy()
    results["match_score"] = (
        pd.Series(scores, index=results.index) * 100
    ).round(1)

    return results.sort_values(
        "match_score",
        ascending=False
    ).head(top_n)


# 5. Modern black dashboard styling
st.markdown("""
<style>
.stApp {
    background: #08090c;
    color: #f5f5f7;
}

.block-container {
    max-width: 1400px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

h1, h2, h3, h4, label {
    color: #f5f5f7;
}

.stApp [data-testid="stMarkdownContainer"] p {
    color: #c4c8d0;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #121419;
    border: 1px solid #262a33;
    border-radius: 18px;
}

.stApp input,
.stApp textarea {
    background: #191c23 !important;
    color: #ffffff !important;
    border-color: #343945 !important;
}

.stApp [data-baseweb="select"] > div {
    background: #191c23;
    color: #ffffff;
    border-color: #343945;
}

.stApp [data-baseweb="select"] span {
    color: #f5f5f7;
}

div.stButton > button {
    background: #f5f5f7;
    color: #08090c;
    border: 1px solid #f5f5f7;
    border-radius: 25px;
    min-height: 42px;
    font-weight: 700;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    background: #a9c7ff;
    border-color: #a9c7ff;
    color: #08090c;
}

div.stDownloadButton > button {
    background: #191c23;
    color: #ffffff;
    border: 1px solid #343945;
    border-radius: 25px;
}

hr {
    border-color: #292d36;
}

.stApp [data-testid="stCaptionContainer"] {
    color: #9299a8;
}

.stApp [data-testid="stExpander"] {
    background: #121419;
    border: 1px solid #262a33;
    border-radius: 12px;
}

.stAlert {
    background: #171b23;
    border: 1px solid #343945;
}

.stApp [data-testid="stToolbar"] {
    background: transparent;
}
</style>
""", unsafe_allow_html=True)


# 6. Load and validate the dataset
try:
    roles = load_roles()
except FileNotFoundError:
    st.error(
        "Dataset not found. Make sure job_roles.csv is inside "
        "the data folder next to app.py."
    )
    st.stop()
except Exception as error:
    st.error(f"Unable to load the dataset: {error}")
    st.stop()


# Normalize column names to avoid errors from accidental spaces
roles.columns = roles.columns.str.strip().str.lower()

required_columns = {
    "role",
    "required_skills",
    "summary",
    "learning_gaps"
}

missing_columns = required_columns - set(roles.columns)

if missing_columns:
    st.error(
        "Your job_roles.csv is missing these columns: "
        + ", ".join(sorted(missing_columns))
    )
    st.stop()

# Ensure text columns contain usable strings
for column in [
    "role",
    "required_skills",
    "summary",
    "learning_gaps"
]:
    roles[column] = roles[column].fillna("").astype(str).str.strip()

# Remove completely blank role records
roles = roles[roles["role"] != ""].copy()

if roles.empty:
    st.error("Your job_roles.csv does not contain any job roles.")
    st.stop()

# Build the skills list from the CSV
all_skills = sorted({
    skill.strip().lower()
    for cell in roles["required_skills"]
    for skill in cell.split(",")
    if skill.strip()
})


# 7. Top navigation
nav1, nav2 = st.columns([2, 3])

with nav1:
    st.markdown("## skillmatch AI")

with nav2:
    st.markdown(
        "<p style='text-align:right;padding-top:15px;'>"
        "Find Roles　 ·　 Skill Analysis　 ·　 Learning Roadmap"
        "</p>",
        unsafe_allow_html=True
    )

st.divider()


# 8. Welcome section
st.title("Find your next career match")

st.caption(
    "Discover roles aligned with your skills and find out "
    "what to learn next."
)


# 9. Search filters
with st.container(border=True):
    st.subheader("Search career opportunities")

    c1, c2, c3 = st.columns([2, 1, 1])

    with c1:
        keyword = st.text_input(
            "Job title or keyword",
            placeholder="e.g. Data Analyst"
        )

    with c2:
        category = st.selectbox(
            "Career area",
            [
                "All areas",
                "Data",
                "Web Development",
                "AI / ML",
                "Cybersecurity",
                "Database",
                "Design",
                "Testing",
                "Business"
            ]
        )

    with c3:
        top_n = st.selectbox(
            "Results",
            [3, 6, 8],
            index=1
        )


# 10. Two-column dashboard
left, right = st.columns([1, 2.2], gap="large")

with left:
    with st.container(border=True):
        st.subheader("Let AI match your skills")

        st.write(
            "Select your current skills to explore career options."
        )

        selected_skills = st.multiselect(
            "Your skills",
            options=all_skills,
            placeholder="Choose Python, SQL, HTML..."
        )

        custom_skills = st.text_input(
            "Add other skills",
            placeholder="e.g. Git, communication"
        )

        st.markdown("---")
        st.write("**Career preferences**")

        goal = st.selectbox(
            "Your goal",
            [
                "Explore careers",
                "Find a data role",
                "Build software",
                "Work in AI / ML",
                "Cybersecurity"
            ]
        )

        find_clicked = st.button(
            "Find my matches →",
            use_container_width=True
        )

        with st.expander("How matching works"):
            st.write(
                "TF-IDF converts skill text into numerical vectors. "
                "Cosine similarity compares your skills with each "
                "role's requirements. Scores are text-similarity "
                "scores, not hiring probabilities."
            )


# 11. Combine selected and custom skills
user_skills = list(selected_skills)

if custom_skills.strip():
    user_skills.extend([
        skill.strip().lower()
        for skill in custom_skills.split(",")
        if skill.strip()
    ])

# Remove duplicate skills
user_skills = list(dict.fromkeys(user_skills))


# 12. Career category mapping
category_map = {
    "Data": [
        "Data Analyst",
        "Business Analyst"
    ],
    "Web Development": [
        "Web Developer",
        "Python Developer"
    ],
    "AI / ML": [
        "Machine Learning Engineer"
    ],
    "Cybersecurity": [
        "Cybersecurity Analyst"
    ],
    "Database": [
        "Database Developer"
    ],
    "Design": [
        "UI UX Designer"
    ],
    "Testing": [
        "QA / Test Engineer"
    ],
    "Business": [
        "Business Analyst"
    ]
}


# 13. Recommended roles
with right:
    st.subheader("Recommended roles")

    if find_clicked and not user_skills:
        st.warning(
            "Please select at least one skill or enter a custom skill."
        )

    elif find_clicked:
        results = recommend_roles(
            user_skills,
            roles,
            top_n=len(roles)
        )

        st.caption(
            f"Dataset loaded: {len(roles)} roles | "
            f"Skills entered: {len(user_skills)}"
        )

        if keyword.strip():
            search_text = keyword.strip().lower()

            keyword_mask = (
                results["role"].str.lower().str.contains(
                    search_text,
                    regex=False,
                    na=False
                )
                | results["summary"].str.lower().str.contains(
                    search_text,
                    regex=False,
                    na=False
                )
                | results["required_skills"].str.lower().str.contains(
                    search_text,
                    regex=False,
                    na=False
                )
                | results["learning_gaps"].str.lower().str.contains(
                    search_text,
                    regex=False,
                    na=False
                )
            )

            results = results[keyword_mask]

        if category != "All areas":
            allowed_roles = category_map.get(category, [])

            results = results[
                results["role"].str.lower().isin(
                    [role_name.lower() for role_name in allowed_roles]
                )
            ]

        results = results.head(top_n)

        if results.empty:
            st.warning(
                "No roles match the current filters. "
                "Clear the keyword and select 'All areas' "
                "to test all available roles."
            )

            with st.expander("Debug information"):
                st.write("Available roles:", roles["role"].tolist())
                st.write("Skills entered:", user_skills)
                st.write("Keyword:", keyword)
                st.write("Career area:", category)

        else:
            st.caption(
                f"Showing {len(results)} role recommendations"
            )

            cards = st.columns(2, gap="medium")

            for i, (_, row) in enumerate(results.iterrows()):
                with cards[i % 2]:
                    with st.container(border=True):
                        st.markdown(
                            f"**{row['match_score']:.1f}% text match**"
                        )

                        st.markdown(f"### {row['role']}")

                        st.write(row["summary"])

                        st.caption("Required skills")
                        st.write(row["required_skills"])

                        st.caption("Suggested skills to learn")
                        st.write(row["learning_gaps"])

            st.subheader("Skill match overview")

            st.bar_chart(
                results.set_index("role")[["match_score"]]
            )

            st.download_button(
                "Download recommendations as CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="skillmatch_recommendations.csv",
                mime="text/csv"
            )

    else:
        with st.container(border=True):
            st.markdown("### Ready when you are")

            st.write(
                "Choose your skills on the left, then click "
                "'Find my matches' to see personalized recommendations."
            )


# 14. Footer
st.divider()

st.caption(
    "SkillMatch AI | AICTE Machine Learning Project | "
    "Educational prototype — similarity scores are not hiring predictions."
)
