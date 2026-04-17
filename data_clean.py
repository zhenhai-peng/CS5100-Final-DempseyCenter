import pandas as pd
import numpy as np
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
FILES = {
    2022: '2022 Annual Client Survey Data.xlsx',
    2023: '2023 Annual Client Survey Data.xlsx',
    2024: '2024 Annual Client Survey.xlsx',
    2025: 'Dempsey Center 2025 Annual Client Survey.xlsx',
}

def normalize_quotes(s):
    """Replace curly/smart quotes with straight quotes."""
    return s.replace('\u2019', "'").replace('\u2018', "'").replace('\u201c', '"').replace('\u201d', '"')

# ============================================================
# SERVICE NORMALIZATION
# ============================================================
SERVICE_NORMALIZE = {
    'Massage Therapy': 'Massage Therapy',
    'Massage': 'Massage Therapy',
    'Massage Therapy (provided by licensed massage therapists trained to support individuals affected by cancer through gentle, therapeutic techniques)': 'Massage Therapy',
    'Massage Therapy (provided by licensed massage therapists trained to support individuals affected by cancer through gentle therapeutic techniques)': 'Massage Therapy',
    'Acupuncture': 'Acupuncture',
    'Acupuncture Session': 'Acupuncture',
    'Acupuncture (provided by licensed acupuncturists)': 'Acupuncture',
    'Reiki': 'Reiki',
    'Reiki Session': 'Reiki',
    'Reiki Session (a gentle, non-invasive energy-based practice)': 'Reiki',
    'Individual Counseling': 'Individual Counseling',
    'Adult Counseling - Individual': 'Individual Counseling',
    'Individual Counseling (one-on-one sessions with licensed counselors)': 'Individual Counseling',
    'Support Groups': 'Support Groups',
    'Adult Counseling - Support Groups': 'Support Groups',
    'Support Groups (connect with others who share similar experiences)': 'Support Groups',
    'Family Counseling': 'Family Counseling',
    'Adult Counseling - Couples/Family': 'Family Counseling',
    'Couples/Family Counseling': 'Family Counseling',
    'Child and Adolescent Counseling': 'Child/Adolescent Counseling',
    'Child/Adolescent Counseling': 'Child/Adolescent Counseling',
    'Movement & Fitness Classes/Workshops': 'Movement & Fitness',
    'Movement & Fitness Classes': 'Movement & Fitness',
    'Movement & Fitness Classes/Workshops (e.g., Chair Yoga, Tai Chi, Strength Training, Walking Group, etc.)': 'Movement & Fitness',
    'Nutrition Consult': 'Nutrition Consult',
    'Nutrition Consultation': 'Nutrition Consult',
    'Nutrition Consultation (one-on-one with a registered dietitian)': 'Nutrition Consult',
    'Nutrition Classes/Workshops': 'Nutrition Classes',
    'Nutrition Class': 'Nutrition Classes',
    'Nutrition Classes/Workshops (e.g., Cooking for Wellness, etc.)': 'Nutrition Classes',
    'Complementary Therapies Workshops': 'Complementary Therapy Workshops',
    'Complementary Therapies Workshops (e.g., Virtual Reiki Group, Mindfulness Meditation or Acupressure for Calm, etc.)': 'Complementary Therapy Workshops',
    'Complementary Therapies Workshops (e.g., Mindfulness Meditation, Acupressure for Calm, Sound Bath, etc.)': 'Complementary Therapy Workshops',
    'Educational Workshops': 'Educational Workshops',
    'Educational Workshop (e.g., Communicating with Your Healthcare Team, Fatigue Factor, Managing Cancer Side Effects with Cannabis/CBD, etc.)': 'Educational Workshops',
    'Educational Workshops (e.g., Look Good Feel Better, Managing Side Effects with Cannabis/CBD, Communicating with your Healthcare Team, etc.)': 'Educational Workshops',
    'Dempsey Dogs': 'Dempsey Dogs',
    'Dempsey Dogs (e.g., interacting with a therapy dog in the Center, or at a Dempsey Center sponsored event)': 'Dempsey Dogs',
    'Comfort Items': 'Comfort Items',
    'Comfort Items (e.g., quilts, blankets, hats, port protectors, heart pillows, mastectomy supplies, etc.)': 'Comfort Items',
    'Art Therapy': 'Art Therapy',
    'Creative Arts': 'Art Therapy',
    'Creative Arts Workshops': 'Art Therapy',
    'Qigong': 'Qigong',
    'Wig Program': 'Wig Program',
    'Wig Lending Library': 'Wig Program',
    'Wellness Coaching': 'Wellness Coaching',
    'Financial Navigation': 'Financial Navigation',
    'Financial Assistance': 'Financial Navigation',
    'Cancer Navigation': 'Cancer Navigation',
    'Patient Navigation': 'Cancer Navigation',
    'Retreat': 'Retreat',
    'Annual Retreat': 'Retreat',
    'Special Events': 'Special Events',
    'Virtual Services': 'Virtual Services',
    'Telehealth Services': 'Virtual Services',
    'Grief Support': 'Grief Support',
    'Bereavement Support': 'Grief Support',
    'Occupational Therapy': 'Occupational Therapy',
    'Physical Therapy': 'Physical Therapy',
    "Clayton's House Stay": "Clayton's House Stay",
    "Clayton's House stay": "Clayton's House Stay",
    "Clayton's House Stay Program": "Clayton's House Stay",
}

# Build case-insensitive + quote-normalized lookup
SERVICE_NORM_LOWER = {
    normalize_quotes(k).lower(): v for k, v in SERVICE_NORMALIZE.items()
}

# ============================================================
# BARRIER NORMALIZATION
# ============================================================
BARRIER_NORMALIZE = {
    'Times of services': 'Inconvenient service times',
    "Times of services (the service I was interested in wasn't scheduled at a time that worked for me)": 'Inconvenient service times',
    'Wait time to access services that I needed (when I needed them)': 'Long wait for services',
    'Too long of a wait for services that I needed (when I needed them)': 'Long wait for services',
    'Too long of a wait to become a client (too long of a wait for an orientation)': 'Long wait for orientation',
    'Too long of a wait to become a client': 'Long wait for orientation',
    'Physical location of services': 'Physical location',
    'Transportation': 'Transportation',
    'Hours of operation': 'Hours of operation',
    'Reliable internet access or lack of reliable hardware': 'Internet/hardware access',
    'Reliable internet access': 'Internet/hardware access',
    'Other': 'Other barrier',
    'Other (please specify)': 'Other barrier',
    'I did not feel welcomed and/or comfortable at the Dempsey Center': 'Unwelcoming environment',
    "Accessibility of the Center's services a lack of comfort and or ability with being able to access services": 'Accessibility issues',
}

BARRIER_NORM_LOWER = {
    normalize_quotes(k).lower(): v for k, v in BARRIER_NORMALIZE.items()
}

# ============================================================
# OUTCOME PREFIXES
# ============================================================
OUTCOME_PREFIXES = [
    'reduced', 'helped', 'improved', 'created', 'provided',
    'increased', 'strengthened', 'gave me', 'allowed me',
    'enhanced', 'supported my', 'enabled',
]


def classify_item(item):
    """Classify a raw multi-select item as service or impact."""
    s = item.strip()
    sl = normalize_quotes(s).lower()

    # Exact match (case-insensitive, quote-normalized)
    if sl in SERVICE_NORM_LOWER:
        return ('service', SERVICE_NORM_LOWER[sl])

    # Outcome prefix check
    for prefix in OUTCOME_PREFIXES:
        if sl.startswith(prefix):
            return ('impact', s)

    # Fuzzy substring
    for raw_lower, canonical in SERVICE_NORM_LOWER.items():
        if len(raw_lower) > 10 and raw_lower in sl:
            return ('service', canonical)

    # Length heuristic
    if len(s) < 50:
        return ('service', s)
    else:
        return ('impact', s)


def normalize_barrier(b):
    """Normalize a barrier string."""
    bl = normalize_quotes(b.strip()).lower()
    if bl in BARRIER_NORM_LOWER:
        return BARRIER_NORM_LOWER[bl]
    for raw_lower, canonical in BARRIER_NORM_LOWER.items():
        if len(raw_lower) > 10 and raw_lower in bl:
            return canonical
    return b.strip()


def make_safe_col(prefix, name):
    """Generate a safe, lowercase one-hot column name."""
    safe = name.lower().replace(' ', '_').replace('/', '_').replace("'", '') \
               .replace(',', '').replace('(', '').replace(')', '').replace('.', '') \
               .replace('&', 'and').replace('-', '_').replace(';', '')
    return f'{prefix}_{safe}'


# ============================================================
# 1. Load raw data
# ============================================================
raw = {}
for year, fp in FILES.items():
    df = pd.read_excel(fp)
    if df.iloc[0].astype(str).str.contains('Response|Open-Ended|None of the above').any():
        df = df.iloc[1:].reset_index(drop=True)
    raw[year] = df
    print(f"[{year}] loaded: {df.shape[0]} rows x {df.shape[1]} cols")

# ============================================================
# 2. Utility functions
# ============================================================
def find_col(df, keywords, exclude=None):
    for c in df.columns:
        cl = c.lower()
        if all(k in cl for k in keywords):
            if exclude and any(e in cl for e in exclude):
                continue
            return c
    return None

def extract_block_between(df, start_col, end_col):
    cols = df.columns.tolist()
    start_idx = cols.index(start_col)
    end_idx = cols.index(end_col) if end_col else len(cols)
    subset = df.iloc[:, start_idx:end_idx]
    result = []
    for _, row in subset.iterrows():
        items = [str(v).strip() for v in row.dropna().values
                 if str(v).strip() not in ('nan', 'None of the above', 'Response', 'None', '')]
        result.append(items)
    return result

def extract_block_from(df, start_col):
    cols = df.columns.tolist()
    idx = cols.index(start_col) + 1
    while idx < len(cols) and str(cols[idx]).startswith('Unnamed'):
        idx += 1
    end_col = cols[idx] if idx < len(cols) else None
    return extract_block_between(df, start_col, end_col)

def get_block(df, start_keywords, end_keywords=None):
    start_col = find_col(df, start_keywords)
    if start_col is None:
        return [[] for _ in range(len(df))]
    if end_keywords:
        end_col = find_col(df, end_keywords)
        if end_col is None:
            return extract_block_from(df, start_col)
        return extract_block_between(df, start_col, end_col)
    else:
        return extract_block_from(df, start_col)

def get_all_service_items(df, year):
    n = len(df)
    raw_items_all = [[] for _ in range(n)]

    if year <= 2023:
        block = get_block(df, ['services', 'used'], ['how often'])
        for i in range(n):
            raw_items_all[i].extend(block[i])
    else:
        block1 = get_block(df, ['ways', 'dempsey', 'better'], ['comfort'])
        block2 = get_block(df, ['comfort', 'wellbeing'], ['integrative'])
        block3 = get_block(df, ['integrative', 'complementary'], ['counseling'])
        block4 = get_block(df, ['counseling', 'services'], ['how often'])
        for i in range(n):
            raw_items_all[i].extend(block1[i])
            raw_items_all[i].extend(block2[i])
            raw_items_all[i].extend(block3[i])
            raw_items_all[i].extend(block4[i])

    services_out = [[] for _ in range(n)]
    impacts_out = [[] for _ in range(n)]
    for i in range(n):
        seen = set()
        for item in raw_items_all[i]:
            kind, value = classify_item(item)
            if kind == 'service':
                if value not in seen:
                    services_out[i].append(value)
                    seen.add(value)
            else:
                impacts_out[i].append(value)

    return services_out, impacts_out

# ============================================================
# 3. Extract standardized fields
# ============================================================
records = []
for year, df in raw.items():
    sat_col       = find_col(df, ['satisfaction'])
    life_col      = find_col(df, ['make your life better'])
    nps_col       = find_col(df, ['recommend'])
    freq_col      = find_col(df, ['how often'])
    cancer_col    = find_col(df, ['best describes', 'cancer impact'])
    treatment_col = find_col(df, ['treatment plan'])
    age_col       = find_col(df, ['age'])
    zip_col       = find_col(df, ['zip'])
    gender_col    = find_col(df, ['gender'])
    income_col    = find_col(df, ['income'])
    feedback_col  = (find_col(df, ['other feedback']) or
                     find_col(df, ['additional feedback']) or
                     find_col(df, ['how could we be better']))

    services_list, impacts_list = get_all_service_items(df, year)

    raw_barriers = get_block(df, ['negatively', 'affect', 'ability'])
    barriers_norm = []
    for row_bars in raw_barriers:
        normed = list(dict.fromkeys([normalize_barrier(b) for b in row_bars]))
        barriers_norm.append(normed)

    locations = get_block(df, ['where', 'receive', 'services'])

    for i in range(len(df)):
        row = df.iloc[i]
        nps_val = row[nps_col] if nps_col else np.nan
        try:
            nps_val = int(float(nps_val))
        except:
            nps_val = np.nan

        records.append({
            'year':             year,
            'satisfaction':     row[sat_col] if sat_col else np.nan,
            'life_better':      row[life_col] if life_col else np.nan,
            'nps_score':        nps_val,
            'frequency':        row[freq_col] if freq_col else np.nan,
            'cancer_status':    row[cancer_col] if cancer_col else np.nan,
            'treatment_stage':  row[treatment_col] if treatment_col else np.nan,
            'age':              row[age_col] if age_col else np.nan,
            'zip_code':         row[zip_col] if zip_col else np.nan,
            'gender':           row[gender_col] if gender_col else np.nan,
            'income':           row[income_col] if income_col else np.nan,
            'services_used':    services_list[i],
            'num_services':     len(services_list[i]),
            'impacts_reported': impacts_list[i],
            'num_impacts':      len(impacts_list[i]),
            'barriers':         barriers_norm[i],
            'num_barriers':     len(barriers_norm[i]),
            'locations':        locations[i],
            'feedback_text':    row[feedback_col] if feedback_col else np.nan,
        })

combined = pd.DataFrame(records)

# ============================================================
# 4. Core encoding
# ============================================================
sat_map = {
    'Very Satisfied': 5, 'Satisfied': 4,
    'Neither Satisfied nor Unsatisfied': 3,
    'Unsatisfied': 2, 'Very Unsatisfied': 1,
}
combined['satisfaction_score'] = combined['satisfaction'].map(sat_map)
combined['life_better_binary'] = combined['life_better'].map({'Yes': 1, 'No': 0})
combined['nps_category'] = pd.cut(
    combined['nps_score'], bins=[-1, 6, 8, 10],
    labels=['Detractor', 'Passive', 'Promoter']
)
freq_map = {'1 time': 1, '2-5 times': 2, '6-9 times': 3, 'More than 10 times': 4}
combined['freq_numeric'] = combined['frequency'].map(freq_map)
combined = combined[combined['satisfaction'].isin(sat_map.keys())].reset_index(drop=True)
print(f"\nValid rows after satisfaction filter: {len(combined)}")

# ============================================================
# 5. Quality fixes
# ============================================================

# 5a. Duplicates
dup_cols = [c for c in combined.columns
            if c not in ('services_used', 'barriers', 'locations', 'impacts_reported')]
n_before = len(combined)
combined = combined.drop_duplicates(subset=dup_cols).reset_index(drop=True)
print(f"[Fix 1] Duplicates removed: {n_before - len(combined)}")

# 5b. Gender (includes all synonym merges)
gender_map = {
    'Woman / Female / Feminine': 'Female',
    'Woman': 'Female',
    'Man / Male / Masculine': 'Male',
    'Man': 'Male',
    'Non-binary': 'Non-binary',
    'Non-Binary': 'Non-binary',
    'Transgender': 'Transgender',
    'Prefer not to answer': 'Prefer not to answer',
    'Prefer not to say': 'Prefer not to answer',
    'Prefer to self-describe:': 'Other/Self-describe',
}
combined['gender'] = combined['gender'].map(gender_map).fillna(combined['gender'])
print(f"[Fix 2] Gender normalized")

# 5c. Age
bad_age = combined['age'].astype(str).str.contains('Stage', case=False, na=False)
combined.loc[bad_age, 'age'] = np.nan
print(f"[Fix 3] Age anomalies removed: {bad_age.sum()}")

age_vals = combined['age'].dropna().astype(str)
if age_vals.str.contains('-').any() or age_vals.str.contains('Under|Over|above|below', case=False).any():
    combined['age_group'] = combined['age'].astype(str).replace('nan', np.nan)
    print(f"  Age is categorical, kept as age_group")
else:
    combined['age_numeric'] = pd.to_numeric(combined['age'], errors='coerce')
    bins = [0, 30, 40, 50, 60, 70, 120]
    labels = ['Under 30', '30-39', '40-49', '50-59', '60-69', '70+']
    combined['age_group'] = pd.cut(combined['age_numeric'], bins=bins, labels=labels)
    combined.drop(columns=['age_numeric'], inplace=True)
    print(f"  Age binned into groups")

# 5d. Cancer status
combined['cancer_status'] = combined['cancer_status'].replace(
    'Other / prefer to self-describe:', 'Other/prefer to self-describe:'
)
print(f"[Fix 4] Cancer status unified")

# 5e. Zip to region
def zip_to_region(z):
    try:
        prefix = int(str(int(float(z))).zfill(5)[:3])
    except:
        return np.nan
    if prefix <= 199:   return 'Northeast'
    elif prefix <= 299: return 'Mid-Atlantic'
    elif prefix <= 399: return 'Southeast'
    elif prefix <= 499: return 'Midwest'
    elif prefix <= 599: return 'Upper Midwest'
    elif prefix <= 699: return 'Central'
    elif prefix <= 799: return 'South'
    elif prefix <= 899: return 'Mountain West'
    elif prefix <= 999: return 'Pacific'
    return np.nan

combined['zip_str'] = combined['zip_code'].apply(
    lambda x: str(int(float(x))).zfill(5) if pd.notna(x) and str(x) not in ('nan', '') else np.nan
)
combined['region'] = combined['zip_code'].apply(zip_to_region)
print(f"[Fix 5] Zip mapped to region")

# 5f. Missing values
combined['nps_score_filled'] = combined['nps_score'].fillna(combined['nps_score'].median())
combined['freq_numeric_filled'] = combined['freq_numeric'].fillna(combined['freq_numeric'].median())
combined['cancer_status_filled'] = combined['cancer_status'].fillna('Unknown')
combined['age_group_filled'] = combined['age_group'].astype(str).replace('nan', 'Unknown')
combined['gender_filled'] = combined['gender'].fillna('Unknown')
combined['income'] = combined['income'].fillna('Prefer not to answer')
print(f"[Fix 6] Missing values handled")

# 5g. One-hot: services (all column names lowercase via make_safe_col)
svc_counter = Counter()
for svcs in combined['services_used']:
    if isinstance(svcs, list):
        svc_counter.update(svcs)
top_svc = [s for s, _ in svc_counter.most_common(20)]

for svc in top_svc:
    col = make_safe_col('svc', svc)
    combined[col] = combined['services_used'].apply(
        lambda x, s=svc: 1 if isinstance(x, list) and s in x else 0
    )

# 5h. One-hot: barriers (normalized, lowercase column names)
bar_counter = Counter()
for bars in combined['barriers']:
    if isinstance(bars, list):
        bar_counter.update(bars)
top_bar = [b for b, _ in bar_counter.most_common(10)]

for bar in top_bar:
    col = make_safe_col('bar', bar)
    combined[col] = combined['barriers'].apply(
        lambda x, b=bar: 1 if isinstance(x, list) and b in x else 0
    )

# 5i. One-hot: gender, cancer_status, frequency
gender_dummies = pd.get_dummies(combined['gender_filled'], prefix='gen').astype(int)
cancer_dummies = pd.get_dummies(combined['cancer_status_filled'], prefix='cancer').astype(int)
freq_dummies = pd.get_dummies(combined['frequency'].fillna('Unknown'), prefix='freq').astype(int)

# Lowercase all dummy column names
gender_dummies.columns = [c.lower().replace(' ', '_') for c in gender_dummies.columns]
cancer_dummies.columns = [c.lower().replace(' ', '_').replace('/', '_') for c in cancer_dummies.columns]
freq_dummies.columns = [c.lower().replace(' ', '_') for c in freq_dummies.columns]

combined = pd.concat([combined, gender_dummies, cancer_dummies, freq_dummies], axis=1)

svc_cols = [c for c in combined.columns if c.startswith('svc_')]
bar_cols = [c for c in combined.columns if c.startswith('bar_')]
gen_cols = [c for c in combined.columns if c.startswith('gen_')]
cancer_oh = [c for c in combined.columns if c.startswith('cancer_')]
freq_oh = [c for c in combined.columns if c.startswith('freq_')]
print(f"[Fix 7] One-hot: {len(svc_cols)} svc, {len(bar_cols)} bar, "
      f"{len(gen_cols)} gen, {len(cancer_oh)} cancer, {len(freq_oh)} freq")

# ============================================================
# 6. Save
# ============================================================
combined.to_pickle('cleaned_data.pkl')
combined.to_csv('cleaned_data.csv', index=False)