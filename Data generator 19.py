import numpy as np
import pandas as pd
import random

#########################
# Configuration
#########################
BASE_PATTERNS = 900  # We'll create 900 base individuals
                     # Each repeated 1..4 times => we might get up to ~3600 rows
                     # We'll trim back to exactly 2000 in the end.

# 6 final gifts (no merges)
FINAL_GIFTS = [
    "Chocolates",
    "Flowers",
    "Tech Gadget",
    "Fashion Accessory",
    "Personalized Gift",
    "Subscription Service"
]
REACTIONS = ["Loved it","Liked it","Disliked it"]

genders = ["Male","Female","Other"]
relationship_statuses = ["Single","Dating","Married","Engaged"]
budgets = ["Low","Medium","High","Very High"]
occasions = ["Valentine's Day","Birthday","Anniversary","Just Because"]
preferences = ["Romantic","Sentimental","Practical","Adventurous","Surprise"]
relationship_lengths = ["<6 months","6-12 months","1-3 years","3-5 years","5+ years"]
personal_interests = ["Fashion","Technology","Cooking","Sports","Music","Art"]

# 5 categories x 4 items each => 20 possible recent purchases
RECENT_PURCHASE_CATS = {
    0: ["Chocolate Bar","Cookies","Cupcakes","Candy"],         
    1: ["Roses","Daisies","Potted Plant","Sunflowers"],        
    2: ["Processor Chip","Headphones","Phone Charger","Smart Speaker"],
    3: ["Meal Kit Sub","Gym Membership","Magazine Sub","Streaming Sub"],
    4: ["Earrings","Necklace","Hat","Handbag"]
}

#########################
# 2. Generate a Base Individual
#########################
def generate_base_individual(idx):
    """
    Partially deterministic, partially random approach
    to create features for each base pattern.
    """
    # We’ll combine stable seeds + some randomness for variety.
    random.seed(idx + 42)
    
    gender = random.choice(genders)
    age = random.randint(20, 60)
    r_status = random.choice(relationship_statuses)
    budget = random.choice(budgets)
    occasion = random.choice(occasions)
    pref = random.choice(preferences)
    r_length = random.choice(relationship_lengths)
    interest = random.choice(personal_interests)
    
    # Past gift is one of the 6 final gifts
    past_item = random.choice(FINAL_GIFTS)
    
    # Reaction logic: if user is "Romantic"/"Sentimental" and the gift is "Chocolates"/"Flowers" => Loved it
    # Otherwise random distribution among [Loved it, Liked it, Disliked it]
    if pref in ["Romantic","Sentimental"] and past_item in ["Chocolates","Flowers"]:
        reaction = "Loved it"
    else:
        reaction = random.choices(REACTIONS, weights=[0.4,0.3,0.3], k=1)[0]
    
    # 3 recent purchases => pick category from personal_interest map
    interest_map = {
        "Fashion": 4,
        "Technology": 2,
        "Cooking": 3,
        "Sports": 2,
        "Music": 3,
        "Art": 4
    }
    cat_idx = interest_map[interest]
    cat_items = RECENT_PURCHASE_CATS[cat_idx]
    
    rp1 = cat_items[(idx + 0) % 4]
    rp2 = cat_items[(idx + 1) % 4]
    rp3 = cat_items[(idx + 2) % 4]
    
    base = {
        "Gender": gender,
        "Age": age,
        "Relationship_Status": r_status,
        "Budget": budget,
        "Occasion": occasion,
        "Preference": pref,
        "Relationship_Length": r_length,
        "Personal_Interest": interest,
        "Past_Gift_Item": past_item,
        "Past_Gift_Reaction": reaction,
        "Recent_Purchase_1": rp1,
        "Recent_Purchase_2": rp2,
        "Recent_Purchase_3": rp3
    }
    return base

#########################
# 3. Deterministic Best Gift Logic
#########################

def pick_best_gift(row):
    """
    Similar priority-based logic:
      1) budget=Very High => Personalized Gift
      2) gender=Other => Subscription Service
      3) gender=Male & interest=Technology => Tech Gadget
      4) preference=Romantic => Flowers
      5) preference=Practical => Chocolates
      6) fallback => Fashion Accessory
    Then skip if it's Past_Gift_Item 90% of the time.
    """
    b = row["Budget"]
    g = row["Gender"]
    interest = row["Personal_Interest"]
    pref = row["Preference"]
    past = row["Past_Gift_Item"]
    
    if b == "Very High":
        initial = "Personalized Gift"
    elif g == "Other":
        initial = "Subscription Service"
    elif g == "Male" and interest == "Technology":
        initial = "Tech Gadget"
    elif pref == "Romantic":
        initial = "Flowers"
    elif pref == "Practical":
        initial = "Chocolates"
    else:
        initial = "Fashion Accessory"
    
    # No re-purchase 90% of the time
    if initial == past:
        if random.random() < 0.9:
            idx = FINAL_GIFTS.index(initial)
            final = FINAL_GIFTS[(idx + 1) % len(FINAL_GIFTS)]
        else:
            final = initial
    else:
        final = initial
    
    return final

#########################
# 4. Create Base Patterns and Repeat 1..4 Times
#########################

def create_dataset():
    random.seed(1234)  # control how we replicate
    base_individuals = []
    
    for i in range(BASE_PATTERNS):
        person_base = generate_base_individual(i)
        base_individuals.append(person_base)
    
    all_rows = []
    
    for i, base_person in enumerate(base_individuals):
        repeat_count = random.randint(1, 4)  # Repeats in [1..4]
        for _ in range(repeat_count):
            row_data = base_person.copy()
            row_data["Best_Gift"] = pick_best_gift(row_data)
            all_rows.append(row_data)
    
    # Now we might have anywhere from 900..3600 rows.
    # We want to ensure we exceed 2000, so let's do a quick check:
    if len(all_rows) < 2000:
        # If by chance the repeats were mostly 1, we might be under 2000
        # We can just keep replicating the last chunk. 
        needed = 2000 - len(all_rows)
        chunk = all_rows[-100:]  # take last 100 as a sample
        while needed > 0:
            for row in chunk:
                all_rows.append(row.copy())
                needed -= 1
                if needed <= 0:
                    break
    
    # If we overshoot, we'll trim
    if len(all_rows) > 2000:
        all_rows = all_rows[:2000]
    
    # 5. Shuffle so duplicates are not consecutive
    random.shuffle(all_rows)
    
    return pd.DataFrame(all_rows)

df = create_dataset()
df.to_csv("valentine_gift.csv", index=False)
print(f"Created 'valentine_gift.csv' with {len(df)} rows (1..4 repeats, no consecutive duplicates).")