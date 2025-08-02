
import pickle
import pandas as pd
import numpy as np
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from fuzzywuzzy import process

# Load data
df = pd.read_csv('TAKE1\chennai_reviews.csv')

# Drop unnecessary columns and NaN values
df.drop(['Sentiment', 'Unnamed: 1'], axis=1, inplace=True)
df.dropna(inplace=True)

# Compile pattern to split reviews
pattern = re.compile(r'[.,!?;:]')

# Function to split the reviews based on the pattern
def split_review(text):
    return pattern.split(text)

# Apply the function to create the new column
df['split_review'] = df['Review_Text'].apply(split_review)

# Explode the dataframe to separate rows for each split review
df_exploded = df.explode('split_review').reset_index(drop=True)

# Strip any leading or trailing whitespace from the split reviews
df_exploded['split_review'] = df_exploded['split_review'].str.strip()

# Replace empty strings with NaN
df_exploded['split_review'].replace('', np.nan, inplace=True)

# Drop rows where split_review is NaN
df_exploded.dropna(subset=['split_review'], inplace=True)

# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize VADER sentiment intensity analyzer
sid = SentimentIntensityAnalyzer()

# Function to classify sentiment
def classify_sentiment(text):
    scores = sid.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 1  # Positive
    elif scores['compound'] <= -0.05:
        return 0  # Negative
    else:
        return 0  # Neutral (treated as negative in this context)

# Apply the sentiment classification function
df_exploded['sentiment'] = df_exploded['split_review'].apply(classify_sentiment)


df =df_exploded

def check_keywords(text, keywords):
    text = text.lower()
    for word in keywords:
        if word in text:
            return 1
    return 0

staff_keywords = ['staff', 'service', 'receptionist', 'front desk', 'hospitality', 'courteous', 'friendly', 'helpful', 'professional', 'attentive', 'responsive', 'efficient', 'accommodating', 'pleasant', 'polite', 'welcoming', 'kind', 'warm', 'knowledgeable', 'smiling', 'team', 'guest', 'client', 'employee', 'worker', 'personnel', 'management', 'crew', 'waiter', 'waitress', 'manager', 'supervisor', 'concierge', 'bellboy', 'valet', 'maid', 'porter', 'bellhop', 'butler', 'host', 'hostess']
food_keywords = ['food', 'cuisine', 'restaurant', 'dining', 'menu', 'delicious', 'tasty', 'flavorful', 'yummy', 'culinary', 'gourmet', 'savory', 'mouthwatering', 'taste', 'dish', 'meal', 'buffet', 'breakfast', 'lunch', 'dinner', 'brunch', 'snack', 'beverage', 'drink', 'dessert', 'appetizer', 'main course', 'soup', 'salad', 'entree', 'grill', 'barbecue', 'bbq', 'bistro', 'cafe', 'pub', 'eatery', 'chefs', 'cook', 'waitstaff', 'pastry', 'bakery', 'baker', 'sommelier', 'mixologist', 'bartender']
room_keywords = ['room', 'accommodation', 'suite', 'bedroom', 'bed', 'pillow', 'mattress', 'linen', 'blanket', 'duvet', 'quilt', 'bedspread', 'comforter', 'towel', 'robe', 'slippers', 'furniture', 'decoration', 'interior', 'design', 'clean', 'tidy', 'spacious', 'cozy', 'luxurious', 'modern', 'chic', 'elegant', 'stylish', 'rustic', 'comfortable', 'airy', 'quiet', 'serene', 'relaxing', 'peaceful', 'private', 'secure', 'safe', 'cosy', 'neat', 'organised', 'well-kept']
spa_keywords = ['spa', 'massage', 'wellness', 'treatment', 'therapist', 'therapeutic', 'relaxation', 'aromatherapy', 'rejuvenation', 'pampering', 'detox', 'reflexology', 'facial', 'body', 'scrub', 'wraps', 'exfoliation', 'hydrotherapy', 'jacuzzi', 'sauna', 'steam room', 'vichy shower', 'mani-pedi', 'manicure', 'pedicure', 'nail', 'polish', 'salon', 'beauty', 'cosmetic', 'esthetician', 'skincare', 'holistic', 'wellbeing', 'zen', 'tranquil', 'serenity', 'oasis', 'calm', 'peace', 'meditation', 'yoga', 'fitness']
environment_keywords = ['environment', 'ambiance', 'atmosphere', 'surroundings', 'setting', 'location', 'scenery', 'landscape', 'view', 'outlook', 'vista', 'panorama', 'setting', 'locale', 'spot', 'situation', 'scene', 'surround']
# Define keywords for environment and wifi
environment_keywords += ['location', 'site', 'neighbourhood', 'area', 'region', 'district', 'place', 'vicinity', 'terrain', 'country', 'surroundings', 'countryside', 'rural', 'urban', 'cityscape', 'townscape', 'street', 'road', 'boulevard', 'avenue', 'alley', 'lane', 'path', 'walkway', 'park', 'garden', 'green', 'landscape', 'lawn', 'plaza', 'square', 'courtyard', 'patio', 'terrace', 'porch', 'deck', 'balcony', 'veranda', 'gazebo', 'pergola', 'arboretum', 'botanical garden', 'zoological garden', 'ocean', 'sea', 'beach', 'coast', 'shore', 'river', 'lake', 'pond', 'stream', 'creek', 'waterfront', 'waterbody', 'island', 'mountain', 'hill', 'valley', 'canyon', 'desert', 'forest', 'wood', 'woodland', 'jungle', 'rainforest', 'savanna', 'grassland', 'prairie', 'steppe', 'tundra', 'arctic', 'polar', 'icecap', 'glacier', 'volcano', 'geyser', 'cave', 'cavern', 'canyon', 'waterfall', 'rapids', 'hot spring', 'spring', 'thermal', 'grotto', 'lagoon', 'fjord', 'fiord', 'sound', 'bay', 'harbor', 'port', 'marina', 'dock', 'quay', 'pier', 'wharf', 'jetty', 'embankment', 'breakwater', 'seawall', 'lighthouse', 'buoy', 'beacon', 'anchor', 'mooring', 'berth', 'boat', 'vessel', 'ship', 'yacht', 'sailboat', 'cruise', 'ferry', 'raft', 'canoe', 'kayak', 'paddleboat', 'rowboat', 'catamaran', 'trimaran', 'tugboat', 'barge']
wifi_keywords = ['wifi', 'internet', 'connection', 'wireless', 'network', 'signal', 'bandwidth', 'speed', 'router', 'modem', 'hotspot', 'access point', 'ethernet', 'LAN', 'WLAN', 'SSID', 'password', 'login', 'connectivity', 'browsing', 'streaming', 'downloading', 'uploading', 'online', 'web', 'browser', 'surfing', 'website', 'page', 'online', 'cyberspace', 'virtual', 'digital', 'online', 'cloud', 'server', 'firewall', 'VPN', 'encryption', 'secure', 'private', 'public', 'open']



# # Apply the keyword checking functions
# df_exploded['staff'] = df_exploded['split_review'].apply(lambda x: check_keywords(x, staff_keywords))
# df_exploded['food'] = df_exploded['split_review'].apply(lambda x: check_keywords(x, food_keywords))
# df_exploded['room'] = df_exploded['split_review'].apply(lambda x: check_keywords(x, room_keywords))
# df_exploded['spa'] = df_exploded['split_review'].apply(lambda x: check_keywords(x, spa_keywords))

df['staff'] = df['Review_Text'].apply(lambda x: check_keywords(x, staff_keywords))
df['food'] = df['Review_Text'].apply(lambda x: check_keywords(x, food_keywords))
df['room'] = df['Review_Text'].apply(lambda x: check_keywords(x, room_keywords))
df['spa'] = df['Review_Text'].apply(lambda x: check_keywords(x, spa_keywords))
df['environment'] = df['Review_Text'].apply(lambda x: check_keywords(x, environment_keywords))
df['wifi'] = df['Review_Text'].apply(lambda x: check_keywords(x, wifi_keywords))


# creating a new dataframe

df2 = pd.DataFrame(columns=['Hotel_name', 'overall', 'staff', 'food', 'room', 'spa', 'env', 'wifi'])
df2['Hotel_name'] = df['Hotel_name'].unique()

def Overall(name):
    
    hotel_df = df[df['Hotel_name'] == name]
                  
    overall_percentage = (hotel_df['Rating_Percentage'].mean() / 10)              
    
    return overall_percentage


pd.options.display.float_format = '{:.2f}'.format

df2['overall'] = df2['Hotel_name'].apply(Overall)

def splitRate(type):
    percentage = (df[type].mean() * 10)
    return percentage

df2['staff']=splitRate('staff')
df2['food']=splitRate('food')
df2['room']=splitRate('room')
df2['spa']=splitRate('spa')
df2['env']=splitRate('environment')
df2['wifi']=splitRate('wifi')

file = "reviews.pkl"
fileob = open(file, "wb")
pickle.dump(df2, fileob)

fileob.close()