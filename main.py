from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List
import sqlite3
import os
import re
import random
import numpy as np

app = FastAPI(title="Travel Buddy ML API", debug=True)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
def init_db():
    conn = sqlite3.connect('travel_chat.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_id TEXT NOT NULL,
            receiver_id TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

class Profile(BaseModel):
    destination: str
    travel_style: str
    hobbies: str
    filter_type: str = "all"
    travel_date: str = None

class ChatMessage(BaseModel):
    message: str
    user_id: str
    receiver_id: str

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]

    async def send_personal_message(self, message: str, user_id: str):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_text(message)

manager = ConnectionManager()

# Database functions
def save_message(sender_id: str, receiver_id: str, message: str):
    conn = sqlite3.connect('travel_chat.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO chat_messages (sender_id, receiver_id, message) VALUES (?, ?, ?)', 
                   (sender_id, receiver_id, message))
    conn.commit()
    conn.close()

def get_chat_history(user1_id: str, user2_id: str):
    conn = sqlite3.connect('travel_chat.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT sender_id, receiver_id, message, timestamp 
        FROM chat_messages 
        WHERE (sender_id = ? AND receiver_id = ?) OR (sender_id = ? AND receiver_id = ?)
        ORDER BY timestamp ASC
    ''', (user1_id, user2_id, user2_id, user1_id))
    messages = cursor.fetchall()
    conn.close()
    return [{"sender_id": msg[0], "receiver_id": msg[1], "message": msg[2], "timestamp": msg[3]} for msg in messages]

@app.get("/")
def home():
    return {"message": "Travel Buddy ML API"}

def clean_destination(destination):
    if pd.isna(destination):
        return ""
    
    destination = str(destination).lower()
    
    # Remove country suffixes and common prefixes
    destination = re.sub(r',\s*(india|usa|uk|united states|united kingdom|canada|australia|japan|china|france|germany|italy|spain|morocco|scotland|indonesia|thailand|brazil|uae|netherlands|mexico|south africa|south korea|new zealand|vietnam|korean|taiwan|emirati|dutch|spanish|italian|greek|portuguese)$', '', destination)
    
    # Remove common prefixes
    destination = re.sub(r'^(traveling to|going to|visit|visiting)\s+', '', destination)
    
    # Clean up extra spaces and strip
    destination = re.sub(r'\s+', ' ', destination).strip()
    
    return destination

def load_real_dataset():
    """Load and clean the real dataset"""
    try:
        # Create a DataFrame from the provided data
        data = []
        for line in """Trip ID,Destination,Start date,End date,Duration (days),Traveler name,Traveler age,Traveler gender,Traveler nationality,Accommodation type,Accommodation cost,Transportation type,Transportation cost
1,"London, UK",5/1/2023,5/8/2023,7,John Smith,35,Male,American,Hotel,1200,Flight,600
2,"Phuket, Thailand",6/15/2023,6/20/2023,5,Jane Doe,28,Female,Canadian,Resort,800,Flight,500
3,"Bali, Indonesia",7/1/2023,7/8/2023,7,David Lee,45,Male,Korean,Villa,1000,Flight,700
4,"New York, USA",8/15/2023,8/29/2023,14,Sarah Johnson,29,Female,British,Hotel,2000,Flight,1000
5,"Tokyo, Japan",9/10/2023,9/17/2023,7,Kim Nguyen,26,Female,Vietnamese,Airbnb,700,Train,200
6,"Paris, France",10/5/2023,10/10/2023,5,Michael Brown,42,Male,American,Hotel,1500,Flight,800
7,"Sydney, Australia",11/20/2023,11/30/2023,10,Emily Davis,33,Female,Australian,Hostel,500,Flight,1200
8,"Rio de Janeiro, Brazil",1/5/2024,1/12/2024,7,Lucas Santos,25,Male,Brazilian,Airbnb,900,Flight,600
9,"Amsterdam, Netherlands",2/14/2024,2/21/2024,7,Laura Janssen,31,Female,Dutch,Hotel,1200,Train,200
10,"Dubai, United Arab Emirates",3/10/2024,3/17/2024,7,Mohammed Ali,39,Male,Emirati,Resort,2500,Flight,800
15,"Marrakech, Morocco",8/20/2024,8/27/2024,7,Fatima Khouri,26,Female,Moroccan,Riad,600,Flight,400
16,"Edinburgh, Scotland",9/5/2024,9/12/2024,7,James MacKenzie,32,Male,Scottish,Hotel,900,Train,150
18,Bali,8/15/2023,8/25/2023,10,Michael Chang,28,Male,Chinese,Resort,"$1,500 ",Plane,$700 
39,"Paris, France",6/12/2022,6/19/2022,7,Mia Johnson,25,Female,American,Hotel,1400,Plane,600
66,Bali,2/1/2024,2/8/2024,7,Tom Wilson,27,Male,American,Resort,2200,Plane,1000
73,"Bali, Indonesia",8/5/2022,8/12/2022,7,Sarah Lee,35,Female,South Korean,Resort,500 USD,Plane,800 USD
78,"Bali, Indonesia",11/12/2023,11/19/2023,7,Amanda Chen,25,Female,Taiwanese,Resort,600 USD,Plane,700 USD
93,Bali,4/15/2022,4/25/2022,11,Putra Wijaya,33,Male,Indonesian,Villa,1500 USD,Car rental,300 USD
104,Bali,7/22/2024,7/28/2024,6,Olivia Kim,29,Female,South Korea,Villa,"$1,200 ",Plane,"$1,000 "
109,Bali,8/12/2022,8/20/2022,8,Lisa Chen,30,Female,Taiwan,Resort,1500,Plane,1200
121,"Bali, Indonesia",7/20/2022,7/30/2022,10,Emily Kim,29,Female,Korean,Hostel,500,Plane,800
127,"Bali, Indonesia",2/10/2023,2/18/2023,8,Katie Johnson,33,Female,Canadian,Hotel,800,Plane,800""".split('\n')[1:]:
            if line.strip():
                parts = line.split(',')
                if len(parts) >= 12:
                    data.append(parts)
        
        df = pd.DataFrame(data[1:], columns=data[0])
        
        # Clean the data
        df = df.fillna("")
        
        # Add missing columns for compatibility
        df["Travel style"] = "Adventurer"  # Default travel style
        df["Interests"] = "Travel, Culture, Food"  # Default interests
        
        print(f"📊 Loaded {len(df)} real profiles from dataset")
        return df
        
    except Exception as e:
        print(f"❌ Error loading real dataset: {e}")
        return pd.DataFrame()

def fix_compatibility_scores(matches):
    """Ensure all compatibility scores are between 0-100%"""
    fixed_matches = []
    for match in matches:
        compatibility = match.get("compatibility", 50)
        # Ensure it's an integer between 0-100
        compatibility = max(0, min(100, int(compatibility)))
        match["compatibility"] = compatibility
        fixed_matches.append(match)
    return fixed_matches

@app.post("/match")
def match_profiles(profile: Profile):
    try:
        # Load datasets
        real_df = load_real_dataset()
        indian_profiles = generate_indian_profiles()
        international_profiles = generate_international_profiles()
        demo_profiles = generate_demo_profiles()
        
        # Combine all datasets
        df = pd.concat([real_df, indian_profiles, international_profiles, demo_profiles], ignore_index=True)
        df = df.fillna("")
        df["cleaned_destination"] = df["Destination"].apply(clean_destination)

        print(f"🔍 Filter: {profile.filter_type}, Destination: {profile.destination}, Date: {profile.travel_date}")
        print(f"📊 Total profiles available: {len(df)}")

        # Apply filters
        if profile.filter_type == "destination":
            # SAME DESTINATION ONLY
            user_dest_clean = clean_destination(profile.destination)
            filtered_df = df[df["cleaned_destination"] == user_dest_clean]
            if len(filtered_df) == 0:
                filtered_df = df[df["cleaned_destination"].str.contains(user_dest_clean, na=False)]
            print(f"📍 Destination matches: {len(filtered_df)}")

        elif profile.filter_type == "dates":
            # SAME DATE + SAME DESTINATION
            if profile.travel_date:
                user_dest_clean = clean_destination(profile.destination)
                
                # First filter by destination
                destination_matches = df[df["cleaned_destination"] == user_dest_clean]
                if len(destination_matches) == 0:
                    destination_matches = df[df["cleaned_destination"].str.contains(user_dest_clean, na=False)]
                
                # Then filter by date
                def date_matches(row):
                    try:
                        if pd.isna(row.get('Start date')) or not row.get('Start date'):
                            return False
                        start_date = pd.to_datetime(row['Start date']).date()
                        user_date = pd.to_datetime(profile.travel_date).date()
                        return start_date == user_date
                    except:
                        return False
                
                filtered_df = destination_matches[destination_matches.apply(date_matches, axis=1)]
                print(f"📅 Date+Destination matches: {len(filtered_df)}")
            else:
                filtered_df = df

        else:  # "all" - ALL USERS
            filtered_df = df
            print(f"🌟 All matches: {len(filtered_df)}")

        # If no matches, return empty
        if len(filtered_df) == 0:
            print("❌ No matches found")
            return {"matches": []}

        # IMPROVED COMPATIBILITY CALCULATION
        def calculate_compatibility(row):
            score = 0
            max_score = 100
            
            # 1. Destination Match (40 points)
            user_dest_clean = clean_destination(profile.destination)
            row_dest_clean = clean_destination(row['Destination'])
            
            if row_dest_clean == user_dest_clean:
                score += 40  # Exact match
            elif user_dest_clean in row_dest_clean or row_dest_clean in user_dest_clean:
                score += 30  # Contains match
            elif any(term in row_dest_clean for term in user_dest_clean.split()):
                score += 20  # Partial match
            else:
                score += 10   # Different destination
            
            # 2. Travel Style Match (30 points)
            user_style = profile.travel_style.lower()
            row_style = str(row.get('Travel style', 'Adventurer')).lower()
            
            if user_style == row_style:
                score += 30
            elif user_style in row_style or row_style in user_style:
                score += 25
            elif any(term in row_style for term in user_style.split()):
                score += 20
            else:
                score += 15
            
            # 3. Interests/Hobbies Match (30 points)
            user_hobbies = set([h.strip().lower() for h in profile.hobbies.split(',')])
            row_interests_str = str(row.get('Interests', 'Travel, Culture, Food'))
            row_interests = set([i.strip().lower() for i in row_interests_str.replace('[', '').replace(']', '').replace("'", "").split(',')])
            
            # Clean interests
            user_hobbies = {hobby for hobby in user_hobbies if hobby}
            row_interests = {interest for interest in row_interests if interest}
            
            common_interests = user_hobbies.intersection(row_interests)
            if common_interests:
                interest_score = min(30, len(common_interests) * 10)
                score += interest_score
            else:
                score += 10
            
            # Ensure score is within bounds
            return min(score, max_score)

        # Apply compatibility calculation
        filtered_df = filtered_df.copy()
        filtered_df["compatibility"] = filtered_df.apply(calculate_compatibility, axis=1)
        
        # Add some random variation for natural distribution (0-5 points)
        filtered_df["compatibility"] = filtered_df["compatibility"] + np.random.randint(0, 6, len(filtered_df))
        filtered_df["compatibility"] = filtered_df["compatibility"].clip(upper=100)
        
        # Sort by compatibility (descending) - Highest first
        filtered_df = filtered_df.sort_values("compatibility", ascending=False)

        # Get top matches
        if profile.filter_type == "all":
            top_matches = filtered_df.head(20)
        else:
            top_matches = filtered_df.head(10)

        # Prepare result
        result = []
        for _, match in top_matches.iterrows():
            # Get traveler name with fallback
            traveler_name = match.get("Traveler name", "Unknown")
            if not traveler_name or traveler_name == "Unknown":
                # Generate a name from nationality
                nationality = match.get("Traveler nationality", "Traveler")
                traveler_name = f"{nationality} Traveler"
            
            result.append({
                "Traveler name": traveler_name,
                "Destination": match.get("Destination", ""),
                "Traveler nationality": match.get("Traveler nationality", ""),
                "Accommodation type": match.get("Accommodation type", ""),
                "Transportation type": match.get("Transportation type", ""),
                "Travel style": match.get("Travel style", "Adventurer"),
                "Traveler age": match.get("Traveler age", 25),
                "Start date": match.get("Start date", ""),
                "End date": match.get("End date", ""),
                "Interests": match.get("Interests", "Travel, Culture, Food"),
                "compatibility": int(match.get("compatibility", 50))
            })

        # Apply final compatibility fix
        result = fix_compatibility_scores(result)

        print(f"✅ Returning {len(result)} matches with proper distribution")
        compatibilities = [r['compatibility'] for r in result]
        if compatibilities:
            print(f"📊 Compatibility range: {min(compatibilities)}% - {max(compatibilities)}%")
            print(f"📈 Average compatibility: {sum(compatibilities)/len(compatibilities):.1f}%")
        
        return {"matches": result}

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"matches": []}

def generate_indian_profiles():
    """Generate Indian travel profiles"""
    indian_destinations = ["Goa, India", "Kerala, India", "Rajasthan, India", "Himachal Pradesh, India"]
    indian_names = ["Aarav Sharma", "Priya Patel", "Rohan Singh", "Ananya Gupta"]
    
    profiles = []
    for i in range(20):
        start_days = random.randint(1, 60)
        start_date = (datetime.now() + timedelta(days=start_days)).strftime("%Y-%m-%d")
        
        profile = {
            "Trip ID": 1000 + i,
            "Destination": random.choice(indian_destinations),
            "Start date": start_date,
            "Traveler name": random.choice(indian_names),
            "Traveler age": random.randint(22, 35),
            "Traveler nationality": "Indian",
            "Accommodation type": random.choice(["Hotel", "Resort", "Airbnb"]),
            "Transportation type": random.choice(["Flight", "Train"]),
            "Travel style": random.choice(["Adventurer", "Cultural Explorer", "Relaxed Traveler"]),
            "Interests": random.choice(["Beaches, Photography", "Temples, Culture", "Food, Shopping"])
        }
        profiles.append(profile)
    
    return pd.DataFrame(profiles)

def generate_international_profiles():
    """Generate International travel profiles"""
    international_destinations = ["Paris, France", "Tokyo, Japan", "London, UK", "New York, USA", "Sydney, Australia"]
    international_names = ["Michael Brown", "Sophie Turner", "David Lee", "Emma Wilson"]
    
    profiles = []
    for i in range(20):
        start_days = random.randint(1, 60)
        start_date = (datetime.now() + timedelta(days=start_days)).strftime("%Y-%m-%d")
        
        profile = {
            "Trip ID": 2000 + i,
            "Destination": random.choice(international_destinations),
            "Start date": start_date,
            "Traveler name": random.choice(international_names),
            "Traveler age": random.randint(25, 45),
            "Traveler nationality": random.choice(["American", "British", "Canadian", "Australian"]),
            "Accommodation type": random.choice(["Hotel", "Airbnb", "Resort"]),
            "Transportation type": random.choice(["Flight", "Train"]),
            "Travel style": random.choice(["Adventurer", "Cultural Explorer", "Luxury Traveler"]),
            "Interests": random.choice(["Museums, Art", "Hiking, Nature", "Food, Wine"])
        }
        profiles.append(profile)
    
    return pd.DataFrame(profiles)

def generate_demo_profiles():
    """Add specific demo profiles for presentation"""
    demo_profiles = [
        {
            "Trip ID": 3001,
            "Destination": "Marrakech, Morocco",
            "Start date": "2024-08-20",
            "Traveler name": "Fatima Khouri",
            "Traveler age": 26,
            "Traveler nationality": "Moroccan",
            "Accommodation type": "Riad",
            "Transportation type": "Flight",
            "Travel style": "Adventurer",
            "Interests": "Cultural Exploration, Local Cuisine, Photography"
        },
        {
            "Trip ID": 3002,
            "Destination": "Edinburgh, Scotland", 
            "Start date": "2024-09-05",
            "Traveler name": "James MacKenzie",
            "Traveler age": 32,
            "Traveler nationality": "Scottish",
            "Accommodation type": "Hotel",
            "Transportation type": "Train",
            "Travel style": "Adventurer",
            "Interests": "Hiking, History, Whiskey Tasting"
        },
        {
            "Trip ID": 3003,
            "Destination": "Bali, Indonesia",
            "Start date": "2024-08-15", 
            "Traveler name": "Michael Chang",
            "Traveler age": 28,
            "Traveler nationality": "Chinese",
            "Accommodation type": "Resort",
            "Transportation type": "Flight",
            "Travel style": "Adventurer", 
            "Interests": "Beaches, Yoga, Local Culture"
        }
    ]
    return pd.DataFrame(demo_profiles)

# WebSocket and other endpoints...
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            save_message(user_id, message_data["receiver_id"], message_data["message"])
            await manager.send_personal_message(
                json.dumps({
                    "type": "message",
                    "sender_id": user_id,
                    "message": message_data["message"],
                    "timestamp": datetime.now().isoformat()
                }),
                message_data["receiver_id"]
            )
    except WebSocketDisconnect:
        manager.disconnect(user_id)

@app.get("/chat/history/{user1_id}/{user2_id}")
def get_chat_history_endpoint(user1_id: str, user2_id: str):
    return {"chat_history": get_chat_history(user1_id, user2_id)}

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "message": "Travel Buddy ML API is running"
    }

# Test matching endpoint
@app.post("/test-match")
def test_matching():
    """Test endpoint to verify matching works"""
    test_profile = Profile(
        destination="Bali, Indonesia",
        travel_style="Adventurer", 
        hobbies="Beaches, Yoga, Photography",
        filter_type="all"
    )
    
    result = match_profiles(test_profile)
    return {
        "test_profile": test_profile.dict(),
        "matches_count": len(result["matches"]),
        "sample_matches": result["matches"][:3] if result["matches"] else []
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)