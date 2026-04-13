import csv
import random

def generate_dataset(filename="data.csv"):
    categories = {
        "electricity": [
            "Power went out in my entire neighborhood 2 hours ago.",
            "There is a loose wire hanging from the pole outside my house.",
            "My electricity bill is unusually high this month.",
            "Voltage fluctuations are damaging my appliances.",
            "Transformer blew up on main street.",
            "Light nahi aa rahi hai subah se.",
            "Power cut since morning, please fix.",
            "Bhai light kab aayegi? Bahut garmi hai.",
            "Street lights are not working in sector 4.",
            "Meter is faulty, showing erratic readings.",
            "No power supply in our building.",
            "Current chala gaya aur abhi tak nahi aaya.",
            "Frequent power trips during the night.",
            "Wire sparking at the junction box.",
            "Please restore electricity ASAP."
        ],
        "water": [
            "No drinking water supply since yesterday.",
            "Pipes are leaking on the main road.",
            "Water pressure is extremely low.",
            "Yellow dirty water is coming from the tap.",
            "Pani nahi aa raha hai.",
            "Sewage water mixed with drinking water supply.",
            "Water bill is incorrect.",
            "Drainage is overflowing onto the street.",
            "Bhai pani ki tanki khali ho gayi, supply kab chalu hogi?",
            "Tap water smells terrible today.",
            "Pipe leak hone ki wajah se pani waste ho raha hai.",
            "Water meter reading is faulty.",
            "No municipal water for 3 days.",
            "Please send a water tanker immediately.",
            "Ganda pani aa raha hai nalo mein."
        ],
        "internet": [
            "Wifi slow hai bro.",
            "Internet connection is completely dead.",
            "Ping is too high for gaming.",
            "Repeated disconnections every 5 minutes.",
            "Net nahi chal raha subah se.",
            "Router is getting a red light.",
            "Speed is 2 Mbps instead of 100 Mbps.",
            "Fiber cable cut by municipal workers.",
            "Internet down ho gaya hai wapas.",
            "Data balance wrongly deducted.",
            "Unable to connect to the broadband network.",
            "WiFi signal is very weak.",
            "Latency issues during video calls.",
            "Bhai net kab theek hoga? Work from home hai.",
            "Lineman is not responding to broadband repair request."
        ],
        "road": [
            "Potholes on the main highway are causing accidents.",
            "Road construction abandoned halfway.",
            "Street is completely flooded after minor rain.",
            "Gadde hi gadde hain road par.",
            "No speed breakers near the school zone.",
            "Traffic signal is not working at the intersection.",
            "Road ka kaam kab pura hoga?",
            "Sidewalks are broken and inaccessible.",
            "Manhole cover is missing on the street.",
            "Heavy traffic jam due to unstructured diversions.",
            "Road pe bahut zyada kachra aur patthar hain.",
            "Asphalt is melting during the day.",
            "Divider breakdown near the crossing.",
            "Accident prone area needs immediate repair.",
            "Bhai road condition is pathetic."
        ],
        "garbage": [
            "Garbage truck has not visited our lane for a week.",
            "Trash bins are overflowing in the park.",
            "Kachra wala nahi aaya teen din se.",
            "People are dumping waste in the empty plot.",
            "Foul smell from the uncollected garbage pile.",
            "Sweeper is not cleaning the street properly.",
            "Kachra jama ho gaya hai kone mein.",
            "Please clear the municipal dustbin.",
            "Dead animal rotting near the garbage dump.",
            "No segregation of dry and wet waste by collectors.",
            "Cleaning staff asking for extra money.",
            "Dustbins are broken and spilling over.",
            "Safai karamchari is absent everyday.",
            "Bhai society ke bahar pura kachra pada hai.",
            "Illegal dumping site created overnight."
        ]
    }

    # Generate more synthetic combinations using templates to reach ~250 rows
    templates = [
        ("I am facing issues with {}.", ""),
        ("This is a complaint regarding {}.", ""),
        ("Please resolve this: {}.", ""),
        ("Urgent: {}.", ""),
        ("{} Support needed.", ""),
        ("{} yaar.", ""),
        ("Help! {}.", "")
    ]

    all_data = []
    
    # Add base sentences
    for cat, sentences in categories.items():
        for s in sentences:
            all_data.append([s, cat])
            
    # Augment to reach at least 250 rows
    for i in range(200):
        cat = random.choice(list(categories.keys()))
        base_sentence = random.choice(categories[cat])
        template = random.choice(templates)
        # Randomly modify casing or add noise
        noisy_sentence = template[0].format(base_sentence)
        if random.random() > 0.8:
            noisy_sentence = noisy_sentence.lower()
        if random.random() > 0.9:
            noisy_sentence = noisy_sentence.replace(".", "!!!")
            
        all_data.append([noisy_sentence, cat])
        
    random.shuffle(all_data)

    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label"])
        for row in all_data:
            writer.writerow(row)
            
    print(f"Dataset generated successfully with {len(all_data)} rows at {filename}.")

if __name__ == "__main__":
    generate_dataset()
